import numpy as np
import mpmath as mp
mp.dps = 200
from generation.gauss_legendre import build_gauss_legendre_irk
from generation.radau import build_radau_irk
from generation.lobatto import build_lobatto_IIIC_irk
try:
    from scipy.linalg import lu_factor, lu_solve
    _HAS_SCIPY_LU = True
except Exception:
    _HAS_SCIPY_LU = False

#defines the finite difference Jacobian
def finite_diff_jac(fun, x, eps=1e-8):
    n = len(x)
    f0 = fun(x)
    J = np.zeros((n, n))
    for j in range(n):
        dx = np.zeros(n)
        step = eps * max(1.0, abs(x[j]))
        dx[j] = step
        f1 = fun(x + dx)
        J[:, j] = (f1 - f0) / step
    return J

#solves the nonlinear system of equations
def newton_solve(residual, y0, jac=None, tol=1e-10, max_iter=12):
    y = y0.copy()
    for _ in range(max_iter):
        r = residual(y)
        if np.linalg.norm(r) < tol:
            return y
        J = jac(y) if jac else finite_diff_jac(residual, y)
        dy = np.linalg.solve(J, -r)
        y += dy
        if np.linalg.norm(dy) < tol:
            break
    return y

def _factorize_linear_system(J):
    if _HAS_SCIPY_LU:
        return ("lu", lu_factor(J))
    return ("dense", J)

def _solve_linear_system(factored, rhs):
    kind, data = factored
    if kind == "lu":
        return lu_solve(data, rhs)
    return np.linalg.solve(data, rhs)

#caches generated tableaux by (family, s)
_TABLEAU_CACHE = {}

#loads the Butcher tableau from the generators (cached)
def get_tableau(family, s):
    family = family.lower()
    key = (family, s)
    if key not in _TABLEAU_CACHE:
        if family == "gauss":
            A, b, c = build_gauss_legendre_irk(s)
        elif family == "radau":
            A, b, c = build_radau_irk(s)
        elif family == "lobatto":
            A, b, c = build_lobatto_IIIC_irk(s)
        else: raise ValueError(f"Unknown family '{family}', must be 'gauss', 'radau', or 'lobatto'.")
        A = np.array([[float(A[i][j]) for j in range(s)] for i in range(s)])
        b = np.array([float(b[i]) for i in range(s)])
        c = np.array([float(c[i]) for i in range(s)])
        _TABLEAU_CACHE[key] = (A, b, c)
    return _TABLEAU_CACHE[key]


#defines the IRK collocation step
def step_collocation(f, t, y, h, A, b, c, jac=None, tol=1e-10, max_iter=12, fd_eps=1e-8, jac_recompute_rate=2, backtrack=True):
    s = len(b)
    n = len(y)
    Y = np.tile(y, (s, 1))
    t_nodes = t + c * h
    I_n = np.eye(n)

    #builds the residual
    def residual(z_flat):
        Z = z_flat.reshape(s, n)
        R = np.zeros_like(Z)
        for i in range(s):
            acc = np.zeros(n)
            for j in range(s):
                acc += A[i, j] * f(t_nodes[j], Z[j])
            R[i] = Z[i] - y - h * acc
        return R.ravel()

    #builds the Jacobian
    def jacobian(z_flat):
        Z = z_flat.reshape(s, n)
        J_full = np.zeros((s * n, s * n))
        for j in range(s):
            Jf_j = jac(t_nodes[j], Z[j]) if jac else finite_diff_jac(lambda z: f(t_nodes[j], z), Z[j], eps=fd_eps)
            for i in range(s):
                block = -h * A[i, j] * Jf_j
                if i == j:
                    block = block + I_n
                row = slice(i * n, (i + 1) * n)
                col = slice(j * n, (j + 1) * n)
                J_full[row, col] = block
        return J_full

    z_star = Y.ravel()
    r = residual(z_star)
    r_norm = np.linalg.norm(r)
    jac_factored = None
    prev_r_norm = None

    #implements a newton solver with Jacobian and LU factorization and reuse
    for it in range(max_iter):
        if r_norm < tol: break

        #refreshes the Jacobian if it is None or if the residual is stagnating
        need_refresh = (jac_factored is None or it % max(1, jac_recompute_rate) == 0 or (prev_r_norm is not None and r_norm > 0.95 * prev_r_norm))
        if need_refresh:
            J_full = jacobian(z_star)
            jac_factored = _factorize_linear_system(J_full)
        try:
            dz = _solve_linear_system(jac_factored, -r)
        except np.linalg.LinAlgError:
            J_full = jacobian(z_star)
            dz = np.linalg.lstsq(J_full, -r, rcond=None)[0]
            jac_factored = _factorize_linear_system(J_full)
        z_trial = z_star + dz
        r_trial = residual(z_trial)
        r_trial_norm = np.linalg.norm(r_trial)

        #backtracks the step if the residual is too large
        if backtrack and r_trial_norm > r_norm:
            alpha = 0.5
            accepted = False
            for _ in range(6):
                z_bt = z_star + alpha * dz
                r_bt = residual(z_bt)
                r_bt_norm = np.linalg.norm(r_bt)
                if r_bt_norm < r_norm:
                    z_trial, r_trial, r_trial_norm = z_bt, r_bt, r_bt_norm
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted: jac_factored = None
        prev_r_norm = r_norm
        z_star, r, r_norm = z_trial, r_trial, r_trial_norm
        if np.linalg.norm(dz) < tol * (1.0 + np.linalg.norm(z_star)): break
    Y = z_star.reshape(s, n)
    K = np.zeros((s, n))
    for i in range(s): K[i] = f(t_nodes[i], Y[i])
    y_next = y + h * np.sum(b[:, None] * K, axis=0)
    return y_next

#main solver for any collocation method
def solve_collocation(f, t_span, y0, h, family="gauss", s=3, jac=None, tol=1e-10, max_iter=12, fd_eps=1e-8):
    A, b, c = get_tableau(family, s)
    t0, tf = t_span
    N = int(np.ceil((tf - t0)/h))
    t_grid = np.linspace(t0, tf, N+1)
    Y = np.zeros((N+1, len(y0)))
    Y[0] = y0
    for n in range(N):
        Y[n+1] = step_collocation(f, t_grid[n], Y[n], h, A, b, c, jac=jac, tol=tol, max_iter=max_iter, fd_eps=fd_eps)
    return t_grid, Y