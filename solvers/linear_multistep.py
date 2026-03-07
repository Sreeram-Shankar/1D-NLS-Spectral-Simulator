import numpy as np
from generation.multistep import get_ab_coeffs, get_am_coeffs
from generation.bdf import bdf_coeffs
import mpmath as mp
try:
    from scipy.linalg import lu_factor, lu_solve
    _HAS_SCIPY_LU = True
except Exception:
    _HAS_SCIPY_LU = False
mp.dps = 200

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

#factorizes the linear system
def _factorize_linear_system(J):
    if _HAS_SCIPY_LU: return ("lu", lu_factor(J))
    return ("dense", J)

#solves the factored linear system
def _solve_linear_system(factored, rhs):
    kind, data = factored
    if kind == "lu": return lu_solve(data, rhs)
    return np.linalg.solve(data, rhs)

#solves nonlinear systems with modified newton and jacobian reuse
def modified_newton_solve(residual, jacobian, y0, tol=1e-10, max_iter=12, jac_recompute_rate=2, backtrack=True):
    y = y0.copy()
    r = residual(y)
    r_norm = np.linalg.norm(r)
    jac_factored = None
    prev_r_norm = None
    for it in range(max_iter):
        if r_norm < tol: break
        need_refresh = jac_factored is None or it % max(1, jac_recompute_rate) == 0 or (prev_r_norm is not None and r_norm > 0.95 * prev_r_norm)
        if need_refresh: jac_factored = _factorize_linear_system(jacobian(y))
        try: dy = _solve_linear_system(jac_factored, -r)
        except np.linalg.LinAlgError:
            J = jacobian(y)
            dy = np.linalg.lstsq(J, -r, rcond=None)[0]
            jac_factored = _factorize_linear_system(J)
        y_trial = y + dy
        r_trial = residual(y_trial)
        r_trial_norm = np.linalg.norm(r_trial)
        if backtrack and r_trial_norm > r_norm:
            alpha = 0.5
            accepted = False
            for _ in range(6):
                y_bt = y + alpha * dy
                r_bt = residual(y_bt)
                r_bt_norm = np.linalg.norm(r_bt)
                if r_bt_norm < r_norm:
                    y_trial, r_trial, r_trial_norm = y_bt, r_bt, r_bt_norm
                    accepted = True
                    break
                alpha *= 0.5
            if not accepted: jac_factored = None
        prev_r_norm = r_norm
        y, r, r_norm = y_trial, r_trial, r_trial_norm
        if np.linalg.norm(dy) < tol * (1.0 + np.linalg.norm(y)): break
    return y

#defines the RK4 step
def _rk4_step(f, t, y, h):
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
    k4 = f(t + h, y + h * k3)
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

#bootstraps the first steps with RK4
def _bootstrap_rk4(f, t_grid, Y, F, start_idx, steps, h):
    for i in range(steps):
        n = start_idx + i
        Y[n + 1] = _rk4_step(f, t_grid[n], Y[n], h)
        F[n + 1] = f(t_grid[n + 1], Y[n + 1])

#defines the AB solver
def solve_ab(f, t_span, y0, h, order=3, prec=500, adaptive=False):
    b = np.asarray(get_ab_coeffs(order, prec=prec, adaptive=adaptive), dtype=float)
    k = len(b)
    t0, tf = t_span
    N = int(np.ceil((tf - t0) / h))
    t_grid = np.linspace(t0, tf, N + 1)
    Y = np.zeros((N + 1, len(y0)))
    F = np.zeros_like(Y)
    Y[0] = y0
    F[0] = f(t_grid[0], Y[0])

    #bootstraps the first steps with RK4
    bootstrap_steps = min(k - 1, N)
    _bootstrap_rk4(f, t_grid, Y, F, 0, bootstrap_steps, h)
    if N <= k - 1: return t_grid, Y

    #main solver loop for AB
    for n in range(k - 1, N):
        acc = 0.0
        for j in range(k):
            acc += b[j] * F[n - j]
        Y[n + 1] = Y[n] + h * acc
        F[n + 1] = f(t_grid[n + 1], Y[n + 1])

    return t_grid, Y

#defines the AM solver
def solve_am(f, t_span, y0, h, order=3, jac=None, tol=1e-10, max_iter=12, fd_eps=1e-8, prec=500, adaptive=False, jac_recompute_rate=2, backtrack=True):
    b = np.asarray(get_am_coeffs(order, prec=prec, adaptive=adaptive), dtype=float)
    k = len(b)
    b0 = b[0]
    t0, tf = t_span
    N = int(np.ceil((tf - t0) / h))
    t_grid = np.linspace(t0, tf, N + 1)
    Y = np.zeros((N + 1, len(y0)))
    F = np.zeros_like(Y)

    Y[0] = y0
    F[0] = f(t_grid[0], Y[0])

    #bootstraps the first steps with RK4
    bootstrap_steps = min(k - 1, N)
    _bootstrap_rk4(f, t_grid, Y, F, 0, bootstrap_steps, h)
    if N <= k - 1: return t_grid, Y

    #main solver loop for AM
    for n in range(k - 1, N):
        t_next = t_grid[n + 1]
        known = sum(b[j] * F[n + 1 - j] for j in range(1, k))

        def R(y_next): return y_next - Y[n] - h * (b0 * f(t_next, y_next) + known)

        def J(y_next):
            Jf = jac(t_next, y_next) if jac else finite_diff_jac(lambda z: f(t_next, z), y_next, eps=fd_eps)
            return np.eye(len(y0)) - h * b0 * Jf

        y_guess = Y[n]
        y_next = modified_newton_solve(R, J, y_guess, tol=tol, max_iter=max_iter, jac_recompute_rate=jac_recompute_rate, backtrack=backtrack)
        Y[n + 1] = y_next
        F[n + 1] = f(t_next, y_next)

    return t_grid, Y

#defines the BDF solver
def solve_bdf(f, t_span, y0, h, order=2, jac=None, tol=1e-10, max_iter=12, fd_eps=1e-8, jac_recompute_rate=2, backtrack=True):
    a, beta0 = bdf_coeffs(order)
    # generator returns mpmath types; cast to float
    a = np.asarray([float(val) for val in a], dtype=float)
    beta0 = float(beta0)
    t0, tf = t_span
    N = int(np.ceil((tf - t0) / h))
    t_grid = np.linspace(t0, tf, N + 1)
    Y = np.zeros((N + 1, len(y0)))

    Y[0] = y0

    #bootstraps the first steps with RK4
    bootstrap_steps = min(order - 1, N)
    F_boot = np.zeros((N + 1, len(y0)))
    F_boot[0] = f(t_grid[0], Y[0])
    _bootstrap_rk4(f, t_grid, Y, F_boot, 0, bootstrap_steps, h)
    if N <= order - 1: return t_grid, Y

    for n in range(order - 1, N):
        t_next = t_grid[n + 1]
        known = np.zeros_like(y0, dtype=float)
        for j in range(1, order + 1):
            known += a[j] * Y[n + 1 - j]

        def R(y_next):
            return a[0] * y_next + known - beta0 * h * f(t_next, y_next)

        def J(y_next):
            Jf = jac(t_next, y_next) if jac else finite_diff_jac(lambda z: f(t_next, z), y_next, eps=fd_eps)
            return a[0] * np.eye(len(y0)) - beta0 * h * Jf

        y_guess = Y[n]
        y_next = modified_newton_solve(R, J, y_guess, tol=tol, max_iter=max_iter, jac_recompute_rate=jac_recompute_rate, backtrack=backtrack)
        Y[n + 1] = y_next
    return t_grid, Y