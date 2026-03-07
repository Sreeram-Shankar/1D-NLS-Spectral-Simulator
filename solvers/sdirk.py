import numpy as np
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

#calls LU factorization if available
def _factorize_linear_system(J):
    if _HAS_SCIPY_LU:
        return ("lu", lu_factor(J))
    return ("dense", J)

#solves the linear system of equations
def _solve_linear_system(factored, rhs):
    kind, data = factored
    if kind == "lu":
        return lu_solve(data, rhs)
    return np.linalg.solve(data, rhs)

#defines the SDIRK step
def step_sdirk(f, t, y, h, A, b, c, jac=None, tol=1e-10, max_iter=12, fd_eps=1e-8, jac_recompute_rate=2, backtrack=True):
    s = len(b)
    n = len(y)
    Y = np.tile(y, (s, 1))
    K = np.zeros((s, n))
    I_n = np.eye(n)

    #iteratives over the stages of the SDIRK
    for i in range(s):
        t_i = t + c[i] * h
        g_known = np.sum(A[i, :i, None] * K[:i], axis=0) if i > 0 else np.zeros(n)

        #builds the residual and Jacobian
        def R(z): return z - y - h * (g_known + A[i, i] * f(t_i, z))
        def J(z):
            Jf = jac(t_i, z) if jac else finite_diff_jac(lambda zz: f(t_i, zz), z, eps=fd_eps)
            return I_n - h * A[i, i] * Jf
        y_i = Y[i].copy()
        r = R(y_i)
        r_norm = np.linalg.norm(r)
        jac_factored = None
        prev_r_norm = None

        #implements a newton solver with Jacobian and LU factorization and reuse
        for it in range(max_iter):
            if r_norm < tol: break

            #refreshes the Jacobian if it is None or if the residual is stagnating
            need_refresh = (jac_factored is None or it % max(1, jac_recompute_rate) == 0 or (prev_r_norm is not None and r_norm > 0.95 * prev_r_norm))
            if need_refresh:
                J_mat = J(y_i)
                jac_factored = _factorize_linear_system(J_mat)
            try:
                dy = _solve_linear_system(jac_factored, -r)
            except np.linalg.LinAlgError:
                J_mat = J(y_i)
                dy = np.linalg.lstsq(J_mat, -r, rcond=None)[0]
                jac_factored = _factorize_linear_system(J_mat)
            y_trial = y_i + dy
            r_trial = R(y_trial)
            r_trial_norm = np.linalg.norm(r_trial)

            #backtracks the step if the residual is too large
            if backtrack and r_trial_norm > r_norm:
                alpha = 0.5
                accepted = False
                for _ in range(6):
                    y_bt = y_i + alpha * dy
                    r_bt = R(y_bt)
                    r_bt_norm = np.linalg.norm(r_bt)
                    if r_bt_norm < r_norm:
                        y_trial, r_trial, r_trial_norm = y_bt, r_bt, r_bt_norm
                        accepted = True
                        break
                    alpha *= 0.5
                if not accepted:
                    jac_factored = None

            prev_r_norm = r_norm
            y_i, r, r_norm = y_trial, r_trial, r_trial_norm
            if np.linalg.norm(dy) < tol * (1.0 + np.linalg.norm(y_i)): break
        Y[i] = y_i
        K[i] = f(t_i, y_i)
    y_next = y + h * np.sum(b[:, None] * K, axis=0)
    return y_next

#defines the SDIRK2 solver
def solve_sdirk2(f, t_span, y0, h, jac=None, tol=1e-10, max_iter=12, fd_eps=1e-8):
    gamma = 1.0 - 1.0/np.sqrt(2.0)
    A = np.array([[gamma, 0.0], [1.0 - gamma, gamma]])
    b = np.array([1.0 - gamma, gamma])
    c = np.array([gamma, 1.0])
    t0, tf = t_span
    N = int(np.ceil((tf - t0)/h))
    t_grid = np.linspace(t0, tf, N+1)
    Y = np.zeros((N+1, len(y0)))
    Y[0] = y0
    for n in range(N):
        Y[n+1] = step_sdirk(
            f, t_grid[n], Y[n], h, A, b, c,
            jac=jac, tol=tol, max_iter=max_iter, fd_eps=fd_eps
        )
    return t_grid, Y

#defines the SDIRK3 solver
def solve_sdirk3(f, t_span, y0, h, jac=None, tol=1e-10, max_iter=12, fd_eps=1e-8):
    gamma = 0.435866521508459
    A = np.array([
        [gamma, 0.0, 0.0],
        [0.2820667395, gamma, 0.0],
        [1.208496649, -0.644363171, gamma]
    ])
    b = np.array([1.208496649, -0.644363171, gamma])
    c = np.array([gamma, 0.7179332605, 1.0])
    t0, tf = t_span
    N = int(np.ceil((tf - t0)/h))
    t_grid = np.linspace(t0, tf, N+1)
    Y = np.zeros((N+1, len(y0)))
    Y[0] = y0
    for n in range(N):
        Y[n+1] = step_sdirk(
            f, t_grid[n], Y[n], h, A, b, c,
            jac=jac, tol=tol, max_iter=max_iter, fd_eps=fd_eps
        )
    return t_grid, Y

#defines the SDIRK4 solver
def solve_sdirk4(f, t_span, y0, h, jac=None, tol=1e-10, max_iter=12, fd_eps=1e-8):
    gamma = 0.572816062482135
    a21 = 0.5 - gamma
    a31 = 2 * gamma
    a32 = 1 - 4 * gamma
    a41 = 2 * gamma
    a42 = 1 - 4 * gamma
    a43 = gamma
    A = np.array([
        [gamma, 0.0, 0.0, 0.0],
        [a21,  gamma, 0.0, 0.0],
        [a31,  a32,  gamma, 0.0],
        [a41,  a42,  a43, gamma]
    ])
    b = np.array([a41, a42, a43, gamma])
    c = np.array([gamma, a21 + gamma, a31 + a32 + gamma, 1.0])

    t0, tf = t_span
    N = int(np.ceil((tf - t0)/h))
    t_grid = np.linspace(t0, tf, N+1)
    Y = np.zeros((N+1, len(y0)))
    Y[0] = y0
    for n in range(N):
        Y[n+1] = step_sdirk(
            f, t_grid[n], Y[n], h, A, b, c,
            jac=jac, tol=tol, max_iter=max_iter, fd_eps=fd_eps
        )
    return t_grid, Y