import numpy as np
from scipy.linalg import lu_factor, lu_solve

#defines the finite difference Jacobian
def finite_diff_jac(fun, t, y, eps=1e-8):
    n = len(y)
    f0 = fun(t, y)
    J = np.zeros((n, n))
    for j in range(n):
        dy = np.zeros(n)
        step = eps * max(1.0, abs(y[j]))
        dy[j] = step
        f1 = fun(t, y + dy)
        J[:, j] = (f1 - f0) / step
    return J

#defines the ROS2 method step
def ros2_step(f, t, y, h, jac=None, jac_cache=None, jac_eps=1e-8, dfdt=None, estimate_dfdt=False, t_eps=1e-8):
    gamma = 1.0 - 1.0 / np.sqrt(2.0)
    n = len(y)

    f0 = f(t, y)
    if jac_cache is not None: Jf = jac_cache
    elif jac is not None:
        Jf = jac(t, y)
    else: Jf = finite_diff_jac(f, t, y, eps=jac_eps)

    A = np.eye(n) - gamma * h * Jf
    lu, piv = lu_factor(A)

    if dfdt is not None: ft = dfdt(t, y)
    elif estimate_dfdt:
        dt = max(t_eps, t_eps * abs(t))
        ft = (f(t + dt, y) - f0) / dt
    else: ft = np.zeros(n)

    rhs1 = f0 + h * gamma * ft
    k1 = lu_solve((lu, piv), rhs1)
    f1 = f(t + h, y + h * k1)
    rhs2 = f1 - 2.0 * gamma * h * (Jf @ k1) - h * gamma * ft
    k2 = lu_solve((lu, piv), rhs2)
    y_next = y + 0.5 * h * (k1 + k2)
    return (y_next,)

#main solver function for the ROS2 method
def solve_ros2(f, t_span, y0, h, jac=None, jac_eps=1e-8, dfdt=None, estimate_dfdt=False, t_eps=1e-8):
    t0, tf = t_span
    y0 = np.asarray(y0, dtype=float)
    N = int(np.ceil((tf - t0) / h))
    t_grid = np.linspace(t0, t0 + N * h, N + 1)
    Y = np.zeros((N + 1, len(y0)))
    Y[0] = y0.copy()
    for n in range(N):
        t = t_grid[n]
        y = Y[n]
        (y_next,) = ros2_step(f, t, y, h, jac=jac, jac_eps=jac_eps, dfdt=dfdt, estimate_dfdt=estimate_dfdt, t_eps=t_eps)
        Y[n + 1] = y_next
    return t_grid, Y