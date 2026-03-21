import numpy as np
from tqdm import tqdm
try:
    import cupy as cp
    _CUPY_AVAILABLE = True
except Exception:
    cp = None
    _CUPY_AVAILABLE = False
import solvers.rk as rk
import solvers.sdirk as sdirk
import solvers.irk as irk
import solvers.linear_multistep as lms
import solvers.rosenbrock2 as rosen
from generation.multistep import get_ab_coeffs, get_am_coeffs
from generation.bdf import bdf_coeffs

#caches generated integrators
_INTEGRATOR_CACHE = {}
_COEFF_CACHE = {}
#caches constant linear Jacobian
_LINEAR_JAC_CACHE = {}

#returns the array module
def get_xp(use_gpu):  return cp if (use_gpu and _CUPY_AVAILABLE) else np

#defines weights for the Yoshida splitting method
def get_yoshida_weights(order):
    if order == 4:
        q = 2.0 ** (1.0 / 3.0)
        w1 = 1.0 / (2.0 - q)
        w2 = -q / (2.0 - q)
        return (w1, w2, w1)
    if order == 6: return (0.0502627644003922, 0.0985536835006498, 0.3149606169276942,-0.4473464826954782, 0.4924263724898759,-0.4251187677976909, 0.2370639139781219,0.1956024886000531, 0.3463581898507269,-0.3627627792543449)
    raise ValueError("splitting_order must be 2, 4, or 6")

#defines a symmetric Strang step
def _strang_step(Y, t, h, linear_step, nonlinear_step, linear_substeps=1, nonlinear_substeps=1):
    for i in range(linear_substeps): Y = linear_step(Y, t + i * (h * 0.5 / linear_substeps), h * 0.5 / linear_substeps)
    t_mid = t + h * 0.5
    for i in range(nonlinear_substeps): Y = nonlinear_step(Y, t_mid + i * (h / nonlinear_substeps), h / nonlinear_substeps)
    for i in range(linear_substeps): Y = linear_step(Y, t_mid + i * (h * 0.5 / linear_substeps), h * 0.5 / linear_substeps)
    return Y


#function that performs a single step of the chosen strategy
def single_step(Y, t, dt, strategy, full_step=None, linear_step=None, nonlinear_step=None, ordering="LN", linear_substeps=1, nonlinear_substeps=1, splitting_order=2):
    #handles the full strategy
    if strategy == "full": return full_step(Y, t, dt)

    #handles the lie splitting strategy for linear and nonlinear steps
    elif strategy == "lie":
        if ordering == "LN":
            for i in range(linear_substeps): Y = linear_step(Y, t + i * (dt / linear_substeps), dt / linear_substeps)
            for i in range(nonlinear_substeps): Y = nonlinear_step(Y, t + i * (dt / nonlinear_substeps), dt / nonlinear_substeps)
        else:
            for i in range(nonlinear_substeps): Y = nonlinear_step(Y, t + i * (dt / nonlinear_substeps), dt / nonlinear_substeps)
            for i in range(linear_substeps): Y = linear_step(Y, t + i * (dt / linear_substeps), dt / linear_substeps)
        return Y

    #handles the Strang splitting strategy with Yoshida composition
    elif strategy == "strang":
        if splitting_order == 2:
            return _strang_step(Y, t, dt, linear_step, nonlinear_step, linear_substeps, nonlinear_substeps)
        if splitting_order in (4, 6):
            weights = get_yoshida_weights(splitting_order)
            t_cur = t
            for w in weights:
                h_i = w * dt
                Y = _strang_step(Y, t_cur, h_i, linear_step, nonlinear_step, linear_substeps, nonlinear_substeps)
                t_cur += h_i
            return Y
        raise ValueError("splitting_order must be 2, 4, or 6")
    else: raise ValueError("Unknown strategy")

#main runner for the nonlinear schrodinger equation backend
def run_nls_backend(Y0, dt, n_steps, strategy, ordering="LN", linear_substeps=1, nonlinear_substeps=1, splitting_order=2, full_rhs=None, full_family=None, full_order=None, linear_type=None, k_sq=None, linear_rhs=None, linear_family=None, linear_order=None, nonlinear_type=None, kappa=None, nonlinear_rhs=None, nonlinear_family=None, nonlinear_order=None, progress_tracker=None, use_gpu=False):
    #handles the state of the real and complex embedding
    Y0 = np.asarray(Y0)
    if np.iscomplexobj(Y0): raise ValueError("Y0 must be real embedding Y = [u; v]; psi = u + iv")
    Y0 = Y0.astype(float)
    N_state = len(Y0)
    if N_state % 2 != 0: raise ValueError("Y0 length must be even (real embedding u,v for N grid points)")

    #uses GPU only for splitting with exact linear and exact nonlinear 
    actual_use_gpu = use_gpu and _CUPY_AVAILABLE
    if strategy == "full": actual_use_gpu = False
    else: actual_use_gpu = actual_use_gpu and (linear_type == "exact" and nonlinear_type == "exact")
    xp = get_xp(actual_use_gpu)

    #moves state and constants to GPU when using CuPy
    if xp is cp:
        Y0 = xp.asarray(Y0)
        if k_sq is not None: k_sq = xp.asarray(k_sq)
    Y = Y0.copy()

    #selects the step functions for the full strategy
    if strategy == "full":
        if full_rhs is None or full_family is None or full_order is None: raise ValueError("full strategy requires full_rhs, full_family, full_order")
        jac_full = build_jac_full(np.asarray(k_sq) if k_sq is not None else None, kappa) if (k_sq is not None and kappa is not None) else None
        full_step = get_full_step(full_family, full_order, full_rhs, jac=jac_full)
        linear_step = nonlinear_step = None
        
    #selects the step for the splitting strategies
    else:
        if linear_type is None or nonlinear_type is None: raise ValueError("lie/strang strategy requires linear_type and nonlinear_type")
        jac_linear = build_jac_linear(np.asarray(k_sq) if k_sq is not None else None) if (linear_type != "exact" and k_sq is not None) else None
        jac_nonlinear = build_jac_nonlinear(kappa) if (nonlinear_type != "exact" and kappa is not None) else None
        linear_step = get_linear_step(linear_type, k_sq=k_sq, linear_rhs=linear_rhs, family=linear_family, order=linear_order, jac=jac_linear, xp=xp)
        nonlinear_step = get_nonlinear_step(nonlinear_type, kappa=kappa, nonlinear_rhs=nonlinear_rhs, family=nonlinear_family, order=nonlinear_order, jac=jac_nonlinear, xp=xp)
        full_step = None

    #configures the progress tracker if one is provided
    if progress_tracker is not None:
        progress_tracker.total_steps = int(n_steps)
        progress_tracker.current_step = 0

    #runs the simulation across the time steps
    t_grid = np.linspace(0.0, n_steps * dt, n_steps + 1)
    Y_traj = xp.zeros((n_steps + 1, N_state), dtype=float)
    Y_traj[0, :] = Y
    t = 0.0
    iterator = range(n_steps)
    if progress_tracker is None: iterator = tqdm(iterator, desc="NLS steps", unit="step")
    for n in iterator:
        Y = single_step(Y, t, dt, strategy, full_step=full_step, linear_step=linear_step, nonlinear_step=nonlinear_step, ordering=ordering, linear_substeps=linear_substeps, nonlinear_substeps=nonlinear_substeps, splitting_order=splitting_order)
        Y_traj[n + 1, :] = Y
        t += dt
        if progress_tracker is not None: progress_tracker.current_step = n + 1
    #returns trajectory 
    if xp is cp:
        Y_traj = Y_traj.get()
    return t_grid, Y_traj

#converts a single state vector from real embedding to complex psi
def Y_to_psi(Y):
    N = len(Y) // 2
    return Y[:N] + 1j * Y[N:]

#converts a trajectory matrix from real embedding to complex
def trajectory_to_psi(Y_traj):
    N = Y_traj.shape[1] // 2
    return Y_traj[:, :N] + 1j * Y_traj[:, N:]

#builds the Jacobian for the linaer nls step
def build_jac_linear(k_sq):
    k_sq = np.asarray(k_sq)
    N = len(k_sq)
    cache_key = (N, float(np.sum(k_sq)))
    if cache_key not in _LINEAR_JAC_CACHE:
        J_lin = np.zeros((2 * N, 2 * N), dtype=float)
        for j in range(N):
            e_j = np.zeros(N); e_j[j] = 1.0
            u_hat = np.fft.fft(e_j)
            w = np.fft.ifft(k_sq * u_hat)
            J_lin[:N, j] = -np.imag(w); J_lin[N:, j] = np.real(w)
        for j in range(N):
            e_j = np.zeros(N); e_j[j] = 1.0
            v_hat = np.fft.fft(e_j)
            w = np.fft.ifft(k_sq * v_hat)
            J_lin[:N, N + j] = np.real(w); J_lin[N:, N + j] = -np.imag(w)
        _LINEAR_JAC_CACHE[cache_key] = J_lin
    J_const = _LINEAR_JAC_CACHE[cache_key]
    def jac_linear(t, Y): return J_const.copy()
    return jac_linear

#builds the Jacobian for the nonlinear nls step
def build_jac_nonlinear(kappa):
    def jac_nonlinear(t, Y):
        n = len(Y) // 2
        u, v = Y[:n], Y[n:]
        r2 = u * u + v * v
        du_du = 2.0 * kappa * u * v
        du_dv = kappa * (u * u + 3.0 * v * v)
        dv_du = -kappa * (3.0 * u * u + v * v)
        dv_dv = -2.0 * kappa * u * v
        J = np.zeros((2 * n, 2 * n), dtype=float)
        for i in range(n): J[i, i] = du_du[i]; J[i, n + i] = du_dv[i]; J[n + i, i] = dv_du[i]; J[n + i, n + i] = dv_dv[i]
        return J
    return jac_nonlinear

#builds the Jacobian for the full nls step
def build_jac_full(k_sq, kappa):
    jac_lin = build_jac_linear(k_sq)
    jac_nl = build_jac_nonlinear(kappa)
    def jac_full(t, Y): return jac_lin(t, Y) + jac_nl(t, Y)
    return jac_full

#function that selects the integrator step 
def get_integrator_step(family: str, order: int):
    family = family.lower().strip()
    key = (family, order)

    #gets the explicit runge-kutta step methods
    if family == "explicit runge-kutta":
        if key not in _INTEGRATOR_CACHE:
            rk_steps = {1: rk.step_rk1, 2: rk.step_rk2, 3: rk.step_rk3, 4: rk.step_rk4, 5: rk.step_rk5, 6: rk.step_rk6, 7: rk.step_rk7}
            if order not in rk_steps: raise ValueError(f"Explicit Runge-Kutta order must be between 1 and 7, got {order}")
            _INTEGRATOR_CACHE[key] = rk_steps[order]
        return _INTEGRATOR_CACHE[key]

    #gets the singly diagonal implicit runge kutta step methods
    elif family == "sdirk":
        if key not in _INTEGRATOR_CACHE:
            if order == 2:
                gamma = 1.0 - 1.0 / np.sqrt(2.0)
                A = np.array([[gamma, 0.0], [1.0 - gamma, gamma]])
                b = np.array([1.0 - gamma, gamma])
                c = np.array([gamma, 1.0])
                _INTEGRATOR_CACHE[key] = lambda f, t, y, h, jac=None: sdirk.step_sdirk(f, t, y, h, A, b, c, jac=jac)
            elif order == 3:
                gamma = 0.435866521508459
                A = np.array([[gamma, 0.0, 0.0], [0.2820667395, gamma, 0.0], [1.208496649, -0.644363171, gamma]])
                b = np.array([1.208496649, -0.644363171, gamma])
                c = np.array([gamma, 0.7179332605, 1.0])
                _INTEGRATOR_CACHE[key] = lambda f, t, y, h, jac=None: sdirk.step_sdirk(f, t, y, h, A, b, c, jac=jac)
            elif order == 4:
                gamma = 0.572816062482135
                a21 = 0.5 - gamma
                a31 = 2 * gamma
                a32 = 1 - 4 * gamma
                a41 = 2 * gamma
                a42 = 1 - 4 * gamma
                a43 = gamma
                A = np.array([[gamma, 0.0, 0.0, 0.0], [a21, gamma, 0.0, 0.0], [a31, a32, gamma, 0.0], [a41, a42, a43, gamma]])
                b = np.array([a41, a42, a43, gamma])
                c = np.array([gamma, a21 + gamma, a31 + a32 + gamma, 1.0])
                _INTEGRATOR_CACHE[key] = lambda f, t, y, h, jac=None: sdirk.step_sdirk(f, t, y, h, A, b, c, jac=jac)
            else: raise ValueError(f"SDIRK order must be between 2 and 4, got {order}")
        return _INTEGRATOR_CACHE[key]
    
    #gets the collocation fully implicit irk step methods
    elif family == "gauss-legendre":
        if key not in _INTEGRATOR_CACHE:
            A, b, c = irk.get_tableau("gauss", order)
            _INTEGRATOR_CACHE[key] = lambda f, t, y, h, jac=None: irk.step_collocation(f, t, y, h, A, b, c, jac=jac)
        return _INTEGRATOR_CACHE[key]
    elif family == "radauiia":
        if key not in _INTEGRATOR_CACHE:
            A, b, c = irk.get_tableau("radau", order)
            _INTEGRATOR_CACHE[key] = lambda f, t, y, h, jac=None: irk.step_collocation(f, t, y, h, A, b, c, jac=jac)
        return _INTEGRATOR_CACHE[key]
    elif family == "lobattoiiia":
        if key not in _INTEGRATOR_CACHE:
            A, b, c = irk.get_tableau("lobatto", order)
            _INTEGRATOR_CACHE[key] = lambda f, t, y, h, jac=None: irk.step_collocation(f, t, y, h, A, b, c, jac=jac)
        return _INTEGRATOR_CACHE[key]

    #handles the Rosenbrock2 method
    elif family == "rosenbrock":
        if order != 2: raise ValueError(f"Rosenbrock family currently supports only order 2 (ROS2), got {order}")
        if key not in _INTEGRATOR_CACHE: _INTEGRATOR_CACHE[key] = lambda f, t, y, h, jac=None: rosen.ros2_step(f, t, y, h, jac=jac)[0]
        return _INTEGRATOR_CACHE[key]
    
    #handles the Adams-Bashforth step and lmm history management
    elif family == "adams-bashforth":
        if key not in _COEFF_CACHE:
            _COEFF_CACHE[key] = np.asarray(get_ab_coeffs(order), dtype=float)
        b = _COEFF_CACHE[key]
        k = len(b)
        F_history = []
        last_t = None
        def ab_step(f, t, y, h, jac=None):
            nonlocal F_history, last_t
            if last_t is not None and t < last_t: F_history = []; last_t = None
            if len(F_history) == 0:
                F_history = [f(t, y)]
                y_current = y
                t_current = t
                for i in range(k - 1): y_current = lms._rk4_step(f, t_current, y_current, h); t_current += h; F_history.append(f(t_current, y_current))
                last_t = t_current
                return y_current
            acc = 0.0
            for j in range(k): acc += b[j] * F_history[-(j + 1)]
            y_next = y + h * acc
            F_history.append(f(t + h, y_next))
            if len(F_history) > k: F_history.pop(0)
            last_t = t + h
            return y_next
        return ab_step

    #handles the Adams-Moulton step and lmm history management
    elif family == "adams-moulton":
        if key not in _COEFF_CACHE:
            _COEFF_CACHE[key] = np.asarray(get_am_coeffs(order), dtype=float)
        b = _COEFF_CACHE[key]
        k = len(b)
        b0 = b[0]
        F_history = []
        last_t = None
        def am_step(f, t, y, h, jac=None):
            nonlocal F_history, last_t
            if last_t is not None and t < last_t: F_history = []; last_t = None
            if len(F_history) == 0:
                F_history = [f(t, y)]
                y_current = y
                t_current = t
                for i in range(k - 1): y_current = lms._rk4_step(f, t_current, y_current, h); t_current += h; F_history.append(f(t_current, y_current))
                last_t = t_current
                return y_current
            t_next = t + h
            known = sum(b[j] * F_history[-(j)] for j in range(1, k))
            def R(y_next): return y_next - y - h * (b0 * f(t_next, y_next) + known)
            def J(y_next): return np.eye(len(y)) - h * b0 * (jac(t_next, y_next) if jac is not None else lms.finite_diff_jac(lambda z: f(t_next, z), y_next))
            y_next = lms.modified_newton_solve(R, J, y, tol=1e-10, max_iter=12, jac_recompute_rate=2, backtrack=True)
            F_history.append(f(t_next, y_next))
            if len(F_history) > k: F_history.pop(0)
            last_t = t_next
            return y_next
        return am_step

    #handles the Adams-Bashforth-Moulton predictor-corrector scheme and lmm history management
    elif family in ("adams-bashforth-moulton", "abm"):
        if key not in _COEFF_CACHE:
            b_ab = np.asarray(get_ab_coeffs(order), dtype=float)
            b_am = np.asarray(get_am_coeffs(order), dtype=float)
            _COEFF_CACHE[key] = (b_ab, b_am)
        b_ab, b_am = _COEFF_CACHE[key]
        k = len(b_ab)
        if len(b_am) != k: raise ValueError(f"ABM requires same order for AB and AM, got {len(b_ab)} and {len(b_am)}")
        b0 = b_am[0]
        F_history = []
        last_t = None
        def abm_step(f, t, y, h, jac=None):
            nonlocal F_history, last_t
            if last_t is not None and t < last_t: F_history = []; last_t = None
            if len(F_history) == 0:
                F_history = [f(t, y)]
                y_current = y
                t_current = t
                for i in range(k - 1): y_current = lms._rk4_step(f, t_current, y_current, h); t_current += h; F_history.append(f(t_current, y_current))
                last_t = t_current
                return y_current

            #takes the predictor step with Adams-Bashforth
            acc = 0.0
            for j in range(k): acc += b_ab[j] * F_history[-(j + 1)]
            y_pred = y + h * acc

            #takes the corrector step with Adams-Moulton
            t_next = t + h
            known = sum(b_am[j] * F_history[-j] for j in range(1, k))
            def R(y_next): return y_next - y - h * (b0 * f(t_next, y_next) + known)
            def J(y_next): return np.eye(len(y)) - h * b0 * (jac(t_next, y_next) if jac is not None else lms.finite_diff_jac(lambda z: f(t_next, z), y_next))
            y_next = lms.modified_newton_solve(R, J, y_pred, tol=1e-10, max_iter=12, jac_recompute_rate=2, backtrack=True)
            F_history.append(f(t_next, y_next))
            if len(F_history) > k: F_history.pop(0)
            last_t = t_next
            return y_next
        return abm_step

    #handles the backward differentiation formula step and lmm history management
    elif family == "bdf":
        if key not in _COEFF_CACHE:
            a_raw, beta0_raw = bdf_coeffs(order)
            _COEFF_CACHE[key] = (np.asarray([float(val) for val in a_raw], dtype=float), float(beta0_raw))
        a, beta0 = _COEFF_CACHE[key]
        Y_history = []
        last_t = None
        def bdf_step(f, t, y, h, jac=None):
            nonlocal Y_history, last_t
            if last_t is not None and t < last_t: Y_history = []; last_t = None
            if len(Y_history) == 0:
                Y_history = [y]
                y_current = y
                t_current = t
                for i in range(order - 1): y_current = lms._rk4_step(f, t_current, y_current, h); t_current += h; Y_history.append(y_current)
                last_t = t_current
                return y_current
            t_next = t + h
            known = np.zeros_like(y, dtype=float)
            for j in range(1, order + 1): known += a[j] * Y_history[-(j)]
            def R(y_next): return a[0] * y_next + known - beta0 * h * f(t_next, y_next)
            def J(y_next): return a[0] * np.eye(len(y)) - beta0 * h * (jac(t_next, y_next) if jac is not None else lms.finite_diff_jac(lambda z: f(t_next, z), y_next))
            y_next = lms.modified_newton_solve(R, J, y, tol=1e-10, max_iter=12, jac_recompute_rate=2, backtrack=True)
            Y_history.append(y_next)
            if len(Y_history) > order: Y_history.pop(0)
            last_t = t_next
            return y_next
        return bdf_step
    else: raise ValueError(f"Unknown integrator family: {family}")


#builds the step function from the integrator and rhs
def step_from_integrator(family: str, order: int, rhs, jac=None):
    raw_step = get_integrator_step(family, order)
    family_lower = family.lower().strip()
    pass_jac = jac is not None and family_lower not in {"explicit runge-kutta", "adams-bashforth"}
    if pass_jac:
        return lambda Y, t, dt: raw_step(rhs, t, Y, dt, jac=jac)
    return lambda Y, t, dt: raw_step(rhs, t, Y, dt)

#returns the exact linear step
def exact_linear_step(k_sq, xp=np):
    def step(Y, t, dt):
        N = len(Y) // 2
        u, v = Y[:N], Y[N:]
        psi = u + 1j * v
        psi_hat = xp.fft.fft(psi)
        psi_hat = xp.exp(-1j * k_sq * dt) * psi_hat
        psi = xp.fft.ifft(psi_hat)
        return xp.concatenate([xp.real(psi), xp.imag(psi)])
    return step

#returns the exact nonlinear step
def exact_nonlinear_step(kappa, xp=np):
    def step(Y, t, dt):
        N = len(Y) // 2
        u, v = Y[:N], Y[N:]
        psi = u + 1j * v
        phase = xp.exp(-1j * kappa * dt * (u * u + v * v))
        psi = phase * psi
        return xp.concatenate([xp.real(psi), xp.imag(psi)])
    return step

#returns the linear step
def get_linear_step(linear_type: str, k_sq=None, linear_rhs=None, family=None, order=None, jac=None, xp=np):
    linear_type = linear_type.lower().strip()
    if linear_type == "exact":
        if k_sq is None: raise ValueError("exact linear step requires k_sq")
        return exact_linear_step(k_sq, xp=xp)
    if linear_rhs is None or family is None or order is None: raise ValueError("numeric linear step requires linear_rhs, family, and order")
    return step_from_integrator(family, order, linear_rhs, jac=jac)

#returns the nonlinear step
def get_nonlinear_step(nonlinear_type: str, kappa=None, nonlinear_rhs=None, family=None, order=None, jac=None, xp=np):
    nonlinear_type = nonlinear_type.lower().strip()
    if nonlinear_type == "exact":
        if kappa is None: raise ValueError("exact nonlinear step requires kappa")
        return exact_nonlinear_step(kappa, xp=xp)
    if nonlinear_rhs is None or family is None or order is None: raise ValueError("numeric nonlinear step requires nonlinear_rhs, family, and order")
    return step_from_integrator(family, order, nonlinear_rhs, jac=jac)

#returns the full step; jac optional for implicit
def get_full_step(family: str, order: int, full_rhs, jac=None): return step_from_integrator(family, order, full_rhs, jac=jac)
