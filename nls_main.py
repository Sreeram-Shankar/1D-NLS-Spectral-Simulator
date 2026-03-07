
import os
import numpy as np
import yaml
from types import SimpleNamespace
from nls_backend import run_nls_backend, trajectory_to_psi

#coerces strings to numbers
def to_number(x):
    if isinstance(x, (int, float)): return x
    if isinstance(x, str):
        try:
            f = float(x)
            i = int(f)
            return i if f == i else f
        except (ValueError, TypeError): return x
    return x

#recursively coerces numbers inside arrays
def coerce_numbers_in(obj):
    if isinstance(obj, dict):
        return {k: coerce_numbers_in(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [coerce_numbers_in(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(coerce_numbers_in(v) for v in obj)
    return to_number(obj)

#opens a yaml file
def load_yaml(path): return yaml.safe_load(open(path, "r", encoding="utf-8"))

#parses a config dictionary from the yaml file into an object
def parse_nls_config(cfg):
    cfg = coerce_numbers_in(cfg) if isinstance(cfg, dict) else cfg
    required = ("domain", "pde", "initial_condition", "time", "strategy")
    for key in required:
        if key not in cfg: raise ValueError(f"Missing required section: {key}")

    #collects the inputs for the domain
    dom = cfg["domain"]
    L = float(to_number(dom["L"]))
    N = int(to_number(dom["N"]))
    if N <= 0 or L <= 0: raise ValueError("domain: N and L must be positive")
    x = np.linspace(-L, L, N, endpoint=False)
    dx = float(x[1] - x[0]) if N > 1 else 0.0
    k = 2.0 * np.pi * np.fft.fftfreq(N, dx)
    k_sq = np.asarray(k ** 2, dtype=float)
    domain = SimpleNamespace(L=L, N=N, x=x, dx=dx, k=k, k_sq=k_sq)

    #collects the inputs for the pde
    pde = SimpleNamespace(kappa=float(to_number(cfg["pde"]["kappa"])))

    #collects the inputs for the initial condition
    ic = cfg["initial_condition"]
    if "type" not in ic: raise ValueError("initial_condition must have 'type'")
    ic_params = coerce_numbers_in(ic.get("params", {}))
    initial_condition = SimpleNamespace(type=str(ic["type"]).strip().lower(), params=ic_params)

    #collects the inputs for the time
    time_cfg = cfg["time"]
    dt = float(to_number(time_cfg["dt"]))
    if "n_steps" in time_cfg: n_steps = int(to_number(time_cfg["n_steps"]))
    elif "t_final" in time_cfg: t_final = float(to_number(time_cfg["t_final"])); n_steps = int(round(t_final / dt))
    else: raise ValueError("time must have 'n_steps' or 't_final'")
    if dt <= 0 or n_steps < 0: raise ValueError("time: dt must be positive and n_steps non-negative")
    time_ns = SimpleNamespace(dt=dt, n_steps=n_steps)

    #collects the inputs for the strategy
    strat = cfg["strategy"]
    name = str(strat.get("name", "strang")).strip().lower()
    if name not in ("full", "lie", "strang"): raise ValueError(f"strategy.name must be 'full', 'lie', or 'strang', got '{name}'")
    ordering = str(strat.get("ordering", "LN")).strip().upper()
    if ordering not in ("LN", "NL"): ordering = "LN"
    splitting_order = int(to_number(strat.get("splitting_order", 2)))
    if splitting_order not in (2, 4, 6): raise ValueError("strategy.splitting_order must be 2, 4, or 6")
    linear_substeps = max(1, int(to_number(strat.get("linear_substeps", 1))))
    nonlinear_substeps = max(1, int(to_number(strat.get("nonlinear_substeps", 1))))
    strategy = SimpleNamespace(name=name, ordering=ordering, splitting_order=splitting_order, linear_substeps=linear_substeps, nonlinear_substeps=nonlinear_substeps)

    #collects the inputs for the integrator if the strategy is full
    if name == "full":
        integ = cfg.get("integrator") or {}
        family = str(integ.get("family", "")).strip()
        order = int(to_number(integ.get("order", 0)))
        if not family or order < 1: raise ValueError("strategy 'full' requires integrator.family and integrator.order")
        integrator = SimpleNamespace(family=family, order=order)
    else: integrator = None

    #collects the inputs for the linear if the strategy is split
    if name in ("lie", "strang"):
        #collects the inputs for the linear part of the nls
        lin = cfg.get("linear")
        if not lin or "type" not in lin: raise ValueError("strategy lie/strang requires section 'linear' with 'type'")
        linear_type = str(lin["type"]).strip().lower()
        if linear_type not in ("exact", "numeric"): raise ValueError("linear.type must be 'exact' or 'numeric'")
        if linear_type == "numeric":
            linear_family = str(lin.get("family", "")).strip()
            linear_order = int(to_number(lin.get("order", 0)))
            if not linear_family or linear_order < 1: raise ValueError("linear type 'numeric' requires linear.family and linear.order")
        else: linear_family = linear_order = None
        linear = SimpleNamespace(type=linear_type, family=linear_family, order=linear_order)

        #collects the inputs for the nonlinear part of the nls
        nlin = cfg.get("nonlinear")
        if not nlin or "type" not in nlin: raise ValueError("strategy lie/strang requires section 'nonlinear' with 'type'")
        nonlinear_type = str(nlin["type"]).strip().lower()
        if nonlinear_type not in ("exact", "numeric"): raise ValueError("nonlinear.type must be 'exact' or 'numeric'")
        if nonlinear_type == "numeric":
            nonlinear_family = str(nlin.get("family", "")).strip()
            nonlinear_order = int(to_number(nlin.get("order", 0)))
            if not nonlinear_family or nonlinear_order < 1: raise ValueError("nonlinear type 'numeric' requires nonlinear.family and nonlinear.order")
        else: nonlinear_family = nonlinear_order = None
        nonlinear = SimpleNamespace(type=nonlinear_type, family=nonlinear_family, order=nonlinear_order)
    else: linear = nonlinear = None

    #collects the inputs for the reference
    ref_cfg = cfg.get("reference") or {}
    ref_mode = ref_cfg.get("mode", "off")
    if ref_mode is True:ref_mode = "on"
    elif ref_mode is False: ref_mode = "off"
    else:
        ref_mode = str(ref_mode).strip().lower()
        if ref_mode not in ("on", "off"): ref_mode = "off"
    reference = SimpleNamespace(mode=ref_mode)

    #collects the inputs for writing the output
    out = cfg.get("output") or {}
    save_path = out.get("save_path") or ""
    if save_path and isinstance(save_path, str): save_path = save_path.strip()
    overwrite = bool(out.get("overwrite", True))
    output = SimpleNamespace(save_path=save_path, overwrite=overwrite)

    #collects optional compute options
    comp = cfg.get("compute") or {}
    use_gpu = bool(comp.get("use_gpu", False))
    compute = SimpleNamespace(use_gpu=use_gpu)

    return SimpleNamespace(domain=domain, pde=pde, initial_condition=initial_condition, time=time_ns, strategy=strategy, integrator=integrator, linear=linear, nonlinear=nonlinear, reference=reference, output=output, compute=compute)

#loads the nls config from a yaml file
def load_nls_config(path):
    cfg = load_yaml(path)
    return parse_nls_config(cfg)

#evaluates a formula string with only whitelisted names
def safe_formula_eval(expr, x, params_dict):
    allowed = {"sin": np.sin, "cos": np.cos, "tan": np.tan, "exp": np.exp, "log": np.log, "sqrt": np.sqrt, "tanh": np.tanh, "sinh": np.sinh, "cosh": np.cosh, "abs": np.abs, "pi": np.pi, "e": np.e}
    def sech(z):
        return 1.0 / np.cosh(z)
    allowed["sech"] = sech
    namespace = {"x": x, **{k: float(to_number(v)) for k, v in (params_dict or {}).items()}, **allowed}
    try: return np.asarray(eval(expr, {"__builtins__": {}}, namespace), dtype=float)
    except Exception as e: raise ValueError(f"Safe eval failed for expression: {expr!r}: {e}") from e

#builds the initial state from the config
def build_initial_state(config):
    x = config.domain.x
    N = config.domain.N
    ic = config.initial_condition
    params = ic.params if hasattr(ic, "params") else {}
    if isinstance(params, dict): params = {k: to_number(v) for k, v in params.items()}

    #builds the initial state for the gaussian initial condition
    if ic.type == "gaussian":
        A = float(to_number(params.get("A", 1.0)))
        sigma = float(to_number(params.get("sigma", 2.0)))
        x0 = float(to_number(params.get("x0", 0.0)))
        k_carrier = float(to_number(params.get("k_carrier", 0.0)))
        phase = float(to_number(params.get("phase", 0.0)))
        envelope = A * np.exp(-((x - x0) ** 2) / (2.0 * sigma ** 2))
        psi0 = envelope * np.exp(1j * (k_carrier * x + phase))
        u0 = np.real(psi0)
        v0 = np.imag(psi0)
        return np.concatenate([u0, v0]).astype(float)

    #builds the initial state for the single soliton initial condition
    if ic.type == "soliton":
        eta = float(to_number(params.get("eta", 1.0)))
        x0 = float(to_number(params.get("x0", 0.0)))
        v = float(to_number(params.get("v", 0.0)))
        phase = float(to_number(params.get("phase", 0.0)))
        psi0 = eta * (1.0 / np.cosh(eta * (x - x0))) * np.exp(1j * (v / 2.0 * (x - x0) + phase))
        u0 = np.real(psi0)
        v0 = np.imag(psi0)
        return np.concatenate([u0, v0]).astype(float)

    #builds the initial state for the multi soliton initial condition
    if ic.type == "multi_soliton":
        solitons = params.get("solitons", [])
        if not solitons:
            raise ValueError("multi_soliton requires params.solitons (non-empty list)")
        relative_phases = params.get("relative_phase", [])
        if isinstance(relative_phases, (int, float)):
            relative_phases = [float(relative_phases)] * len(solitons)
        psi0 = np.zeros_like(x, dtype=complex)
        for i, sol in enumerate(solitons):
            sol = {k: to_number(v) for k, v in sol.items()} if isinstance(sol, dict) else {}
            eta = float(to_number(sol.get("eta", 1.0)))
            x0 = float(to_number(sol.get("x0", 0.0)))
            v = float(to_number(sol.get("v", 0.0)))
            phase = float(to_number(sol.get("phase", 0.0)))
            if i < len(relative_phases):
                phase += float(to_number(relative_phases[i]))
            psi0 += eta * (1.0 / np.cosh(eta * (x - x0))) * np.exp(1j * (v / 2.0 * (x - x0) + phase))
        u0 = np.real(psi0)
        v0 = np.imag(psi0)
        return np.concatenate([u0, v0]).astype(float)

    #builds the initial state for the random spectral initial condition
    if ic.type == "random_spectral":
        seed = int(to_number(params.get("seed", 0)))
        spectral_cutoff = int(to_number(params.get("spectral_cutoff", N // 2)))
        spectral_slope = float(to_number(params.get("spectral_slope", 0.0)))
        amplitude_scaling = float(to_number(params.get("amplitude_scaling", 1.0)))
        mode = str(params.get("mode", "normal")).strip().lower()
        if mode not in ("uniform", "normal"): mode = "normal"
        enforce_real = bool(params.get("enforce_real", False))
        rng = np.random.default_rng(seed)
        n_pos = min(spectral_cutoff, N // 2 + 1)
        k_indices = np.arange(1, n_pos, dtype=float)

        #builds the initial state for the random spectral initial condition
        if spectral_slope != 0:
            weights = k_indices ** (spectral_slope / 2.0)
            weights[weights == 0] = 1.0
        else: weights = np.ones_like(k_indices)

        #using uniform or normal distribution for the coefficients
        if mode == "uniform": coeffs_pos = (rng.uniform(-1, 1, len(k_indices)) + 1j * rng.uniform(-1, 1, len(k_indices))) / weights
        else: coeffs_pos = (rng.standard_normal(len(k_indices)) + 1j * rng.standard_normal(len(k_indices))) / weights
        coeffs_pos *= amplitude_scaling
        psi_hat = np.zeros(N, dtype=complex)
        psi_hat[0] = amplitude_scaling * (rng.standard_normal() if mode == "normal" else rng.uniform(-1, 1))
        psi_hat[1:n_pos] = coeffs_pos

        #enforces real symmetry
        if enforce_real:
            for j in range(1, n_pos):
                if N - j > j: psi_hat[N - j] = np.conj(psi_hat[j])
        else:
            n_neg = min(n_pos - 1, N - n_pos)
            if n_neg > 0:
                k_neg = np.arange(1, n_neg + 1, dtype=float)
                w_neg = (k_neg ** (spectral_slope / 2.0)) if spectral_slope != 0 else np.ones(n_neg)
                w_neg[w_neg == 0] = 1.0
                neg_coeffs = (rng.standard_normal(n_neg) + 1j * rng.standard_normal(n_neg)) / w_neg
                psi_hat[N - n_neg:N][::-1] = neg_coeffs * amplitude_scaling
        psi0 = np.fft.ifft(psi_hat)
        if enforce_real: psi0 = np.real(psi0).astype(complex)
        u0 = np.real(psi0)
        v0 = np.imag(psi0)
        return np.concatenate([u0, v0]).astype(float)

    #builds the initial state for the custom initial condition using the given formula options
    if ic.type == "custom":
        formula_real = params.get("formula_real") or params.get("formula")
        if not formula_real: raise ValueError("custom initial_condition requires params.formula_real or params.formula")
        formula_imag = params.get("formula_imag")
        sub_params = params.get("params", params)
        if isinstance(sub_params, dict): sub_params = {k: to_number(v) for k, v in sub_params.items() if k not in ("formula_real", "formula_imag", "formula")}
        u0 = safe_formula_eval(str(formula_real), x, sub_params)
        if formula_imag is not None and str(formula_imag).strip(): v0 = safe_formula_eval(str(formula_imag), x, sub_params)
        else: v0 = np.zeros_like(x, dtype=float)
        if u0.shape != x.shape or v0.shape != x.shape: raise ValueError("custom formula must produce arrays of same length as grid")
        return np.concatenate([u0, v0]).astype(float)
    raise ValueError(f"Unknown initial_condition type: {ic.type}")

#builds the right hand side of the nls from the configuration
def build_rhs(config):
    k_sq = config.domain.k_sq
    kappa = config.pde.kappa

    #builds the full right hand side of the nls
    def full_rhs(t, Y):
        n = len(Y) // 2
        u, v = Y[:n], Y[n:]
        psi = u + 1j * v
        psi_hat = np.fft.fft(psi)
        linear_part = np.fft.ifft(k_sq * psi_hat)
        nonlinear_part = kappa * (u * u + v * v) * psi
        F = -linear_part + nonlinear_part
        return np.concatenate([-np.imag(F), np.real(F)])

    #builds the linear right hand side of the nls
    def linear_rhs(t, Y):
        n = len(Y) // 2
        u, v = Y[:n], Y[n:]
        psi = u + 1j * v
        psi_hat = np.fft.fft(psi)
        minus_psi_xx = np.fft.ifft(k_sq * psi_hat)
        return np.concatenate([-np.imag(minus_psi_xx), np.real(minus_psi_xx)])

    #builds the nonlinear right hand side of the nls
    def nonlinear_rhs(t, Y):
        n = len(Y) // 2
        u, v = Y[:n], Y[n:]
        sq = u * u + v * v
        return np.concatenate([kappa * sq * v, -kappa * sq * u])
    return full_rhs, linear_rhs, nonlinear_rhs

#runs the reference trajectory if requested, which is with h = dt/100 with exact subflows 6th order Yoshida splitting
def run_reference(Y0, dt, n_steps, k_sq, kappa, linear_rhs, nonlinear_rhs, progress_tracker=None, use_gpu=False):
    dt_ref = dt / 100.0
    n_steps_ref = n_steps * 100
    kwargs = dict(Y0=Y0, dt=dt_ref, n_steps=n_steps_ref, strategy="strang", ordering="LN", linear_substeps=2, nonlinear_substeps=2, splitting_order=6, k_sq=k_sq, kappa=kappa, linear_type="exact", nonlinear_type="exact", use_gpu=use_gpu)
    return run_nls_backend(progress_tracker=progress_tracker, **kwargs)

#runs the main nls simulation only
def run_main_only(yaml_file, progress_tracker=None, use_gpu=None):
    #loads the config and builds the initial state and the right hand side
    config = load_nls_config(yaml_file)
    if use_gpu is None: use_gpu = getattr(config.compute, "use_gpu", False)
    Y0 = build_initial_state(config)
    full_rhs, linear_rhs, nonlinear_rhs = build_rhs(config)
    dt = config.time.dt
    n_steps = config.time.n_steps
    strategy = config.strategy.name
    ordering = config.strategy.ordering
    linear_substeps = config.strategy.linear_substeps
    nonlinear_substeps = config.strategy.nonlinear_substeps
    splitting_order = config.strategy.splitting_order
    k_sq = config.domain.k_sq
    kappa = config.pde.kappa
    kwargs = dict(Y0=Y0, dt=dt, n_steps=n_steps, strategy=strategy, ordering=ordering, linear_substeps=linear_substeps, nonlinear_substeps=nonlinear_substeps, splitting_order=splitting_order, k_sq=k_sq, kappa=kappa)
    
    #selects the right hand side and the simulation parameters
    if strategy == "full":
        kwargs["full_rhs"] = full_rhs
        kwargs["full_family"] = config.integrator.family
        kwargs["full_order"] = config.integrator.order
    else:
        kwargs["linear_type"] = config.linear.type
        kwargs["nonlinear_type"] = config.nonlinear.type
        if config.linear.type == "numeric":
            kwargs["linear_rhs"] = linear_rhs
            kwargs["linear_family"] = config.linear.family
            kwargs["linear_order"] = config.linear.order
        if config.nonlinear.type == "numeric":
            kwargs["nonlinear_rhs"] = nonlinear_rhs
            kwargs["nonlinear_family"] = config.nonlinear.family
            kwargs["nonlinear_order"] = config.nonlinear.order
    
    #calls the backend simulation and saves the results
    kwargs["use_gpu"] = use_gpu
    t_grid, Y_traj = run_nls_backend(progress_tracker=progress_tracker, **kwargs)
    psi_traj = trajectory_to_psi(Y_traj)
    save_path = getattr(config.output, "save_path", None) or ""
    if save_path and isinstance(save_path, str) and save_path.strip():
        save_path = save_path.strip()
        overwrite = getattr(config.output, "overwrite", True)
        if not overwrite and os.path.exists(save_path): raise FileExistsError(f"Output file exists and overwrite is false: {save_path}")
        save_dir = os.path.dirname(save_path)
        if save_dir: os.makedirs(save_dir, exist_ok=True)
        if not save_path.endswith(".npz"): save_path = save_path + ".npz"
        np.savez_compressed(save_path, t_grid=t_grid, Y_traj=Y_traj)
    else: save_path = None
    return {"config": config, "main": {"t_grid": t_grid, "Y_traj": Y_traj, "psi_traj": psi_traj}, "metadata": {"x": config.domain.x, "dx": config.domain.dx, "k": config.domain.k, "k_sq": config.domain.k_sq, "kappa": config.pde.kappa, "strategy": config.strategy.name}, "save_path": save_path}

#runs the reference trajectory only
def run_reference_only(yaml_file, progress_tracker=None, use_gpu=None):
    config = load_nls_config(yaml_file)
    if use_gpu is None: use_gpu = getattr(config.compute, "use_gpu", False)
    Y0 = build_initial_state(config)
    full_rhs, linear_rhs, nonlinear_rhs = build_rhs(config)
    dt = config.time.dt
    n_steps = config.time.n_steps
    k_sq = config.domain.k_sq
    kappa = config.pde.kappa
    t_grid_ref, Y_traj_ref = run_reference(Y0, dt, n_steps, k_sq, kappa, linear_rhs, nonlinear_rhs, progress_tracker=progress_tracker, use_gpu=use_gpu)
    return {"t_grid": t_grid_ref, "Y_traj": Y_traj_ref, "psi_traj": trajectory_to_psi(Y_traj_ref)}

#runs the nls simulation
def run(yaml_file, progress_tracker=None, main_progress_tracker=None, ref_progress_tracker=None, stage_callback=None):
    if main_progress_tracker is None: main_progress_tracker = progress_tracker

    #loads the config from the yaml file
    config = load_nls_config(yaml_file)
    Y0 = build_initial_state(config)

    #builds the right hand side and the simulation parameters
    full_rhs, linear_rhs, nonlinear_rhs = build_rhs(config)
    dt = config.time.dt
    n_steps = config.time.n_steps
    strategy = config.strategy.name
    ordering = config.strategy.ordering
    linear_substeps = config.strategy.linear_substeps
    nonlinear_substeps = config.strategy.nonlinear_substeps
    splitting_order = config.strategy.splitting_order
    k_sq = config.domain.k_sq
    kappa = config.pde.kappa

    #builds the arguments for the backend
    kwargs = dict(Y0=Y0, dt=dt, n_steps=n_steps, strategy=strategy, ordering=ordering, linear_substeps=linear_substeps, nonlinear_substeps=nonlinear_substeps, splitting_order=splitting_order, k_sq=k_sq, kappa=kappa)

    #adds the right hand side
    if strategy == "full":
        kwargs["full_rhs"] = full_rhs
        kwargs["full_family"] = config.integrator.family
        kwargs["full_order"] = config.integrator.order
    else:
        kwargs["linear_type"] = config.linear.type
        kwargs["nonlinear_type"] = config.nonlinear.type
        if config.linear.type == "numeric":
            kwargs["linear_rhs"] = linear_rhs
            kwargs["linear_family"] = config.linear.family
            kwargs["linear_order"] = config.linear.order
        if config.nonlinear.type == "numeric":
            kwargs["nonlinear_rhs"] = nonlinear_rhs
            kwargs["nonlinear_family"] = config.nonlinear.family
            kwargs["nonlinear_order"] = config.nonlinear.order

    #runs the main nls simulation
    if callable(stage_callback): stage_callback("main")
    t_grid, Y_traj = run_nls_backend(progress_tracker=main_progress_tracker, **kwargs)
    psi_traj = trajectory_to_psi(Y_traj)

    #runs the reference trajectory if requested
    ref_result = None
    ref_save_path = None
    if getattr(config.reference, "mode", "off") == "on":
        if callable(stage_callback): stage_callback("reference")
        t_grid_ref, Y_traj_ref = run_reference(Y0, dt, n_steps, k_sq, kappa, linear_rhs, nonlinear_rhs, progress_tracker=ref_progress_tracker)
        ref_result = (t_grid_ref, Y_traj_ref)

    #saves the results
    save_path = getattr(config.output, "save_path", None) or ""
    if save_path and isinstance(save_path, str) and save_path.strip():
        save_path = save_path.strip()
        overwrite = getattr(config.output, "overwrite", True)
        if not overwrite and os.path.exists(save_path): raise FileExistsError(f"Output file exists and overwrite is false: {save_path}")
        save_dir = os.path.dirname(save_path)
        if save_dir: os.makedirs(save_dir, exist_ok=True)
        if not save_path.endswith(".npz"): save_path = save_path + ".npz"
        np.savez_compressed(save_path, t_grid=t_grid, Y_traj=Y_traj)
        if ref_result is not None:
            ref_save_path = save_path.replace(".npz", "_ref.npz") if save_path.endswith(".npz") else save_path + "_ref.npz"
            np.savez_compressed(ref_save_path, t_grid=ref_result[0], Y_traj=ref_result[1])

    #builds a structured result for frontend visualization and reporting
    result = {"config": config, "main": {"t_grid": t_grid, "Y_traj": Y_traj, "psi_traj": psi_traj}, "reference": None, "save_path": save_path if save_path and isinstance(save_path, str) and save_path.strip() else None, "reference_save_path": ref_save_path, "metadata": {"x": config.domain.x, "dx": config.domain.dx, "k": config.domain.k, "k_sq": config.domain.k_sq, "kappa": config.pde.kappa, "strategy": config.strategy.name}}
    if ref_result is not None: result["reference"] = {"t_grid": ref_result[0], "Y_traj": ref_result[1], "psi_traj": trajectory_to_psi(ref_result[1])}
    if callable(stage_callback): stage_callback("completed")
    return result