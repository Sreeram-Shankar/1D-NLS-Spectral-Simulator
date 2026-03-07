# 1D Nonlinear Schrödinger (NLS) Spectral Simulator

A configurable, YAML-driven numerical testbed for the one-dimensional focusing/defocusing Nonlinear Schrödinger equation with spectral (FFT) spatial discretization, multiple time-stepping strategies, and a desktop GUI for editing configuration, running simulations, and viewing results.

---

## Table of Contents

- [Overview](#overview)
- [The 1D NLS Equation](#the-1d-nls-equation)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration (YAML)](#configuration-yaml)
- [Initial Conditions](#initial-conditions)
- [Time-Stepping Strategies](#time-stepping-strategies)
- [Integrator Families](#integrator-families)
- [Presets](#presets)
- [Outputs and Visualizations](#outputs-and-visualizations)
- [Reference Solution and Metrics](#reference-solution-and-metrics)

---

## Overview

This application solves the **1D Nonlinear Schrödinger equation** on a periodic spatial domain using a **spectral (pseudo-spectral) method**: the Laplacian is applied in Fourier space, and the nonlinearity in physical space. The state is advanced in time via either **full integration** (one ODE integrator on the full RHS) or **operator splitting** (linear and nonlinear parts advanced separately with exact or numeric sub-steps, composed with Lie or Strang splitting and optional Yoshida higher-order composition). A **reference trajectory** (finer time step with exact sub-flows) can be computed for comparison and error metrics. The GUI provides a YAML editor, run progress, and access to all generated plots and exports. The goal of this program is to be a numerical testbed to compare different integrator families, strategies, and splitting and their affects on the Nonlinear Schrödinger model. 

---

## The 1D NLS Equation

The equation solved is

$$
i \psi_t = -\psi_{xx} + \kappa \, |\psi|^2 \psi
$$

on a periodic domain $x \in [-L, L)$ with $N$ grid points. $\psi(x,t)$ is the complex wave function; $\kappa > 0$ is **focusing**, $\kappa < 0$ is **defocusing**. The state is stored as a **real embedding** $Y = [u; v]$ with $\psi = u + iv$ (length $2N$). Spatial derivatives are computed via FFT: $\psi_{xx}$ is $\mathcal{F}^{-1}(-k^2 \hat{\psi})$. The **linear part** is $-\psi_{xx}$; the **nonlinear part** is $\kappa |\psi|^2 \psi$.

**Mass:** $M(t) = \int_{-L}^{L} |\psi|^2 \, dx$  
**Hamiltonian:** $H(t) = \int_{-L}^{L} |\psi_x|^2 \, dx - \frac{\kappa}{2} \int_{-L}^{L} |\psi|^4 \, dx$

---

## Features

- **Spectral discretization**: Fast Fourier Transform-based Laplacian; configurable $L$ and $N$.
- **Initial conditions**: Gaussian, single soliton, multi-soliton, random spectral, custom formula.
- **Time-stepping strategy**: **Full** (one integrator), **Lie** or **Strang** splitting with **exact** or **numeric** sub-flows; Strang with Yoshida orders 2, 4, 6.
- **Integrators (orders: n indicates arbitrary generation)**: Explicit RK (1–7), SDIRK (2–4), Gauss–Legendre (2n), Radau IIA(2n-1), Lobatto IIIC (2n-2), Rosenbrock (2), Adams–Bashforth (n), Adams–Moulton (n), ABM Predictor Corrector (n), BDF (n).
- **Reference run**: Strang, 6th-order Yoshida, exact linear+nonlinear, $\Delta t/100$, for comparison and metrics.
- **GUI**: CustomTkinter YAML editor, Run with progress, main and comparison plots, export graphs/data, theme toggle.
- **Optional GPU**: CuPy for splitting with exact linear and exact nonlinear (YAML: `compute.use_gpu`).

---

## Project Structure

```
├── nls_main.py
├── nls_backend.py
├── nls_frontend.py
├── nls_visuals.py
├── nls_config.yaml
├── nls_yaml_info.txt
├── theme.json
├── presets/
│   ├── gaussian_pulse.yaml
│   ├── single_solition.yaml
│   ├── double_solition.yaml
│   ├── random_spectral.yaml
│   └── custom_formula.yaml
├── solvers/
│   ├── rk.py
│   ├── sdirk.py
│   ├── irk.py
│   ├── linear_multistep.py
│   └── rosenbrock2.py
├── generation/
│   ├── multistep.py
│   ├── bdf.py
│   ├── gauss_legendre.py
│   ├── radau.py
│   └── lobatto.py
```

---

## Installation

**Python 3.8+** recommended. From the project root:

```bash
pip install numpy pyyaml matplotlib customtkinter Pillow tqdm mpmath
```

Optional: `scipy` (faster linear algebra). Optional GPU: `cupy-cuda13x` (or match your CUDA).

---

## Usage

**GUI:**

```bash
python nls_frontend.py
```

Edit YAML, Save, Run. Use result buttons to open plots; Export All Graphs / Export All Data; Info, Reset, Theme, Restart, Exit.

**CLI:**

```bash
python nls_main.py presets/gaussian_pulse.yaml
```

---

## Configuration (YAML)

Required: `domain` (L, N), `pde` (kappa), `initial_condition` (type, params), `time` (dt, n_steps or t_final), `strategy` (name, ordering, splitting_order, linear_substeps, nonlinear_substeps).  
If `strategy: full`: `integrator` (family, order).  
If `strategy: lie` or `strang`: `linear` and `nonlinear` (type: exact|numeric; if numeric: family, order).  
Optional: `reference` (mode: on|off), `output` (save_path, overwrite), `compute` (use_gpu).

---

## Initial Conditions

| Type | Main params |
|------|-------------|
| **gaussian** | A, sigma, x0, k_carrier, phase |
| **soliton** | eta, x0, v, phase |
| **multi_soliton** | solitons (list of {eta, x0, v, phase}), relative_phase |
| **random_spectral** | seed, spectral_cutoff, spectral_slope, amplitude_scaling, mode, enforce_real |
| **custom** | formula_real, formula_imag (optional), params (constants); safe eval only |

---

## Time-Stepping Strategies

- **full**: Integrate full RHS with chosen integrator (family + order).
- **lie**: Linear step $\Delta t$, then nonlinear $\Delta t$ (or reverse if ordering NL); each sub-flow exact or numeric.
- **strang**: Linear $\Delta t/2$ → nonlinear $\Delta t$ → linear $\Delta t/2$; each exact or numeric. **Yoshida** orders 2, 4, 6 compose multiple Strang steps with weights for higher order.

---

## Integrator Families

`explicit runge-kutta` (1–7), `sdirk` (2–4), `gauss-legendre`, `radauiia`, `lobattoiiic`, `rosenbrock` (2), `adams-bashforth`, `adams-moulton`, `adams-bashforth-moulton` or `abm`, `bdf`. Implicit/multistep methods use analytical NLS Jacobians when available.

---

## Presets

| File | Description |
|------|-------------|
| **gaussian_pulse.yaml** | Gaussian IC; Strang 4th-order Yoshida; linear & nonlinear Adams–Moulton 2. |
| **single_solition.yaml** | Single soliton; Strang 2; linear Gauss–Legendre 2, nonlinear Rosenbrock 2. |
| **double_solition.yaml** | Two solitons; Strang 6th-order Yoshida; linear & nonlinear RK4. |
| **random_spectral.yaml** | Random spectral IC; full BDF3. |
| **custom_formula.yaml** | Custom formula IC; Strang 4th-order Yoshida; exact linear & nonlinear. |

---

## Outputs and Visualizations

**Directory:** `nls_outputs/<YYYYMMDD_HHMMSS>/`

**plots/:** amplitude heatmap, Re/Im snapshots, spectrum evolution, mass drift, energy drift, peak/RMS envelope, **profiles_gallery/** (|\psi(x)| at selected times).  
**metrics/** (if reference on): nls_reference_metrics.npz and .txt (relative final error, time-averaged L2, drift ratios, peak/time offsets, spectral mismatch).  
Trajectory: optional `output.save_path` (.npz); reference saved as _ref.npz when applicable.  
GUI: buttons open PNGs; Export All Graphs (ZIP); Export All Data (ZIP of trajectories + parameters).

---

## Reference Solution and Metrics

When `reference.mode: on`, a reference trajectory is computed with Strang, 6th-order Yoshida, exact linear and nonlinear, $\Delta t_{\mathrm{ref}} = \Delta t/100$, same $t_{\mathrm{final}}$. It is interpolated onto the main time grid. Comparison plots: L2 error vs time, phase-aligned error, invariant bias (mass/energy drift main vs ref). Metrics are written to **metrics/**.

---

## License

See repository license file, if present.
