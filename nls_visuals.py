import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


#returns style values from the current theme
def _theme_colors(theme):
  mode = str(theme or "light").lower()
  if mode == "dark":
    return {
      "face": "#0f172a",
      "axes": "#111827",
      "text": "#e5e7eb",
      "grid": "#334155",
      "line1": "#60a5fa",
      "line2": "#fca5a5",
    }
  return {
    "face": "#ffffff",
    "axes": "#ffffff",
    "text": "#111827",
    "grid": "#d1d5db",
    "line1": "#1f5fbf",
    "line2": "#b42318",
  }

#creates an output directory if it does not exist
def _ensure_dir(path):
  os.makedirs(path, exist_ok=True)
  return path

#computes the  amplitude of the wave function across time
def _mass_series(psi_traj, dx): return dx * np.sum(np.abs(psi_traj) ** 2, axis=1)

#computes the energy of the wave function across time
def _energy_series(psi_traj, k, kappa, dx):
  psi_hat = np.fft.fft(psi_traj, axis=1)
  psi_x = np.fft.ifft(1j * k[None, :] * psi_hat, axis=1)
  kinetic = dx * np.sum(np.abs(psi_x) ** 2, axis=1)
  potential = 0.5 * kappa * dx * np.sum(np.abs(psi_traj) ** 4, axis=1)
  return kinetic - potential

#aligns reference trajectory to the main time grid by interpolation
def _align_reference(t_main, t_ref, psi_ref):
  real_interp = np.array([np.interp(t_main, t_ref, np.real(psi_ref[:, j])) for j in range(psi_ref.shape[1])]).T
  imag_interp = np.array([np.interp(t_main, t_ref, np.imag(psi_ref[:, j])) for j in range(psi_ref.shape[1])]).T
  return real_interp + 1j * imag_interp

#saves the amplitude heatma
def plot_amplitude_heatmap(t_grid, x, psi_traj, output_dir, theme="light"):
  colors = _theme_colors(theme)
  path = os.path.join(output_dir, "nls_amplitude_heatmap.png")
  fig, ax = plt.subplots(figsize=(11, 5))
  fig.patch.set_facecolor(colors["face"])
  ax.set_facecolor(colors["axes"])
  im = ax.pcolormesh(x, t_grid, np.abs(psi_traj), shading="auto", cmap="viridis")
  ax.set_title("|psi(x,t)| amplitude heatmap", color=colors["text"])
  ax.set_xlabel("x", color=colors["text"])
  ax.set_ylabel("t", color=colors["text"])
  ax.tick_params(colors=colors["text"])
  cbar = fig.colorbar(im, ax=ax, label="|psi|")
  cbar.ax.yaxis.label.set_color(colors["text"])
  cbar.ax.tick_params(colors=colors["text"])
  fig.tight_layout()
  fig.savefig(path, dpi=160)
  plt.close(fig)
  return path


#saves the real and imaginary parts of the wave function at selected times
def plot_re_im_snapshots(t_grid, x, psi_traj, output_dir, theme="light"):
  colors = _theme_colors(theme)
  path = os.path.join(output_dir, "nls_re_im_snapshots.png")
  picks = np.linspace(0, len(t_grid) - 1, min(5, len(t_grid)), dtype=int)
  fig, axes = plt.subplots(1, 2, figsize=(12, 4))
  fig.patch.set_facecolor(colors["face"])
  for ax in axes:
    ax.set_facecolor(colors["axes"])
    ax.grid(True, alpha=0.3, color=colors["grid"])
    ax.tick_params(colors=colors["text"])
  for idx in picks:
    axes[0].plot(x, np.real(psi_traj[idx]), lw=1.2, label=f"t={t_grid[idx]:.3f}")
    axes[1].plot(x, np.imag(psi_traj[idx]), lw=1.2, label=f"t={t_grid[idx]:.3f}")
  axes[0].set_title("Re(psi) snapshots", color=colors["text"])
  axes[1].set_title("Im(psi) snapshots", color=colors["text"])
  axes[0].set_xlabel("x", color=colors["text"])
  axes[1].set_xlabel("x", color=colors["text"])
  axes[0].set_ylabel("amplitude", color=colors["text"])
  axes[1].set_ylabel("amplitude", color=colors["text"])
  axes[0].legend(fontsize=8)
  axes[1].legend(fontsize=8)
  fig.tight_layout()
  fig.savefig(path, dpi=160)
  plt.close(fig)
  return path


#saves the spectrum evolution as power vs mode index and time
def plot_spectrum_evolution(t_grid, k, psi_traj, output_dir, theme="light"):
  colors = _theme_colors(theme)
  path = os.path.join(output_dir, "nls_spectrum_evolution.png")
  power = np.abs(np.fft.fft(psi_traj, axis=1)) ** 2
  k_sorted_idx = np.argsort(k)
  fig, ax = plt.subplots(figsize=(11, 5))
  fig.patch.set_facecolor(colors["face"])
  ax.set_facecolor(colors["axes"])
  im = ax.pcolormesh(k[k_sorted_idx], t_grid, power[:, k_sorted_idx], shading="auto", cmap="magma")
  ax.set_title("spectral power evolution", color=colors["text"])
  ax.set_xlabel("k", color=colors["text"])
  ax.set_ylabel("t", color=colors["text"])
  ax.tick_params(colors=colors["text"])
  cbar = fig.colorbar(im, ax=ax, label="|psi_hat|^2")
  cbar.ax.yaxis.label.set_color(colors["text"])
  cbar.ax.tick_params(colors=colors["text"])
  fig.tight_layout()
  fig.savefig(path, dpi=160)
  plt.close(fig)
  return path


#saves the mass drift curve
def plot_mass_drift(t_grid, psi_traj, dx, output_dir, theme="light"):
  colors = _theme_colors(theme)
  path = os.path.join(output_dir, "nls_mass_drift.png")
  mass = _mass_series(psi_traj, dx)
  drift = np.abs(mass - mass[0])
  fig, ax = plt.subplots(figsize=(10, 4))
  fig.patch.set_facecolor(colors["face"])
  ax.set_facecolor(colors["axes"])
  ax.semilogy(t_grid, np.maximum(drift, 1e-18), color=colors["line1"], lw=1.8)
  ax.set_title("mass drift |M(t)-M(0)|", color=colors["text"])
  ax.set_xlabel("t", color=colors["text"])
  ax.set_ylabel("drift", color=colors["text"])
  ax.grid(True, alpha=0.3, color=colors["grid"])
  ax.tick_params(colors=colors["text"])
  fig.tight_layout()
  fig.savefig(path, dpi=160)
  plt.close(fig)
  return path


#saves the energy drift curve
def plot_energy_drift(t_grid, psi_traj, k, kappa, dx, output_dir, theme="light"):
  colors = _theme_colors(theme)
  path = os.path.join(output_dir, "nls_energy_drift.png")
  energy = _energy_series(psi_traj, k, kappa, dx)
  drift = energy - energy[0]
  fig, ax = plt.subplots(figsize=(10, 4))
  fig.patch.set_facecolor(colors["face"])
  ax.set_facecolor(colors["axes"])
  ax.plot(t_grid, drift, color=colors["line2"], lw=1.8)
  ax.set_title("energy drift H(t)-H(0)", color=colors["text"])
  ax.set_xlabel("t", color=colors["text"])
  ax.set_ylabel("drift", color=colors["text"])
  ax.grid(True, alpha=0.3, color=colors["grid"])
  ax.tick_params(colors=colors["text"])
  fig.tight_layout()
  fig.savefig(path, dpi=160)
  plt.close(fig)
  return path


#saves the envelope curves for the peak and rms amplitude
def plot_peak_and_rms_envelope(t_grid, psi_traj, output_dir, theme="light"):
  colors = _theme_colors(theme)
  path = os.path.join(output_dir, "nls_peak_rms_envelope.png")
  amp = np.abs(psi_traj)
  peak = np.max(amp, axis=1)
  rms = np.sqrt(np.mean(amp ** 2, axis=1))
  fig, ax = plt.subplots(figsize=(10, 4))
  fig.patch.set_facecolor(colors["face"])
  ax.set_facecolor(colors["axes"])
  ax.plot(t_grid, peak, label="peak |psi|", color=colors["line1"], lw=1.8)
  ax.plot(t_grid, rms, label="rms |psi|", color=colors["line2"], lw=1.8)
  ax.set_title("peak and rms amplitude envelope", color=colors["text"])
  ax.set_xlabel("t", color=colors["text"])
  ax.set_ylabel("amplitude", color=colors["text"])
  ax.grid(True, alpha=0.3, color=colors["grid"])
  ax.tick_params(colors=colors["text"])
  ax.legend()
  fig.tight_layout()
  fig.savefig(path, dpi=160)
  plt.close(fig)
  return path 

#saves a gallery of the amplitude of the wave function over time
def plot_profiles_gallery(t_grid, x, psi_traj, output_dir, theme="light"):
  colors = _theme_colors(theme)
  gallery_dir = _ensure_dir(os.path.join(output_dir, "profiles_gallery"))
  sample_count = min(24, len(t_grid))
  picks = np.linspace(0, len(t_grid) - 1, sample_count, dtype=int)
  paths = []
  for idx in picks:
    path = os.path.join(gallery_dir, f"profile_t_{idx:04d}.png")
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor(colors["face"])
    ax.set_facecolor(colors["axes"])
    ax.plot(x, np.abs(psi_traj[idx]), color=colors["line1"], lw=2.0)
    ax.set_title(f"|psi(x)| at t={t_grid[idx]:.4f}", color=colors["text"])
    ax.set_xlabel("x", color=colors["text"])
    ax.set_ylabel("|psi|", color=colors["text"])
    ax.grid(True, alpha=0.3, color=colors["grid"])
    ax.tick_params(colors=colors["text"])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
  return paths

#saves the L2 error between the main and reference trajectories
def plot_l2_error_vs_time(t_grid, psi_main, psi_ref_aligned, output_dir, theme="light"):
  colors = _theme_colors(theme)
  path = os.path.join(output_dir, "nls_l2_error_vs_time.png")
  l2_err = np.linalg.norm(psi_main - psi_ref_aligned, axis=1)
  fig, ax = plt.subplots(figsize=(10, 4))
  fig.patch.set_facecolor(colors["face"])
  ax.set_facecolor(colors["axes"])
  ax.semilogy(t_grid, np.maximum(l2_err, 1e-18), color=colors["line2"], lw=1.8)
  ax.set_title("main vs reference L2 error", color=colors["text"])
  ax.set_xlabel("t", color=colors["text"])
  ax.set_ylabel("L2 error", color=colors["text"])
  ax.grid(True, alpha=0.3, color=colors["grid"])
  ax.tick_params(colors=colors["text"])
  fig.tight_layout()
  fig.savefig(path, dpi=160)
  plt.close(fig)
  return path


#saves the phase-aligned error between the main and reference trajectories
def plot_phase_error_vs_time(t_grid, psi_main, psi_ref_aligned, output_dir, theme="light"):
  colors = _theme_colors(theme)
  path = os.path.join(output_dir, "nls_phase_error_vs_time.png")
  phase_error = np.zeros(len(t_grid), dtype=float)
  for i in range(len(t_grid)):
    overlap = np.vdot(psi_ref_aligned[i], psi_main[i])
    theta = np.angle(overlap) if np.abs(overlap) > 0 else 0.0
    aligned = psi_main[i] * np.exp(-1j * theta)
    phase_error[i] = np.linalg.norm(aligned - psi_ref_aligned[i])
  fig, ax = plt.subplots(figsize=(10, 4))
  fig.patch.set_facecolor(colors["face"])
  ax.set_facecolor(colors["axes"])
  ax.semilogy(t_grid, np.maximum(phase_error, 1e-18), color=colors["line1"], lw=1.8)
  ax.set_title("phase-aligned main vs reference error", color=colors["text"])
  ax.set_xlabel("t", color=colors["text"])
  ax.set_ylabel("error", color=colors["text"])
  ax.grid(True, alpha=0.3, color=colors["grid"])
  ax.tick_params(colors=colors["text"])
  fig.tight_layout()
  fig.savefig(path, dpi=160)
  plt.close(fig)
  return path


#saves a combined invariant drift comparison figure between the main and reference trajectories
def plot_invariant_bias_main_vs_ref(t_grid, psi_main, psi_ref_aligned, k, kappa, dx, output_dir, theme="light"):
  colors = _theme_colors(theme)
  path = os.path.join(output_dir, "nls_invariant_bias_main_vs_ref.png")
  mass_main = _mass_series(psi_main, dx)
  mass_ref = _mass_series(psi_ref_aligned, dx)
  energy_main = _energy_series(psi_main, k, kappa, dx)
  energy_ref = _energy_series(psi_ref_aligned, k, kappa, dx)
  fig, axes = plt.subplots(1, 2, figsize=(12, 4))
  fig.patch.set_facecolor(colors["face"])
  for ax in axes:
    ax.set_facecolor(colors["axes"])
    ax.grid(True, alpha=0.3, color=colors["grid"])
    ax.tick_params(colors=colors["text"])
  axes[0].plot(t_grid, np.abs((mass_main - mass_main[0]) - (mass_ref - mass_ref[0])), color=colors["line1"], lw=1.6)
  axes[0].set_title("mass drift bias", color=colors["text"])
  axes[0].set_xlabel("t", color=colors["text"])
  axes[0].set_ylabel("|deltaM_main-deltaM_ref|", color=colors["text"])
  axes[1].plot(t_grid, (energy_main - energy_main[0]) - (energy_ref - energy_ref[0]), color=colors["line2"], lw=1.6)
  axes[1].set_title("energy drift bias", color=colors["text"])
  axes[1].set_xlabel("t", color=colors["text"])
  axes[1].set_ylabel("deltaH_main-deltaH_ref", color=colors["text"])
  fig.tight_layout()
  fig.savefig(path, dpi=160)
  plt.close(fig)
  return path


#computes the metrics for the main-vs-reference comparison
def compute_reference_metrics(t_grid, psi_main, psi_ref_aligned, k, kappa, dx):
  l2_err = np.linalg.norm(psi_main - psi_ref_aligned, axis=1)
  denom = np.linalg.norm(psi_ref_aligned[-1]) + 1e-16
  rel_final_error = float(np.linalg.norm(psi_main[-1] - psi_ref_aligned[-1]) / denom)
  time_avg_l2_error = float(np.mean(l2_err))

  mass_main = _mass_series(psi_main, dx)
  mass_ref = _mass_series(psi_ref_aligned, dx)
  energy_main = _energy_series(psi_main, k, kappa, dx)
  energy_ref = _energy_series(psi_ref_aligned, k, kappa, dx)
  mass_drift_main = np.abs(mass_main - mass_main[0])
  mass_drift_ref = np.abs(mass_ref - mass_ref[0])
  energy_drift_main = np.abs(energy_main - energy_main[0])
  energy_drift_ref = np.abs(energy_ref - energy_ref[0])
  mass_drift_ratio = float(np.max(mass_drift_main) / (np.max(mass_drift_ref) + 1e-16))
  energy_drift_ratio = float(np.max(energy_drift_main) / (np.max(energy_drift_ref) + 1e-16))

  amp_main = np.abs(psi_main)
  amp_ref = np.abs(psi_ref_aligned)
  peak_main = np.max(amp_main, axis=1)
  peak_ref = np.max(amp_ref, axis=1)
  t_peak_main = float(t_grid[np.argmax(peak_main)])
  t_peak_ref = float(t_grid[np.argmax(peak_ref)])
  peak_time_offset = t_peak_main - t_peak_ref
  peak_value_offset = float(peak_main[np.argmax(peak_main)] - peak_ref[np.argmax(peak_ref)])

  power_main = np.abs(np.fft.fft(psi_main, axis=1)) ** 2
  power_ref = np.abs(np.fft.fft(psi_ref_aligned, axis=1)) ** 2
  cutoff = int(max(1, power_main.shape[1] // 4))
  hk_main = np.sum(power_main[:, cutoff:], axis=1) / (np.sum(power_main, axis=1) + 1e-16)
  hk_ref = np.sum(power_ref[:, cutoff:], axis=1) / (np.sum(power_ref, axis=1) + 1e-16)
  spectral_mismatch = float(np.mean(np.abs(hk_main - hk_ref)))

  return {
    "relative_final_state_error": rel_final_error,
    "time_averaged_l2_error": time_avg_l2_error,
    "mass_drift_ratio_main_over_ref": mass_drift_ratio,
    "energy_drift_ratio_main_over_ref": energy_drift_ratio,
    "peak_time_offset": float(peak_time_offset),
    "peak_value_offset": float(peak_value_offset),
    "spectral_mismatch": spectral_mismatch,
  }


#writes the metrics to npz and text files
def save_metrics(metrics, output_dir):
  npz_path = os.path.join(output_dir, "nls_reference_metrics.npz")
  txt_path = os.path.join(output_dir, "nls_reference_metrics.txt")
  np.savez_compressed(npz_path, **metrics)
  lines = ["NLS main vs reference metrics", "================================", ""]
  for key in sorted(metrics.keys()):
    lines.append(f"{key}: {metrics[key]:.10e}")
  with open(txt_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
  return npz_path, txt_path, "\n".join(lines[3:])

#runs the main nls visuals
def generate_main_visuals(run_result, output_dir, theme="light"):
  #creates the output directories
  _ensure_dir(output_dir)
  plots_dir = _ensure_dir(os.path.join(output_dir, "plots"))
  t_grid = run_result["main"]["t_grid"]
  psi_main = run_result["main"]["psi_traj"]
  x = run_result["metadata"]["x"]
  k = run_result["metadata"]["k"]
  dx = run_result["metadata"]["dx"]
  kappa = run_result["metadata"]["kappa"]

  #creates the main nls visuals
  plot_paths = []
  plot_paths.append(plot_amplitude_heatmap(t_grid, x, psi_main, plots_dir, theme=theme))
  plot_paths.append(plot_re_im_snapshots(t_grid, x, psi_main, plots_dir, theme=theme))
  plot_paths.append(plot_spectrum_evolution(t_grid, k, psi_main, plots_dir, theme=theme))
  plot_paths.append(plot_mass_drift(t_grid, psi_main, dx, plots_dir, theme=theme))
  plot_paths.append(plot_energy_drift(t_grid, psi_main, k, kappa, dx, plots_dir, theme=theme))
  plot_paths.append(plot_peak_and_rms_envelope(t_grid, psi_main, plots_dir, theme=theme))
  gallery_paths = plot_profiles_gallery(t_grid, x, psi_main, plots_dir, theme=theme)
  plot_paths.extend(gallery_paths)
  return {"plot_paths": plot_paths, "plots_dir": plots_dir}

#runs only reference comparison visual and metric generation; run_result must include reference
def generate_comparison_visuals(run_result_with_ref, output_dir, theme="light"):
  #creates the output directories
  _ensure_dir(output_dir)
  plots_dir = _ensure_dir(os.path.join(output_dir, "plots"))
  metrics_dir = _ensure_dir(os.path.join(output_dir, "metrics"))

  #extracts the main and reference simulation data
  t_grid = run_result_with_ref["main"]["t_grid"]
  psi_main = run_result_with_ref["main"]["psi_traj"]
  t_ref = run_result_with_ref["reference"]["t_grid"]
  psi_ref = run_result_with_ref["reference"]["psi_traj"]
  psi_ref_aligned = _align_reference(t_grid, t_ref, psi_ref)
  x = run_result_with_ref["metadata"]["x"]
  k = run_result_with_ref["metadata"]["k"]
  dx = run_result_with_ref["metadata"]["dx"]
  kappa = run_result_with_ref["metadata"]["kappa"]

  #creates the comparison nls visuals
  plot_paths = []
  plot_paths.append(plot_l2_error_vs_time(t_grid, psi_main, psi_ref_aligned, plots_dir, theme=theme))
  plot_paths.append(plot_phase_error_vs_time(t_grid, psi_main, psi_ref_aligned, plots_dir, theme=theme))
  plot_paths.append(plot_invariant_bias_main_vs_ref(t_grid, psi_main, psi_ref_aligned, k, kappa, dx, plots_dir, theme=theme))
  metrics = compute_reference_metrics(t_grid, psi_main, psi_ref_aligned, k, kappa, dx)
  npz_path, txt_path, _ = save_metrics(metrics, metrics_dir)
  metric_paths = [npz_path, txt_path]
  return {"plot_paths": plot_paths, "metric_paths": metric_paths, "plots_dir": plots_dir, "metrics_dir": metrics_dir}

#runs all the visual and metric generation steps and returns the generated paths
def generate_all_visuals(run_result, output_dir, theme="light"):
  _ensure_dir(output_dir)
  plots_dir = _ensure_dir(os.path.join(output_dir, "plots"))
  metrics_dir = _ensure_dir(os.path.join(output_dir, "metrics"))
  plot_paths = []
  metric_paths = []
  metric_text = "Reference metrics unavailable (reference mode is off)."

  #extracts main simulation data
  t_grid = run_result["main"]["t_grid"]
  psi_main = run_result["main"]["psi_traj"]
  x = run_result["metadata"]["x"]
  k = run_result["metadata"]["k"]
  dx = run_result["metadata"]["dx"]
  kappa = run_result["metadata"]["kappa"]

  #creates the core nls visuals
  plot_paths.append(plot_amplitude_heatmap(t_grid, x, psi_main, plots_dir, theme=theme))
  plot_paths.append(plot_re_im_snapshots(t_grid, x, psi_main, plots_dir, theme=theme))
  plot_paths.append(plot_spectrum_evolution(t_grid, k, psi_main, plots_dir, theme=theme))
  plot_paths.append(plot_mass_drift(t_grid, psi_main, dx, plots_dir, theme=theme))
  plot_paths.append(plot_energy_drift(t_grid, psi_main, k, kappa, dx, plots_dir, theme=theme))
  plot_paths.append(plot_peak_and_rms_envelope(t_grid, psi_main, plots_dir, theme=theme))
  plot_paths.extend(plot_profiles_gallery(t_grid, x, psi_main, plots_dir, theme=theme))

  #creates reference comparison visuals and metrics when reference exists
  if run_result.get("reference") is not None:
    t_ref = run_result["reference"]["t_grid"]
    psi_ref = run_result["reference"]["psi_traj"]
    psi_ref_aligned = _align_reference(t_grid, t_ref, psi_ref)
    plot_paths.append(plot_l2_error_vs_time(t_grid, psi_main, psi_ref_aligned, plots_dir, theme=theme))
    plot_paths.append(plot_phase_error_vs_time(t_grid, psi_main, psi_ref_aligned, plots_dir, theme=theme))
    plot_paths.append(plot_invariant_bias_main_vs_ref(t_grid, psi_main, psi_ref_aligned, k, kappa, dx, plots_dir, theme=theme))
    metrics = compute_reference_metrics(t_grid, psi_main, psi_ref_aligned, k, kappa, dx)
    npz_path, txt_path, metric_text = save_metrics(metrics, metrics_dir)
    metric_paths.extend([npz_path, txt_path])

  return {"plot_paths": plot_paths, "metric_paths": metric_paths, "metric_text": metric_text, "plots_dir": plots_dir, "metrics_dir": metrics_dir}
