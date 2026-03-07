import customtkinter as ctk
from tkinter import scrolledtext, filedialog
import os
import subprocess
import sys
import threading
import zipfile
import numpy as np
from datetime import datetime
from PIL import Image
from customtkinter import CTkImage
import yaml
from nls_main import run, parse_nls_config, run_main_only, run_reference_only
from nls_visuals import generate_all_visuals, generate_main_visuals, generate_comparison_visuals

#sets the appearance and theme of the overall window
ctk.set_default_color_theme("theme.json")
ctk.set_appearance_mode("system")

#class that tracks simulation progress
class ProgressTracker:
  def __init__(self):
    self.current_step = 0
    self.total_steps = 1

#class that contains the window and all widgets
class NLSApp(ctk.CTk):
  #creates and configures the root
  def __init__(self):
    super().__init__()
    self.title("1D NLS Spectral Simulator")
    self.geometry("1000x800")
    self.resizable(False, False)
    self.yaml_file = "nls_config.yaml"
    self.simulation_running = False
    self.simulation_started = False
    self.main_progress_tracker = None
    self.ref_progress_tracker = None
    self.reference_enabled = False
    self.run_stage = "idle"
    self.latest_run_result = None
    self.latest_visual_output_dir = None
    self.latest_visual_summary = None
    self.build_gui()

  #function that builds all the gui components of the window
  def build_gui(self):
    #configures the grid layout of the window
    self.grid_rowconfigure(0, weight=0)
    self.grid_rowconfigure(1, weight=1)
    self.grid_rowconfigure(2, weight=0)
    self.grid_rowconfigure(3, weight=0)
    for j in range(10): self.grid_columnconfigure(j, weight=1)

    #creates and places the main label
    self.main_label = ctk.CTkLabel(self, text="1D NLS Spectral Simulator - YAML Configuration Editor", font=("Times New Roman", 28))
    self.main_label.grid(row=0, column=0, columnspan=10, pady=5, sticky="ew")

    #creates the text editor frame
    self.editor_frame = ctk.CTkFrame(self)
    self.editor_frame.grid(row=1, column=0, columnspan=10, padx=10, pady=5, sticky="nsew")
    self.editor_frame.grid_rowconfigure(0, weight=1)
    self.editor_frame.grid_columnconfigure(0, weight=1)

    #creates the text editor with theme-appropriate colors
    self.text_editor = scrolledtext.ScrolledText(self.editor_frame, wrap="none", font=("Consolas", 16))
    self.text_editor.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    self.update_text_editor_colors()

    #loads an initial yaml file into the editor
    self.load_yaml()

    #creates the button frame at the bottom
    self.button_frame = ctk.CTkFrame(self)
    self.button_frame.grid(row=2, column=0, columnspan=10, padx=10, pady=10, sticky="ew")
    self.button_frame.grid_rowconfigure(0, weight=1)
    for j in range(7): self.button_frame.grid_columnconfigure(j, weight=1)

    #defines the buttons
    buttons = [
      ("Info", self.show_yaml_info),
      ("Open", self.open_yaml),
      ("Save", self.save_yaml),
      ("Reset", self.reset_yaml),
      ("Theme", self.toggle_theme),
      ("Run", self.run_simulation),
      ("Exit", self.exit_app),
    ]

    #places the buttons
    for i, (text, command) in enumerate(buttons):
      btn = ctk.CTkButton(self.button_frame, text=text, command=command, font=("Times New Roman", 18), width=140)
      btn.grid(row=0, column=i, padx=5, pady=5)

  #function that updates the text editor colors based on the theme
  def update_text_editor_colors(self):
    mode = ctk.get_appearance_mode().lower()
    if mode == "light":
      bg_color = "#ffffff"
      fg_color = "#1f2933"
      insert_color = "#1f5fbf"
    else:
      bg_color = "#111827"
      fg_color = "#e5e7eb"
      insert_color = "#3b82f6"
    self.text_editor.configure(bg=bg_color, fg=fg_color, insertbackground=insert_color)

  #function that toggles the theme of the window
  def toggle_theme(self):
    current = ctk.get_appearance_mode().lower()
    ctk.set_appearance_mode("light" if current == "dark" else "dark")
    self.update_text_editor_colors()

  #function that loads the yaml file into the editor
  def load_yaml(self):
    #ensures the yaml file exists
    if not os.path.exists(self.yaml_file):
      example_path = os.path.join("presets", "nls_config_example.yaml")
      if os.path.exists(example_path):
        try:
          with open(example_path, "r", encoding="utf-8") as f:
            content = f.read()
          with open(self.yaml_file, "w", encoding="utf-8") as f:
            f.write(content)
        except Exception: pass

    #loads the yaml file into the editor
    try:
      if os.path.exists(self.yaml_file):
        with open(self.yaml_file, "r", encoding="utf-8") as f:
          content = f.read()
      else:
        content = "# YAML file not found. Create your configuration here.\n"
      self.text_editor.delete("1.0", "end")
      self.text_editor.insert("1.0", content)
    except Exception as e:
      self.text_editor.delete("1.0", "end")
      self.text_editor.insert("1.0", f"# Error loading YAML: {e}\n")

  #function that opens a yaml file from the file system
  def open_yaml(self):
    #opens a file dialog and loads the selected yaml file
    filename = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml;*.yml"), ("All files", "*.*")])
    if not filename: return
    self.yaml_file = filename
    self.load_yaml()
    self.main_label.configure(text=f"Loaded {os.path.basename(filename)}")
    if not self.simulation_started: self.after(2000, lambda: self.main_label.configure(text="1D NLS Spectral Simulator - YAML Configuration Editor"))

  #function that saves the yaml file from the editor
  def save_yaml(self):
    #saves the current editor contents to the yaml file 
    content = self.text_editor.get("1.0", "end-1c")
    try:
      with open(self.yaml_file, "w", encoding="utf-8") as f:
        f.write(content)
      self.main_label.configure(text="YAML saved successfully")
      if not self.simulation_started:
        self.after(2000, lambda: self.main_label.configure(text="1D NLS Spectral Simulator - YAML Configuration Editor"))
    except Exception as e:
      self.main_label.configure(text=f"Error saving YAML: {e}")

  #function that opens a popup window showing the congiguration information
  def show_yaml_info(self):
    info_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nls_yaml_info.txt")
    try:
      if os.path.exists(info_path):
        with open(info_path, "r", encoding="utf-8") as f: content = f.read()
      else: content = f"# File not found: {info_path}\n"
    except Exception as e: content = f"# Error loading nls_yaml_info.txt: {e}\n"
    self.show_info_popup("NLS YAML Configuration", content)

  #function that shows a popup window with the given title and text content
  def show_info_popup(self, title, content):
    #creates a toplevel window for the info
    popup = ctk.CTkToplevel(self)
    popup.title(title)
    popup.geometry("900x700")
    popup.resizable(False, False)

    #configures the grid
    popup.grid_rowconfigure(0, weight=1)
    popup.grid_rowconfigure(1, weight=0)
    popup.grid_columnconfigure(0, weight=1)

    #creates a frame for the scrollable text
    text_frame = ctk.CTkFrame(popup)
    text_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
    text_frame.grid_rowconfigure(0, weight=1)
    text_frame.grid_columnconfigure(0, weight=1)

    #gets current theme to set appropriate colors for the text widget
    mode = ctk.get_appearance_mode().lower()
    if mode == "light":
      bg_color = "#ffffff"
      fg_color = "#1f2933"
      insert_color = "#1f5fbf"
    else:
      bg_color = "#111827"
      fg_color = "#e5e7eb"
      insert_color = "#3b82f6"

    #creates the scrollable text widget and inserts the content
    text_widget = scrolledtext.ScrolledText(text_frame, wrap="word", font=("Consolas", 16), bg=bg_color, fg=fg_color, insertbackground=insert_color, width=80, height=30)
    text_widget.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
    text_widget.insert("1.0", content)
    text_widget.configure(state="disabled")

    #adds a close button
    close_btn = ctk.CTkButton(popup, text="Close", command=popup.destroy, font=("Times New Roman", 18))
    close_btn.grid(row=1, column=0, padx=20, pady=10)

  #function that resets the yaml editor to the current file contents
  def reset_yaml(self):
    self.load_yaml()
    self.main_label.configure(text="YAML reset to file contents")
    if not self.simulation_started:
      self.after(2000, lambda: self.main_label.configure(text="1D NLS Spectral Simulator - YAML Configuration Editor"))

  #function that validates the yaml in the editor before running
  def validate_yaml(self):
    content = self.text_editor.get("1.0", "end-1c")
    try:
      raw = yaml.safe_load(content)
    except Exception as e:
      self.show_validation_error(f"YAML parse error: {e}")
      return None
    try:
      return parse_nls_config(raw)
    except Exception as e:
      self.show_validation_error(f"Configuration error: {e}")
      return None

  #function that shows a validation or config error in the main label
  def show_validation_error(self, error_msg):
    self.main_label.configure(text=error_msg, font=("Times New Roman", 22))
    self.after(4500, lambda: self.main_label.configure(text="1D NLS Spectral Simulator - YAML Configuration Editor", font=("Times New Roman", 28)))

  #function that updates the stage from the backend thread
  def set_stage(self, stage):
    self.run_stage = str(stage or "").strip().lower()

  #function that begins the simulation (planetary-style: hide config, show only main label and progress bar)
  def run_simulation(self):
    self.simulation_started = True
    self.save_yaml()
    config = self.validate_yaml()
    if config is None: return
    self.reference_enabled = getattr(config.reference, "mode", "off") == "on"

    #destroys all config widgets except the main label 
    for widget in list(self.winfo_children()):
      if widget != self.main_label:
        widget.destroy()

    #configures grid so label and progress bar sit in the center of the screen
    for i in range(20): self.grid_rowconfigure(i, weight=1)
    for j in range(10): self.grid_columnconfigure(j, weight=1)

    #configures the main label for simulation (centered in the middle of the screen)
    self.main_label.configure(text="Running main NLS simulation...", font=("Times New Roman", 54))
    self.main_label.grid(row=7, column=0, columnspan=10, pady=(0, 15), sticky="nsew")

    #creates main progress tracker and single progress bar for main run (below the label, centered)
    self.main_progress_tracker = ProgressTracker()
    self.main_progress_label = ctk.CTkLabel(self, text="Main Progress: 0/0", font=("Times New Roman", 28))
    self.main_progress_label.grid(row=8, column=0, columnspan=10, pady=(0, 5), sticky="nsew")
    self.main_progress_bar = ctk.CTkProgressBar(self, orientation="horizontal", width=500, height=24, corner_radius=10)
    self.main_progress_bar.set(0)
    self.main_progress_bar.grid(row=9, column=0, columnspan=10, pady=(0, 15), sticky="n")

    #sets the stage and starts the simulation
    self.run_stage = "main"
    self.simulation_running = True
    self.update_progress()
    thread = threading.Thread(target=self.run_main_thread, daemon=True)
    thread.start()

  #function that runs the main simulation in a separate thread
  def run_main_thread(self):
    try:
      self.latest_run_result = run_main_only(self.yaml_file, self.main_progress_tracker)
      self.after(0, self.after_main_simulation)
    except Exception as e:
      self.after(0, lambda err=str(e): self.show_error(err))

  #function that updates the progress bar(s) from tracker values
  def update_progress(self):
    try:
      #updates the main progress bar and label
      if self.main_progress_tracker is not None and hasattr(self, "main_progress_bar") and self.main_progress_bar.winfo_exists():
        current = int(self.main_progress_tracker.current_step)
        total = int(self.main_progress_tracker.total_steps)
        if total > 0:
          self.main_progress_bar.set(max(0.0, min(1.0, current / total)))
          self.main_progress_label.configure(text=f"Main Progress: {current}/{total}")

      #updates the reference progress bar and label
      if self.ref_progress_tracker is not None and hasattr(self, "ref_progress_bar") and self.ref_progress_bar.winfo_exists():
        current = int(self.ref_progress_tracker.current_step)
        total = int(self.ref_progress_tracker.total_steps)
        if total > 0:
          self.ref_progress_bar.set(max(0.0, min(1.0, current / total)))
          self.ref_progress_label.configure(text=f"Reference Progress: {current}/{total}")
    except Exception: pass
    if self.simulation_running:
      self.after(200, self.update_progress)

  #function that runs after main simulation completes to create main visuals
  def after_main_simulation(self):
    self.simulation_running = False
    if hasattr(self, "main_progress_bar") and self.main_progress_bar.winfo_exists():
      self.main_progress_bar.destroy()
    if hasattr(self, "main_progress_label") and self.main_progress_label.winfo_exists():
      self.main_progress_label.destroy()

    #resets the main label to the top of the screen
    self.main_label.configure(text="Creating main visualizations", font=("Times New Roman", 54))
    self.main_label.grid(row=0, column=0, columnspan=10, pady=10, sticky="ew")
    self.update()
    thread = threading.Thread(target=self.create_main_visuals_thread, daemon=True)
    thread.start()

  #function that creates main visuals in a separate thread
  def create_main_visuals_thread(self):
    try:
      #creates the output directory
      output_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nls_outputs")
      os.makedirs(output_root, exist_ok=True)
      output_dir = os.path.join(output_root, datetime.now().strftime("%Y%m%d_%H%M%S"))
      os.makedirs(output_dir, exist_ok=True)
      self.latest_visual_output_dir = output_dir

      #generates the main visuals
      theme = ctk.get_appearance_mode().lower()
      summary = generate_main_visuals(self.latest_run_result, output_dir, theme=theme)
      self.graphs_dir = summary["plots_dir"]
      self.after(0, self.show_main_results_screen)

      #sets up the reference simulation if enabled
      if self.reference_enabled: self.after(0, self.setup_reference_simulation)
      else: self.after(0, self.finalize_simulation_no_ref)
    except Exception as e:
      self.after(0, lambda err=str(e): self.show_error(err))

  #function that shows the main results screen with buttons
  def show_main_results_screen(self):
    #configures the grid for the main results screen
    for i in range(10): self.grid_rowconfigure(i, weight=0)
    self.grid_rowconfigure(1, weight=1)
    self.grid_rowconfigure(2, weight=1)
    self.grid_rowconfigure(4, weight=2)
    for j in range(4): self.grid_columnconfigure(j, weight=1)
    self.main_label.configure(text="Simulation Results", font=("Times New Roman", 42))
    self.main_label.grid(row=0, column=0, columnspan=4, pady=(20, 18), sticky="ew")

    #creates the buttons for the main plots
    main_plots = [
      ("Amplitude heatmap", "nls_amplitude_heatmap.png", "|psi(x,t)| amplitude heatmap"),
      ("Re/Im snapshots", "nls_re_im_snapshots.png", "Re and Im snapshots"),
      ("Spectrum evolution", "nls_spectrum_evolution.png", "Spectral power evolution"),
      ("Mass drift", "nls_mass_drift.png", "Mass drift |M(t)-M(0)|"),
      ("Energy drift", "nls_energy_drift.png", "Energy drift H(t)-H(0)"),
      ("Peak & RMS envelope", "nls_peak_rms_envelope.png", "Peak and RMS amplitude envelope"),
      ("Profile gallery", None, "Profile gallery"),
    ]
    for i, (btn_text, filename, title) in enumerate(main_plots):
      r, c = 1 + (i // 4), i % 4
      if filename: btn = ctk.CTkButton(self, text=btn_text, font=("Times New Roman", 16), height=36, command=lambda f=filename, t=title: self.open_plot_window(f, t))
      else: btn = ctk.CTkButton(self, text=btn_text, font=("Times New Roman", 16), height=36, command=self.open_profile_gallery)
      btn.grid(row=r, column=c, padx=8, pady=(10, 10), sticky="ew")

  #function that sets up the reference simulation
  def setup_reference_simulation(self):
    #configures the grid for the reference simulation
    self.main_label.configure(text="Main complete, running reference simulation\nThis may take a long time", font=("Times New Roman", 38))
    self.ref_progress_tracker = ProgressTracker()
    self.ref_progress_label = ctk.CTkLabel(self, text="Reference Progress: 0/0", font=("Times New Roman", 24))
    self.ref_progress_label.grid(row=8, column=0, columnspan=4, pady=(10, 5), sticky="ew")
    self.ref_progress_bar = ctk.CTkProgressBar(self, orientation="horizontal", width=500, height=24, corner_radius=10)
    self.ref_progress_bar.set(0)
    self.ref_progress_bar.grid(row=9, column=0, columnspan=4, pady=(0, 15), sticky="n")

    #starts the reference simulation
    self.simulation_running = True
    self.update_progress()
    thread = threading.Thread(target=self.run_reference_thread, daemon=True)
    thread.start()

  #function that runs the reference simulation in a separate thread
  def run_reference_thread(self):
    try:
      ref_result = run_reference_only(self.yaml_file, self.ref_progress_tracker)
      self.latest_run_result["reference"] = ref_result
      self.after(0, self.after_reference_simulation)
    except Exception as e:
      self.after(0, lambda err=str(e): self.show_error(err))

  #function that runs after reference simulation completes to create comparison visuals
  def after_reference_simulation(self):
    #stops the reference simulation
    self.simulation_running = False
    if hasattr(self, "ref_progress_bar") and self.ref_progress_bar.winfo_exists(): self.ref_progress_bar.destroy()
    if hasattr(self, "ref_progress_label") and self.ref_progress_label.winfo_exists(): self.ref_progress_label.destroy()

    #configures the main label for comparison visuals
    self.main_label.configure(text="Creating comparison visualizations", font=("Times New Roman", 54))
    self.update()
    thread = threading.Thread(target=self.create_comparison_visuals_thread, daemon=True)
    thread.start()

  #function that creates comparison visuals in a separate thread
  def create_comparison_visuals_thread(self):
    try:
      #generates the comparison visuals
      theme = ctk.get_appearance_mode().lower()
      generate_comparison_visuals(self.latest_run_result, self.latest_visual_output_dir, theme=theme)

      #finalizes the simulation
      self.after(0, self.finalize_simulation)
    except Exception as e:
      self.after(0, lambda err=str(e): self.show_error(err))

  #function that adds the action buttons
  def _add_action_buttons(self):
    #configures the button to export all graphs
    export_graphs_btn = ctk.CTkButton(self, text="Export All Graphs", font=("Times New Roman", 18), height=40, command=self.export_all_graphs)
    export_graphs_btn.grid(row=5, column=0, columnspan=2, padx=8, pady=(14, 10), sticky="ew")

    #configures the button to export all data
    export_data_btn = ctk.CTkButton(self, text="Export All Data", font=("Times New Roman", 18), height=40, command=self.export_all_data)
    export_data_btn.grid(row=5, column=2, columnspan=2, padx=8, pady=(14, 10), sticky="ew")

    #configures the button to restart the application
    restart_btn = ctk.CTkButton(self, text="Restart", font=("Times New Roman", 18), height=40, command=self.restart_app)
    restart_btn.grid(row=6, column=0, columnspan=2, padx=8, pady=(6, 18), sticky="ew")

    #configures the button to exit the application
    exit_btn = ctk.CTkButton(self, text="Exit", font=("Times New Roman", 18), height=40, command=self.exit_app)
    exit_btn.grid(row=6, column=2, columnspan=2, padx=8, pady=(6, 18), sticky="ew")

  #function that finalizes the simulation screen with comparison and action buttons
  def finalize_simulation(self):
    self.main_label.configure(text="Simulation Results", font=("Times New Roman", 42))

    #creates the buttons for the comparison plots
    comp_plots = [
      ("L2 error vs time", "nls_l2_error_vs_time.png", "Main vs reference L2 error"),
      ("Phase error vs time", "nls_phase_error_vs_time.png", "Phase-aligned error"),
      ("Invariant bias", "nls_invariant_bias_main_vs_ref.png", "Mass/energy drift bias"),
    ]
    for i, (btn_text, filename, title) in enumerate(comp_plots):
      btn = ctk.CTkButton(self, text=btn_text, font=("Times New Roman", 16), height=36, command=lambda f=filename, t=title: self.open_plot_window(f, t))
      btn.grid(row=3, column=i, padx=8, pady=(10, 10), sticky="ew")
    self._add_action_buttons()

  #function that finalizes the simulation screen when reference is disabled 
  def finalize_simulation_no_ref(self): self._add_action_buttons()

  #function that opens a plot in a popup window
  def open_plot_window(self, filename, title):

    #checks if the graphs directory exists
    if not hasattr(self, "graphs_dir") or not self.graphs_dir:
      return

    #checks if the images directory exists
    img_path = os.path.join(self.graphs_dir, filename)
    if not os.path.exists(img_path):
      self.main_label.configure(text=f"Graph not found: {filename}")
      return

    #opens the image with pillow and adjust it
    pil_img = Image.open(img_path)
    img_width, img_height = pil_img.size
    display_height = 500
    aspect_ratio = img_width / img_height
    display_width = min(int(display_height * aspect_ratio), 1200)

    #creates a top level window for the image
    win = ctk.CTkToplevel(self)
    win.title(title)
    win.geometry(f"{display_width + 20}x520")
    win.resizable(False, False)
    img = CTkImage(light_image=pil_img, dark_image=pil_img, size=(display_width, display_height))
    panel = ctk.CTkLabel(win, image=img, text="")
    panel.image = img
    panel.pack(pady=5)

  #function that opens the profile gallery folder in the file manager
  def open_profile_gallery(self):
    gallery_dir = os.path.join(self.graphs_dir, "profiles_gallery")

    #checks if the profile gallery directory exists
    if not os.path.exists(gallery_dir):
      self.main_label.configure(text="Profile gallery not found")
      return
    try: os.startfile(gallery_dir)
    except Exception as e: self.main_label.configure(text=f"Cannot open folder: {e}")

  #function that exports all graphs to a zip file
  def export_all_graphs(self):
    #checks if the plots directory exists
    plots_dir = getattr(self, "graphs_dir", None) or (os.path.join(self.latest_visual_output_dir, "plots") if self.latest_visual_output_dir else "")
    if not plots_dir or not os.path.exists(plots_dir) or not os.listdir(plots_dir):
      self.main_label.configure(text="No graphs to export")
      self.after(3000, lambda: self.main_label.configure(text="Simulation Results"))
      return

    #asks user to select the destination using a file dialog
    zip_path = filedialog.asksaveasfilename(defaultextension=".zip", initialfile="NLS_Graphs.zip", title="Save All Graphs", filetypes=[("ZIP Archive", "*.zip")])
    if not zip_path: return
    with zipfile.ZipFile(zip_path, "w") as z:
      for root, dirs, files in os.walk(plots_dir):
        for file in files:
          if file.endswith(".png"): z.write(os.path.join(root, file), arcname=os.path.relpath(os.path.join(root, file), plots_dir))

  #function that exports all data to a zip file
  def export_all_data(self):
    #asks the user to select the destination using a file dialog
    zip_path = filedialog.asksaveasfilename(defaultextension=".zip", initialfile="NLS_Data.zip", title="Save All Data", filetypes=[("ZIP Archive", "*.zip")])
    if not zip_path: return

    #creates a temporary directory to store the data
    temp_dir = os.path.join(self.latest_visual_output_dir, "export_temp")
    os.makedirs(temp_dir, exist_ok=True)

    #saves the trajectory data
    r = self.latest_run_result
    np.savez_compressed(os.path.join(temp_dir, "main_trajectory.npz"), t_grid=r["main"]["t_grid"], Y_traj=r["main"]["Y_traj"])
    if r.get("reference") is not None: np.savez_compressed(os.path.join(temp_dir, "reference_trajectory.npz"), t_grid=r["reference"]["t_grid"], Y_traj=r["reference"]["Y_traj"])
    cfg = r["config"]
    params = {"L": cfg.domain.L, "N": cfg.domain.N, "kappa": cfg.pde.kappa, "dt": cfg.time.dt, "n_steps": cfg.time.n_steps, "strategy": cfg.strategy.name}
    np.savez(os.path.join(temp_dir, "simulation_parameters.npz"), **params)
    with zipfile.ZipFile(zip_path, "w") as zipf:
      for file in os.listdir(temp_dir):
        zipf.write(os.path.join(temp_dir, file), arcname=file)
    for file in os.listdir(temp_dir):
      os.remove(os.path.join(temp_dir, file))
    os.rmdir(temp_dir)

  #function that restarts the application (subprocess re-launch like CLE)
  def restart_app(self):
    self.destroy()
    subprocess.Popen([sys.executable, sys.argv[0]])

  #function that displays a run-time simulation error
  def show_error(self, error_msg):
    self.simulation_running = False
    self.run_stage = "error"

    #removes the progress bars and configures the gui
    for name in ("main_progress_bar", "main_progress_label", "ref_progress_bar", "ref_progress_label"):
      w = getattr(self, name, None)
      if w is not None and hasattr(w, "winfo_exists") and w.winfo_exists():
        w.destroy()
    for j in range(4): self.grid_columnconfigure(j, weight=1)

    #configures the main label with the error message
    self.main_label.configure(text=f"Error running simulation:\n{error_msg}", font=("Times New Roman", 26))
    self.main_label.grid(row=0, column=0, columnspan=4, pady=20, sticky="ew")

    #creates the buttons to restart and exit the application
    restart_btn = ctk.CTkButton(self, text="Restart", font=("Times New Roman", 18), height=40, command=self.restart_app)
    restart_btn.grid(row=1, column=0, columnspan=2, padx=10, pady=10)
    exit_btn = ctk.CTkButton(self, text="Exit", font=("Times New Roman", 18), height=40, command=self.exit_app)
    exit_btn.grid(row=1, column=2, columnspan=2, padx=10, pady=10)

  #function that exits the application
  def exit_app(self): self.destroy()

#runs the application
if __name__ == "__main__":
  app = NLSApp()
  ctk.set_appearance_mode("system")
  app.update_text_editor_colors()
  app.mainloop()