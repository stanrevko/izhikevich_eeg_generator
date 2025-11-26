import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import multiprocessing
import time
from eeg_generator import IzhikevichEEGGenerator
import queue

# Constants
FS = 1000  # Sampling frequency (Hz)
CHUNK_SIZE = 100  # Samples per chunk (100ms)
BUFFER_SIZE = 2000  # Display buffer size (2 seconds)
FFT_WINDOW = 1024   # Window size for FFT

def generator_process(control_queue, data_queue):
    """
    Process that runs the EEG generator.
    """
    # Initialize generator with default seed
    gen = IzhikevichEEGGenerator(n_exc=800, n_inh=200, seed=42)
    
    # Default parameters
    params = {
        "thalamic_mean_exc": 5.0,
        "thalamic_mean_inh": 2.0,
        "noise_level": 0.0,
        "alpha_drive_amplitude": 0.0,
        "alpha_drive_freq": 8.0,
        "running": False
    }
    
    buffer = []
    
    while True:
        # Check for control messages
        try:
            while True:
                msg = control_queue.get_nowait()
                if msg["type"] == "STOP_PROCESS":
                    return
                elif msg["type"] == "UPDATE_PARAMS":
                    params.update(msg["params"])
                    gen.measurement_noise_std = params["noise_level"]
                elif msg["type"] == "RESET":
                    gen.reset()
                elif msg["type"] == "WARMUP":
                    gen.warmup(duration_ms=2000)
                elif msg["type"] == "REINIT":
                    # Reinitialize generator with new seed
                    seed = msg.get("seed", None)
                    gen = IzhikevichEEGGenerator(n_exc=800, n_inh=200, seed=seed)
                    gen.measurement_noise_std = params["noise_level"]
        except queue.Empty:
            pass
        
        if params["running"]:
            # Generate a chunk of data
            chunk = []
            for _ in range(CHUNK_SIZE):
                val = gen.step(
                    thalamic_mean_exc=params["thalamic_mean_exc"],
                    thalamic_mean_inh=params["thalamic_mean_inh"],
                    alpha_drive_amplitude=params["alpha_drive_amplitude"],
                    alpha_drive_freq=params["alpha_drive_freq"]
                )
                chunk.append(val)
            
            # Send data to UI
            try:
                data_queue.put(chunk)
            except queue.Full:
                pass  # Drop data if UI is slow
            
            # Throttle to approx real time
            time.sleep(0.09) 
        else:
            time.sleep(0.1)

class EEGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Izhikevich EEG Generator")
        self.root.geometry("1200x800")
        
        # Multiprocessing setup
        self.control_queue = multiprocessing.Queue()
        self.data_queue = multiprocessing.Queue(maxsize=10)
        self.process = multiprocessing.Process(
            target=generator_process, 
            args=(self.control_queue, self.data_queue)
        )
        self.process.daemon = True
        self.process.start()
        
        # Data buffers
        self.time_data = np.zeros(BUFFER_SIZE)
        self.ptr = 0
        self.running = False
        
        self._setup_ui()
        self._start_update_loop()
        
    def _setup_ui(self):
        # Main layout: Side panel (Left) + Plots (Right)
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Side Panel
        side_panel = ttk.Frame(main_frame, width=250, padding=10)
        side_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        ttk.Label(side_panel, text="Controls", font=("Helvetica", 16, "bold")).pack(pady=10)
        
        # Start/Stop Button
        self.btn_start = ttk.Button(side_panel, text="Start", command=self.toggle_start)
        self.btn_start.pack(fill=tk.X, pady=5)
        
        self.btn_reset = ttk.Button(side_panel, text="Reset", command=self.reset_generator)
        self.btn_reset.pack(fill=tk.X, pady=5)
        
        ttk.Separator(side_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Seed control
        seed_frame = ttk.Frame(side_panel)
        seed_frame.pack(fill=tk.X, pady=5)
        ttk.Label(seed_frame, text="Random Seed:").pack(anchor="w")
        
        seed_input_frame = ttk.Frame(seed_frame)
        seed_input_frame.pack(fill=tk.X)
        
        self.seed_var = tk.StringVar(value="42")
        self.seed_entry = ttk.Entry(seed_input_frame, textvariable=self.seed_var, width=10)
        self.seed_entry.pack(side=tk.LEFT, padx=(0, 5))
        
        ttk.Button(seed_input_frame, text="Apply Seed", command=self.apply_seed).pack(side=tk.LEFT)
        
        ttk.Label(seed_frame, text="(empty = random)", font=("Helvetica", 8)).pack(anchor="w")
        
        ttk.Separator(side_panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # Parameters
        self.params = {
            "thalamic_mean_exc": tk.DoubleVar(value=5.0),
            "thalamic_mean_inh": tk.DoubleVar(value=2.0),
            "noise_level": tk.DoubleVar(value=0.0),
            "alpha_drive_amplitude": tk.DoubleVar(value=0.0),
            "alpha_drive_freq": tk.DoubleVar(value=10.0)
        }
        
        self._create_slider(side_panel, "Input (Exc)", "thalamic_mean_exc", 0.0, 20.0)
        self._create_slider(side_panel, "Input (Inh)", "thalamic_mean_inh", 0.0, 20.0)
        self._create_slider(side_panel, "Noise Level", "noise_level", 0.0, 5.0)
        self._create_slider(side_panel, "Alpha Drive Amplitude", "alpha_drive_amplitude", 0.0, 2.0)
        self._create_slider(side_panel, "Alpha Drive Freq (Hz)", "alpha_drive_freq", 8.0, 12.0)
        
        # Plot Area
        plot_frame = ttk.Frame(main_frame)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Matplotlib Figures
        self.fig = plt.Figure(figsize=(8, 8), dpi=100)
        self.ax_time = self.fig.add_subplot(211)
        self.ax_freq = self.fig.add_subplot(212)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial Plot Setup
        # X-axis: -2s to 0s
        self.x_data = np.linspace(-BUFFER_SIZE/FS, 0, BUFFER_SIZE)
        self.line_time, = self.ax_time.plot(self.x_data, self.time_data, 'b-', lw=1)
        self.ax_time.set_title("Real-time EEG Signal")
        self.ax_time.set_ylabel("Amplitude (mV)")
        self.ax_time.set_xlabel("Time (s)")
        self.ax_time.set_ylim(-1000, 1000)  # Fixed range as requested
        self.ax_time.grid(True)
        
        self.line_freq, = self.ax_freq.plot([], [], 'k-', lw=1)
        self.ax_freq.set_title("Power Spectral Density")
        self.ax_freq.set_ylabel("Power (dB)")
        self.ax_freq.set_xlabel("Frequency (Hz)")
        self.ax_freq.set_xlim(0, 50)
        self.ax_freq.set_ylim(0, 100)
        self.ax_freq.grid(True)
        
        # Highlight Alpha Band (8-12 Hz)
        self.ax_freq.axvspan(8, 12, color='yellow', alpha=0.3, label='Alpha (8-12 Hz)')
        self.ax_freq.legend()
        
        # Add spacing between subplots
        self.fig.subplots_adjust(hspace=0.4)
        
    def _create_slider(self, parent, label, param_key, min_val, max_val):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)
        ttk.Label(frame, text=label).pack(anchor="w")
        
        # Value label with 2 decimal rounding
        lbl_val = ttk.Label(frame, text=f"{self.params[param_key].get():.2f}")
        lbl_val.pack(anchor="e")
        
        def on_slide(val):
            # Round value to 2 decimals
            rounded_val = round(float(val), 2)
            self.params[param_key].set(rounded_val)
            lbl_val.config(text=f"{rounded_val:.2f}")
            self.update_params()

        scale = ttk.Scale(
            frame, from_=min_val, to=max_val, 
            variable=self.params[param_key], 
            command=on_slide
        )
        scale.pack(fill=tk.X)

    def toggle_start(self):
        self.running = not self.running
        self.btn_start.config(text="Stop" if self.running else "Start")
        
        if self.running:
            # Send warmup command before starting
            self.control_queue.put({"type": "WARMUP"})
            
        self.control_queue.put({
            "type": "UPDATE_PARAMS",
            "params": {"running": self.running}
        })
        
    def reset_generator(self):
        self.time_data[:] = 0
        # Clear any pending data in the queue to avoid "old" data appearing
        try:
            while True:
                self.data_queue.get_nowait()
        except queue.Empty:
            pass
            
        self.control_queue.put({"type": "RESET"})
        self.control_queue.put({"type": "WARMUP"})
    
    def apply_seed(self):
        """Apply new seed to generator."""
        seed_str = self.seed_var.get().strip()
        
        if seed_str == "":
            seed = None
        else:
            try:
                seed = int(seed_str)
            except ValueError:
                # Invalid seed, ignore
                return
        
        # Clear data
        self.time_data[:] = 0
        try:
            while True:
                self.data_queue.get_nowait()
        except queue.Empty:
            pass
        
        # Reinitialize generator with new seed
        self.control_queue.put({"type": "REINIT", "seed": seed})
        self.control_queue.put({"type": "WARMUP"})
        
    def update_params(self):
        params = {
            "thalamic_mean_exc": round(self.params["thalamic_mean_exc"].get(), 2),
            "thalamic_mean_inh": round(self.params["thalamic_mean_inh"].get(), 2),
            "noise_level": round(self.params["noise_level"].get(), 2),
            "alpha_drive_amplitude": round(self.params["alpha_drive_amplitude"].get(), 2),
            "alpha_drive_freq": round(self.params["alpha_drive_freq"].get(), 2),
            "running": self.running
        }
        self.control_queue.put({
            "type": "UPDATE_PARAMS",
            "params": params
        })
        
    def _start_update_loop(self):
        self._process_data()
        self.root.after(50, self._start_update_loop) # 20Hz update
        
    def _process_data(self):
        # Read all available data from queue
        try:
            while True:
                chunk = self.data_queue.get_nowait()
                chunk_arr = np.array(chunk)
                
                # Roll buffer and add new data
                self.time_data = np.roll(self.time_data, -len(chunk_arr))
                self.time_data[-len(chunk_arr):] = chunk_arr
        except queue.Empty:
            pass
        
        # Update Plots
        if self.running:
            # Time Domain
            self.line_time.set_ydata(self.time_data)
            
            # Auto-scale Y if needed (optional, but good for stability)
            # self.ax_time.relim()
            # self.ax_time.autoscale_view()
            
            # Frequency Domain (FFT)
            # Use last FFT_WINDOW samples
            if len(self.time_data) >= FFT_WINDOW:
                data_for_fft = self.time_data[-FFT_WINDOW:]
                # Windowing
                window = np.hanning(len(data_for_fft))
                fft_vals = np.fft.rfft(data_for_fft * window)
                fft_freq = np.fft.rfftfreq(len(data_for_fft), 1/FS)
                
                power = np.abs(fft_vals)**2
                # Convert to dB
                power_db = 10 * np.log10(power + 1e-10)
                
                self.line_freq.set_data(fft_freq, power_db)
                
                # Auto-scale Y for freq
                max_p = np.max(power_db) if len(power_db) > 0 else 100
                self.ax_freq.set_ylim(0, max_p + 10)
            
            self.canvas.draw()

    def on_close(self):
        self.control_queue.put({"type": "STOP_PROCESS"})
        self.process.join(timeout=1)
        self.process.terminate()
        self.root.destroy()

if __name__ == "__main__":
    # Support for multiprocessing on macOS
    multiprocessing.set_start_method('spawn', force=True)
    
    root = tk.Tk()
    app = EEGApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
