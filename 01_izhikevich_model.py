"""
Simple EEG signal generator using Izhikevich neuron model.

This script creates a network of 1000 neurons (800 excitatory, 200 inhibitory)
and generates realistic EEG-like signals that are saved to a file.
"""

from brian2 import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal
import os

# Set simulation preferences
prefs.codegen.target = 'numpy'  # Use numpy for faster compilation
seed(42)  # For reproducible results

def create_izhikevich_network(N_total=1000, exc_ratio=0.8):
    """
    Create Izhikevich neuron network with realistic parameters.
    
    Parameters:
    -----------
    N_total : int
        Total number of neurons
    exc_ratio : float
        Fraction of excitatory neurons
        
    Returns:
    --------
    neurons : NeuronGroup
        Brian2 neuron group
    synapses : Synapses
        Network connections
    """
    
    N_exc = int(N_total * exc_ratio)  # 800 excitatory
    N_inh = N_total - N_exc           # 200 inhibitory
    
    print(f"Creating network: {N_exc} excitatory, {N_inh} inhibitory neurons")
    
    # Izhikevich neuron model equations
    eqs = '''
    dv/dt = (0.04*v**2 + 5*v + 140 - u + I_syn + I_ext + I_noise)/ms : 1
    du/dt = a*(b*v - u)/ms : 1
    I_syn : 1      # Synaptic current
    I_ext : 1      # External driving current
    I_noise : 1    # Noise current
    a : 1          # Recovery time constant
    b : 1          # Sensitivity of recovery
    c : 1          # After-spike reset value for v
    d : 1          # After-spike increment for u
    '''
    
    # Create neuron group
    neurons = NeuronGroup(N_total, eqs, 
                         threshold='v >= 30', 
                         reset='v = c; u += d',
                         method='euler')
    
    # Set parameters for excitatory neurons (regular spiking)
    r_exc = np.random.rand(N_exc)
    neurons.a[:N_exc] = 0.02
    neurons.b[:N_exc] = 0.2
    neurons.c[:N_exc] = -65 + 15 * r_exc**2
    neurons.d[:N_exc] = 8 - 6 * r_exc**2
    
    # Set parameters for inhibitory neurons (fast spiking)
    r_inh = np.random.rand(N_inh)
    neurons.a[N_exc:] = 0.02 + 0.08 * r_inh
    neurons.b[N_exc:] = 0.25 - 0.05 * r_inh
    neurons.c[N_exc:] = -65
    neurons.d[N_exc:] = 2
    
    # Initialize membrane potentials
    neurons.v = -65 + 10 * np.random.randn(N_total)
    neurons.u = neurons.b * neurons.v
    
    # Set external currents (increased for more realistic spiking)
    neurons.I_ext = 8 + 3 * np.random.randn(N_total)  # Increased driving current
    
    # Add noise for realistic dynamics (increased)
    neurons.I_noise = 2.0 * np.random.randn(N_total)  # More noise for variability
    
    # Create synaptic connections
    synapses = create_network_connections(neurons, N_exc, N_inh)
    
    return neurons, synapses, N_exc, N_inh

def create_network_connections(neurons, N_exc, N_inh):
    """Create realistic synaptic connections."""
    
    N_total = len(neurons)
    
    # Excitatory to all connections (increased connectivity)
    syn_exc = Synapses(neurons[:N_exc], neurons,
                       'w : 1',
                       on_pre='I_syn_post += w')
    syn_exc.connect(p=0.15)  # Increased from 10% to 15%
    syn_exc.w = 0.8 + 0.3 * np.random.randn(len(syn_exc))  # Stronger weights
    
    # Inhibitory to all connections  
    syn_inh = Synapses(neurons[N_exc:], neurons,
                       'w : 1', 
                       on_pre='I_syn_post -= w')  # Inhibitory (negative)
    syn_inh.connect(p=0.25)  # Increased inhibitory connectivity
    syn_inh.w = 1.2 + 0.4 * np.random.randn(len(syn_inh))  # Balanced inhibition
    
    return [syn_exc, syn_inh]

def generate_eeg_signal(neurons, N_exc, duration=60.0, dt=0.1*ms, fs=1000):
    """
    Generate EEG signal from neural network activity.
    
    Parameters:
    -----------
    neurons : NeuronGroup
        Network neurons
    N_exc : int
        Number of excitatory neurons
    duration : float
        Simulation duration in seconds
    dt : Quantity
        Time step
    fs : int
        Sampling frequency for EEG
        
    Returns:
    --------
    times : np.ndarray
        Time points
    eeg_signal : np.ndarray
        Generated EEG signal in microvolts
    """
    
    print(f"Generating EEG signal for {duration} seconds...")
    
    # Setup monitoring
    state_mon = StateMonitor(neurons, 'v', record=range(N_exc), dt=1*ms)
    spike_mon = SpikeMonitor(neurons)
    
    # Add periodic noise updates to maintain activity
    @network_operation(dt=10*ms)
    def update_noise():
        neurons.I_noise = 2.0 * np.random.randn(len(neurons))
    
    # Add periodic modulation to simulate alpha rhythm
    @network_operation(dt=1*ms)
    def alpha_modulation():
        t = float(defaultclock.t / second)
        alpha_freq = 10  # Hz
        modulation = 3 * np.sin(2 * np.pi * alpha_freq * t)  # Increased amplitude
        neurons.I_ext[:N_exc] += modulation
    
    # Add slower oscillations (theta)
    @network_operation(dt=1*ms)
    def theta_modulation():
        t = float(defaultclock.t / second)
        theta_freq = 6  # Hz
        modulation = 1.5 * np.sin(2 * np.pi * theta_freq * t)  # Increased amplitude
        neurons.I_ext[:N_exc] += modulation
    
    # Run simulation with progress monitoring
    print("Running neural simulation...")
    defaultclock.dt = dt
    
    # Run in chunks to monitor spiking activity
    chunk_duration = 10  # seconds
    total_chunks = int(duration / chunk_duration)
    N_neurons = len(neurons)  # Get total number of neurons
    
    for chunk in range(total_chunks):
        print(f"  Simulating chunk {chunk+1}/{total_chunks} ({chunk*chunk_duration}-{(chunk+1)*chunk_duration}s)")
        run(chunk_duration * second, report='stdout')
        
        # Check spiking activity
        if len(spike_mon.t) > 0:
            current_spikes = len(spike_mon.t)
            avg_rate = current_spikes / ((chunk+1) * chunk_duration * N_neurons)
            print(f"    Spikes so far: {current_spikes:,}, Avg rate: {avg_rate*1000:.1f} Hz/neuron")
        else:
            print("    WARNING: No spikes detected yet!")
    
    print(f"Simulation complete. Total spikes: {len(spike_mon.t):,}")
    
    # Convert membrane potentials to EEG signal
    print("Converting neural activity to EEG...")
    
    # Sum excitatory membrane potentials (main EEG source)
    membrane_sum = np.sum(state_mon.v, axis=0)  # Sum across excitatory neurons
    
    # Scale to typical human EEG amplitude range (10-100 µV)
    eeg_raw = (membrane_sum / N_exc) / 500000.0  # Final scaling for realistic human EEG levels
    
    # Apply realistic EEG filtering
    times = np.array(state_mon.t / second)
    eeg_filtered = apply_eeg_filtering(eeg_raw, fs)
    
    # Add realistic EEG noise (proportional to signal)
    noise_level = 0.05 * np.std(eeg_filtered)  # Reduced noise level
    eeg_signal = eeg_filtered + noise_level * np.random.randn(len(eeg_filtered))
    
    # Convert to microvolts
    eeg_signal = eeg_signal * 1e6  # Convert to µV
    
    print(f"EEG signal generated: {len(eeg_signal)} samples")
    print(f"Signal range: {np.min(eeg_signal):.1f} to {np.max(eeg_signal):.1f} µV")
    
    return times, eeg_signal, spike_mon

def apply_eeg_filtering(signal_data, fs):
    """Apply realistic EEG bandpass filtering."""
    
    # Bandpass filter 0.5-100 Hz (typical EEG range)
    low = 0.5 / (fs / 2)
    high = 100 / (fs / 2)
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_signal = signal.filtfilt(b, a, signal_data)
    
    # Notch filter at 50 Hz (power line interference)
    b_notch, a_notch = signal.iirnotch(50, 30, fs)
    filtered_signal = signal.filtfilt(b_notch, a_notch, filtered_signal)
    
    return filtered_signal

def analyze_eeg_spectrum(times, eeg_signal, fs=1000):
    """Analyze frequency content of generated EEG."""
    
    # Compute power spectral density
    freqs, psd = signal.welch(eeg_signal, fs=fs, nperseg=2048)
    
    # Extract band powers
    def get_band_power(freqs, psd, band):
        mask = (freqs >= band[0]) & (freqs <= band[1])
        return np.mean(psd[mask])
    
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8), 
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }
    
    band_powers = {name: get_band_power(freqs, psd, band) 
                   for name, band in bands.items()}
    
    # Find peak alpha frequency
    alpha_mask = (freqs >= 8) & (freqs <= 13)
    alpha_freqs = freqs[alpha_mask]
    alpha_psd = psd[alpha_mask]
    peak_alpha_freq = alpha_freqs[np.argmax(alpha_psd)]
    
    return freqs, psd, band_powers, peak_alpha_freq

def save_results(times, eeg_signal, spike_mon, freqs, psd, band_powers, 
                peak_alpha_freq, output_dir='./eeg_output'):
    """Save all results to files in specified formats."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Save EEG signal in the requested format (hh:mm:ss.mmm format with Fcz electrode)
    eeg_formatted_file = os.path.join(output_dir, 'eeg_signal_fcz.txt')
    
    # Downsample to 500 Hz (every 2ms) to match the requested format
    downsample_factor = 2
    times_ds = times[::downsample_factor]
    eeg_signal_ds = eeg_signal[::downsample_factor]
    
    with open(eeg_formatted_file, 'w') as f:
        # Write header
        f.write("hh:mm:ss.mmm\tFcz\n")
        
        # Write data in requested format
        for i, (t, eeg_val) in enumerate(zip(times_ds, eeg_signal_ds)):
            # Convert time to hh:mm:ss.mmm format
            hours = int(t // 3600)
            minutes = int((t % 3600) // 60)
            seconds = int(t % 60)
            milliseconds = int((t % 1) * 1000)
            
            time_str = f"{hours}:{minutes}:{seconds}.{milliseconds:03d}"
            f.write(f"{time_str}\t{eeg_val:.6f}\n")
    
    print(f"EEG signal (Fcz format) saved to: {eeg_formatted_file}")
    
    # Also save original CSV format for compatibility
    eeg_df = pd.DataFrame({
        'time_s': times,
        'eeg_signal_uv': eeg_signal
    })
    eeg_csv_file = os.path.join(output_dir, 'eeg_signal.csv')
    eeg_df.to_csv(eeg_csv_file, index=False)
    print(f"EEG signal (CSV format) saved to: {eeg_csv_file}")
    
    # Save spectral analysis
    spectral_df = pd.DataFrame({
        'frequency_hz': freqs,
        'power_spectral_density': psd
    })
    spectral_file = os.path.join(output_dir, 'eeg_spectrum.csv')
    spectral_df.to_csv(spectral_file, index=False)
    print(f"Spectral analysis saved to: {spectral_file}")
    
    # Save band powers and summary statistics
    summary = {
        'peak_alpha_frequency_hz': peak_alpha_freq,
        'signal_duration_s': times[-1],
        'sampling_rate_original_hz': len(times) / times[-1],
        'sampling_rate_fcz_format_hz': len(times_ds) / times_ds[-1],
        'signal_mean_uv': np.mean(eeg_signal),
        'signal_std_uv': np.std(eeg_signal),
        'signal_range_uv': np.ptp(eeg_signal),
        'total_spikes': len(spike_mon.t),
        'average_firing_rate_hz': len(spike_mon.t) / (times[-1] * len(spike_mon.source)),
        **{f'{band}_power': power for band, power in band_powers.items()}
    }
    
    summary_df = pd.DataFrame([summary])
    summary_file = os.path.join(output_dir, 'eeg_summary.csv')
    summary_df.to_csv(summary_file, index=False)
    print(f"Summary statistics saved to: {summary_file}")
    
    return eeg_formatted_file, eeg_csv_file, spectral_file, summary_file

def plot_results(times, eeg_signal, freqs, psd, band_powers, peak_alpha_freq,
                spike_mon, N_neurons, output_dir='./eeg_output'):
    """Create comprehensive visualization of results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # EEG signal (first 10 seconds)
    time_mask = times <= 10
    axes[0, 0].plot(times[time_mask], eeg_signal[time_mask], 'b-', linewidth=0.8)
    axes[0, 0].set_title('Generated EEG Signal (First 10s)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Amplitude (µV)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Power spectral density
    axes[0, 1].semilogy(freqs, psd, 'r-', linewidth=1.5)
    axes[0, 1].axvline(peak_alpha_freq, color='green', linestyle='--', 
                      label=f'Peak Alpha: {peak_alpha_freq:.1f} Hz')
    axes[0, 1].set_title('Power Spectral Density')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Power (µV²/Hz)')
    axes[0, 1].set_xlim(0, 50)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Band powers
    bands = list(band_powers.keys())
    powers = list(band_powers.values())
    colors = ['purple', 'orange', 'green', 'blue', 'red']
    
    bars = axes[0, 2].bar(bands, powers, color=colors, alpha=0.7)
    axes[0, 2].set_title('EEG Band Powers')
    axes[0, 2].set_ylabel('Power')
    axes[0, 2].tick_params(axis='x', rotation=45)
    
    # Add values on bars
    for bar, power in zip(bars, powers):
        height = bar.get_height()
        axes[0, 2].text(bar.get_x() + bar.get_width()/2., height,
                       f'{power:.3f}', ha='center', va='bottom', fontsize=9)
    
    # Spike raster plot (show more spikes and better time window)
    if len(spike_mon.t) > 0:
        # Show spikes from first 5 seconds for better visibility
        time_window = 5.0  # seconds
        time_mask = spike_mon.t <= time_window * second
        
        if np.any(time_mask):
            spike_times = spike_mon.t[time_mask] / second
            spike_neurons = spike_mon.i[time_mask]
            
            print(f"Plotting {len(spike_times):,} spikes in first {time_window}s")
            
            axes[1, 0].scatter(spike_times, spike_neurons, s=1, alpha=0.8, c='black')
            axes[1, 0].set_title(f'Neural Spike Raster (First {time_window}s, {len(spike_times):,} spikes)')
            axes[1, 0].set_xlabel('Time (s)')
            axes[1, 0].set_ylabel('Neuron Index')
            axes[1, 0].set_xlim(0, time_window)
            axes[1, 0].set_ylim(0, N_neurons-1)
        else:
            axes[1, 0].text(0.5, 0.5, f'No spikes in first {time_window}s', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Neural Spike Raster - NO ACTIVITY')
    else:
        axes[1, 0].text(0.5, 0.5, 'No spikes recorded!', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Neural Spike Raster - NO SPIKES')
    
    # Spectrogram
    f, t, Sxx = signal.spectrogram(eeg_signal[:10000], fs=1000, nperseg=256, noverlap=128)
    im = axes[1, 1].pcolormesh(t, f, 10 * np.log10(Sxx + 1e-10), 
                              shading='gouraud', cmap='viridis')
    axes[1, 1].set_title('EEG Spectrogram (First 10s)')
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].set_ylabel('Frequency (Hz)')
    axes[1, 1].set_ylim(0, 50)
    plt.colorbar(im, ax=axes[1, 1], label='Power (dB)')
    
    # Signal statistics
    stats_text = f"""
    Duration: {times[-1]:.1f} s
    Samples: {len(eeg_signal):,}
    Mean: {np.mean(eeg_signal):.2f} µV
    Std: {np.std(eeg_signal):.2f} µV
    Range: {np.ptp(eeg_signal):.2f} µV
    Peak Alpha: {peak_alpha_freq:.1f} Hz
    Total Spikes: {len(spike_mon.t):,}
    Avg Rate: {len(spike_mon.t)/(times[-1]*N_neurons):.1f} Hz/neuron
    """
    
    axes[1, 2].text(0.1, 0.9, stats_text, transform=axes[1, 2].transAxes,
                   fontsize=11, verticalalignment='top', fontfamily='monospace')
    axes[1, 2].set_title('Signal Statistics')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, 'eeg_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Analysis plot saved to: {plot_file}")
    plt.show()

def main():
    """Main function to run EEG generation simulation."""
    
    print("=" * 60)
    print("EEG Signal Generation using Izhikevich Neuron Model")
    print("=" * 60)
    
    # Configuration
    N_TOTAL = 1000
    EXC_RATIO = 0.8
    DURATION = 60.0  # seconds
    OUTPUT_DIR = './eeg_output'
    
    print(f"Configuration:")
    print(f"  Total neurons: {N_TOTAL}")
    print(f"  Excitatory ratio: {EXC_RATIO}")
    print(f"  Duration: {DURATION} seconds")
    print(f"  Output directory: {OUTPUT_DIR}")
    print()
    
    # Create network
    start_time = time.time()
    neurons, synapses, N_exc, N_inh = create_izhikevich_network(N_TOTAL, EXC_RATIO)
    
    # Generate EEG signal
    times, eeg_signal, spike_mon = generate_eeg_signal(neurons, N_exc, DURATION)
    
    # Analyze spectrum
    freqs, psd, band_powers, peak_alpha_freq = analyze_eeg_spectrum(times, eeg_signal)
    
    # Save results
    eeg_fcz_file, eeg_csv_file, spectral_file, summary_file = save_results(
        times, eeg_signal, spike_mon, freqs, psd, band_powers, 
        peak_alpha_freq, OUTPUT_DIR)
    
    # Create plots
    plot_results(times, eeg_signal, freqs, psd, band_powers, peak_alpha_freq,
                spike_mon, N_TOTAL, OUTPUT_DIR)
    
    # Performance summary
    elapsed_time = time.time() - start_time
    print()
    print("=" * 60)
    print("SIMULATION COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Simulation time: {elapsed_time:.1f} seconds")
    print(f"Generated {len(eeg_signal):,} EEG samples ({times[-1]:.1f} s recording)")
    print(f"Original sampling rate: 1000 Hz")
    print(f"Fcz format sampling rate: 500 Hz (downsampled)")
    print(f"Peak alpha frequency: {peak_alpha_freq:.1f} Hz")
    print(f"Alpha power: {band_powers['alpha']:.4f}")
    print(f"Total spikes recorded: {len(spike_mon.t):,}")
    print()
    print("Output files:")
    print(f"  EEG data (Fcz format): {eeg_fcz_file}")
    print(f"  EEG data (CSV format): {eeg_csv_file}")
    print(f"  Spectrum: {spectral_file}")
    print(f"  Summary: {summary_file}")
    print(f"  Plot: {os.path.join(OUTPUT_DIR, 'eeg_analysis.png')}")

if __name__ == "__main__":
    main()