"""
EEG Generator using Izhikevich Spiking Neural Network.

This module implements the cortical EEG generator from Davelaar (2018) Simulation Study 1.
The network consists of 800 excitatory and 200 inhibitory Izhikevich neurons that generate
an alpha-band EEG proxy signal through low-pass filtering of excitatory membrane potentials.

Reference:
    Davelaar, E.J. (2018). Mechanisms of Neurofeedback: A Computation-theoretic Approach.
    Neuroscience, 378, 175-188.
"""

import numpy as np
from typing import Optional, Tuple
from scipy.signal import butter, sosfilt, sosfilt_zi

# Try to import numba for JIT compilation (optional dependency)
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Time step in milliseconds (two half-steps per millisecond for stability)
DT_MS = 0.5


# JIT-compiled helper functions for performance
@jit(nopython=True, cache=True)
def _compute_synaptic_input_exc(w_ee, w_ie, spikes_exc, spikes_inh):
    """Compute synaptic input to excitatory neurons."""
    I_syn_ee = np.zeros(w_ee.shape[0])
    I_syn_ie = np.zeros(w_ie.shape[1])

    if np.any(spikes_exc):
        for i in range(w_ee.shape[0]):
            for j in range(w_ee.shape[1]):
                if spikes_exc[j]:
                    I_syn_ee[i] += w_ee[i, j]

    if np.any(spikes_inh):
        for i in range(w_ie.shape[1]):
            for j in range(w_ie.shape[0]):
                if spikes_inh[j]:
                    I_syn_ie[i] += w_ie[j, i]

    return I_syn_ee, I_syn_ie


@jit(nopython=True, cache=True)
def _compute_synaptic_input_inh(w_ei, w_ii, spikes_exc, spikes_inh):
    """Compute synaptic input to inhibitory neurons."""
    I_syn_ei = np.zeros(w_ei.shape[1])
    I_syn_ii = np.zeros(w_ii.shape[0])

    if np.any(spikes_exc):
        for i in range(w_ei.shape[1]):
            for j in range(w_ei.shape[0]):
                if spikes_exc[j]:
                    I_syn_ei[i] += w_ei[j, i]

    if np.any(spikes_inh):
        for i in range(w_ii.shape[0]):
            for j in range(w_ii.shape[1]):
                if spikes_inh[j]:
                    I_syn_ii[i] += w_ii[i, j]

    return I_syn_ei, I_syn_ii


@jit(nopython=True, cache=True)
def _update_neurons_exc(v, u, I_total, a, b, c, d, dt):
    """Update excitatory neurons (one half-step)."""
    # Compute derivatives
    dv = (0.04 * v**2 + 5 * v + 140 - u + I_total) * dt
    du = a * (b * v - u) * dt

    # Update state
    v_new = v + dv
    u_new = u + du

    # Handle spikes
    spikes = v_new >= 30
    if np.any(spikes):
        for i in range(len(v_new)):
            if spikes[i]:
                v_new[i] = c[i]
                u_new[i] += d[i]

    return v_new, u_new, spikes


@jit(nopython=True, cache=True)
def _update_neurons_inh(v, u, I_total, a, b, c, d, dt):
    """Update inhibitory neurons (one half-step)."""
    # Compute derivatives
    dv = (0.04 * v**2 + 5 * v + 140 - u + I_total) * dt
    du = a * (b * v - u) * dt

    # Update state
    v_new = v + dv
    u_new = u + du

    # Handle spikes
    spikes = v_new >= 30
    if np.any(spikes):
        for i in range(len(v_new)):
            if spikes[i]:
                v_new[i] = c[i]
                u_new[i] += d[i]

    return v_new, u_new, spikes


class IzhikevichEEGGenerator:
    """
    Izhikevich spiking neural network for EEG generation.
    
    Implements a network of 800 excitatory (regular spiking) and 200 inhibitory
    (fast spiking) neurons with all-to-all connectivity. The EEG signal is computed
    as a low-pass filtered sum of excitatory membrane potentials.
    """
    
    def __init__(
        self,
        n_exc: int = 800,
        n_inh: int = 200,
        seed: Optional[int] = None,
        measurement_noise_std: float = 0.0,
        track_spikes: bool = False,
        target_neuron_index: Optional[int] = None
    ):
        """
        Initialize the Izhikevich EEG generator network.
        
        Args:
            n_exc: Number of excitatory neurons (default: 800)
            n_inh: Number of inhibitory neurons (default: 200)
            seed: Random seed for reproducibility (None for random)
            measurement_noise_std: Standard deviation of additive white Gaussian noise
                                 added to EEG signal (default: 0.0, no noise)
            track_spikes: Whether to track spike times (default: False for performance)
            target_neuron_index: Index of target excitatory neuron (default: 0, for Davelaar model)
        """
        self.n_exc = n_exc
        self.n_inh = n_inh
        self.n_total = n_exc + n_inh
        self.measurement_noise_std = measurement_noise_std
        self.track_spikes = track_spikes
        
        # Target neuron (for Davelaar model) - one of the excitatory neurons
        if target_neuron_index is None:
            target_neuron_index = 0
        if target_neuron_index >= n_exc:
            raise ValueError(f"target_neuron_index must be < n_exc ({n_exc}), got {target_neuron_index}")
        self.target_neuron_index = target_neuron_index
        self.target_neuron_active = False  # Track if target neuron is currently active
        
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize EEG signal
        self.eeg_signal = 0.0
        
        # Initialize excitatory neuron parameters (Regular Spiking)
        r_e = np.random.rand(n_exc)
        self.a_exc = np.full(n_exc, 0.02)
        self.b_exc = np.full(n_exc, 0.2)
        self.c_exc = -65 + 15 * r_e**2  # Range: [-65, -50]
        self.d_exc = 8 - 6 * r_e**2      # Range: [2, 8]
        
        # Initialize inhibitory neuron parameters (Fast Spiking / Low Threshold Spiking)
        r_i = np.random.rand(n_inh)
        self.a_inh = 0.02 + 0.08 * r_i   # Range: [0.02, 0.10]
        self.b_inh = 0.25 - 0.05 * r_i   # Range: [0.20, 0.25]
        self.c_inh = np.full(n_inh, -65)
        self.d_inh = np.full(n_inh, 2)
        
        # Initialize membrane potentials and recovery variables
        self.v_exc = np.random.uniform(-65, -60, n_exc)
        self.u_exc = self.b_exc * self.v_exc
        
        self.v_inh = np.random.uniform(-65, -60, n_inh)
        self.u_inh = self.b_inh * self.v_inh
        
        # Initialize EEG signal to approximate steady-state value
        sum_v_exc_init = np.sum(self.v_exc)
        self.eeg_signal = sum_v_exc_init
        
        # Store initial values for reset
        self.v_exc_init = self.v_exc.copy()
        self.u_exc_init = self.u_exc.copy()
        self.v_inh_init = self.v_inh.copy()
        self.u_inh_init = self.u_inh.copy()
        
        # Create synapses (all-to-all connectivity)
        self.w_ee = np.random.uniform(0, 0.5, size=(n_exc, n_exc))
        self.w_ei = np.random.uniform(0, 0.5, size=(n_exc, n_inh))
        self.w_ie = -np.random.uniform(0, 1.0, size=(n_inh, n_exc))
        self.w_ii = -np.random.uniform(0, 1.0, size=(n_inh, n_inh))
        
        if self.track_spikes:
            self.spike_history_exc = []
            self.spike_history_inh = []
        else:
            self.spike_history_exc = None
            self.spike_history_inh = None
        self.current_time = 0.0
        
        # Initialize Bandpass Filter (1-50 Hz)
        fs = 1000.0
        lowcut = 1.0
        highcut = 50.0
        order = 4
        self.sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
        self.zi = sosfilt_zi(self.sos)
        
    def warmup(self, duration_ms: int = 1000) -> None:
        """
        Warm up the network to reduce initialization transients.
        
        Runs the network for a period without recording, allowing it to
        settle into a stable oscillatory state.
        
        Args:
            duration_ms: Duration of warm-up period in milliseconds (default: 1000)
        """
        # Run network without recording spikes
        for _ in range(duration_ms):
            self.step()
        
        # Clear spike history from warm-up
        if self.track_spikes:
            self.spike_history_exc = []
            self.spike_history_inh = []
        self.current_time = 0.0
        
        # Reset EEG signal baseline after warmup (though filter handles it)
        self.eeg_signal = 0.0

    def step(self, I: Optional[np.ndarray] = None, target_active: bool = False, 
             thalamic_mean_exc: float = 5.0, thalamic_mean_inh: float = 2.0,
             alpha_drive_amplitude: float = 0.0, alpha_drive_freq: float = 10.0) -> float:
        """
        Advance the network by one time step (1 ms) and update EEG signal.
        
        Args:
            I: Optional array of input currents for excitatory neurons (shape: (n_exc,)).
               If None, uses default baseline input (Gaussian noise).
            target_active: Whether the target neuron is active.
            thalamic_mean_exc: Mean of thalamic input to excitatory neurons (default 5.0).
            thalamic_mean_inh: Mean of thalamic input to inhibitory neurons (default 2.0).
            alpha_drive_amplitude: Amplitude of sinusoidal drive at alpha frequency (default 0.0).
            alpha_drive_freq: Frequency of oscillatory drive in Hz (default 10.0).
            
        Returns:
            Current EEG signal value (Bandpass filtered 1-50 Hz)
        """
        self.target_neuron_active = target_active
        
        # Calculate oscillatory drive at alpha frequency
        t_sec = self.current_time / 1000.0  # Convert ms to seconds
        alpha_drive = alpha_drive_amplitude * np.sin(2 * np.pi * alpha_drive_freq * t_sec)
        
        # Set input for excitatory neurons
        if I is not None:
            if len(I) != self.n_exc:
                raise ValueError(f"I must have length {self.n_exc}, got {len(I)}")
            I_exc = I.copy()
        else:
            # Default baseline input: Gaussian noise + alpha drive
            I_exc = np.random.normal(thalamic_mean_exc + alpha_drive, 1.0, self.n_exc)
        
        if target_active:
            I_exc[self.target_neuron_index] += 5.0
        
        # Inhibitory neurons get default baseline input
        I_inh = np.random.normal(thalamic_mean_inh, 1.0, self.n_inh)
        
        # Two half-step (0.5 ms) updates
        for _ in range(2):
            spikes_exc = self.v_exc >= 30
            spikes_inh = self.v_inh >= 30

            I_syn_ee, I_syn_ie = _compute_synaptic_input_exc(
                self.w_ee, self.w_ie, spikes_exc, spikes_inh
            )
            I_syn_ei, I_syn_ii = _compute_synaptic_input_inh(
                self.w_ei, self.w_ii, spikes_exc, spikes_inh
            )

            I_exc_total = I_exc + I_syn_ee + I_syn_ie
            I_inh_total = I_inh + I_syn_ei + I_syn_ii

            self.v_exc, self.u_exc, spike_mask_exc = _update_neurons_exc(
                self.v_exc, self.u_exc, I_exc_total,
                self.a_exc, self.b_exc, self.c_exc, self.d_exc, DT_MS
            )
            self.v_inh, self.u_inh, spike_mask_inh = _update_neurons_inh(
                self.v_inh, self.u_inh, I_inh_total,
                self.a_inh, self.b_inh, self.c_inh, self.d_inh, DT_MS
            )

            if self.track_spikes:
                if np.any(spike_mask_exc):
                    spike_indices = np.where(spike_mask_exc)[0]
                    for idx in spike_indices:
                        self.spike_history_exc.append((self.current_time, idx))
                if np.any(spike_mask_inh):
                    spike_indices = np.where(spike_mask_inh)[0]
                    for idx in spike_indices:
                        self.spike_history_inh.append((self.current_time, idx))

            self.current_time += DT_MS
        
        # Compute EEG signal
        # Raw sum of potentials
        raw_eeg = np.sum(self.v_exc)
        
        # Apply bandpass filter (1-50 Hz)
        # sosfilt expects array, so we wrap and unwrap
        filtered_eeg, self.zi = sosfilt(self.sos, [raw_eeg], zi=self.zi)
        self.eeg_signal = filtered_eeg[0]
        
        if self.measurement_noise_std > 0:
            measurement_noise = np.random.normal(0.0, self.measurement_noise_std)
            self.eeg_signal += measurement_noise
        
        return self.eeg_signal
    
    def reset(self) -> None:
        """Reset the network to initial state."""
        self.v_exc = self.v_exc_init.copy()
        self.u_exc = self.u_exc_init.copy()
        self.v_inh = self.v_inh_init.copy()
        self.u_inh = self.u_inh_init.copy()
        self.eeg_signal = 0.0
        self.current_time = 0.0
        
        # Reset filter state
        self.zi = sosfilt_zi(self.sos)
        
        if self.track_spikes:
            self.spike_history_exc = []
            self.spike_history_inh = []
