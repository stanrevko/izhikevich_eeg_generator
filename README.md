# EEG Signal Generator using Izhikevich Neuron Model

Generate realistic EEG signals using a biologically plausible neural network model with 1000 Izhikevich neurons (800 excitatory, 200 inhibitory).

## Quick Start

```bash
# Install dependencies
pip install brian2 numpy scipy matplotlib pandas

# Run the script
python 01_izhikevich_model.py
```

## Overview

This script creates a network of 1000 neurons and generates realistic EEG-like signals with:
- **Prominent alpha rhythm** (~10 Hz)
- **Realistic amplitude** (50-100 μV)
- **Natural variability** and noise
- **Proper spectral characteristics**
- **Multiple output formats**

![eeg_analysis](https://github.com/user-attachments/assets/0956e2da-c787-439c-bf56-5ace3ca986c0)

## Output Files

The script generates files in `./eeg_output/`:

- `eeg_signal_fcz.txt` - **Main EEG data** in timestamp format (`hh:mm:ss.mmm	Fcz`)
- `eeg_signal.csv` - EEG data in CSV format (`time_s, eeg_signal_uv`)
- `eeg_spectrum.csv` - Power spectral density analysis
- `eeg_summary.csv` - Summary statistics and band powers
- `eeg_analysis.png` - Comprehensive visualization plots

## Frequency Control Parameters

### 🎵 **Direct Frequency Control (Most Important!)**

The primary way to control EEG frequencies is through rhythm modulation parameters:

```python
# Alpha rhythm (8-13 Hz) - DOMINANT
alpha_freq = 10                    # Alpha frequency (Hz)
alpha_amplitude = 3                # Alpha rhythm strength

# Theta rhythm (4-8 Hz)
theta_freq = 6                     # Theta frequency (Hz)  
theta_amplitude = 1.5              # Theta rhythm strength

# Applied in code as:
modulation = alpha_amplitude * sin(2π * alpha_freq * t)
```

### **Frequency Band Recipes:**

#### 🎯 **Strong Alpha Rhythm (8-12 Hz)**
```python
alpha_freq = 10          # Keep in alpha range
alpha_amplitude = 5      # Increase strength (from 3 to 5)
theta_amplitude = 0.5    # Reduce competing theta
```

#### 🚀 **Beta Rhythm Dominance (13-30 Hz)**
```python
alpha_freq = 20          # Move to beta range
alpha_amplitude = 4      # Increase amplitude
theta_amplitude = 0.5    # Reduce low frequencies
neurons.I_ext = 10 + 4 * np.random.randn()  # Higher excitation
```

#### 🧘 **Theta Rhythm Dominance (4-8 Hz)**
```python
alpha_freq = 6           # Move to theta range
theta_freq = 5           # Additional theta component
theta_amplitude = 3      # Increase theta amplitude
alpha_amplitude = 1      # Reduce alpha
neurons.I_ext = 6 + 2 * np.random.randn()  # Lower excitation
```

#### ⚡ **Gamma Activity (30-80 Hz)**
```python
# Add high-frequency modulation
@network_operation(dt=1*ms)
def gamma_modulation():
    t = float(defaultclock.t / second)
    gamma_freq = 40      # Gamma frequency
    modulation = 2 * np.sin(2 * np.pi * gamma_freq * t)
    neurons.I_ext[:N_exc] += modulation

# Plus fast neurons
neurons.a[:N_exc] = 0.04     # Very fast recovery
neurons.I_ext = 12 + 5 * np.random.randn()  # High excitation
```

### **Neuron Speed Parameters (Affect Natural Frequencies)**

```python
# Faster neurons → higher frequencies
neurons.a[:N_exc] = 0.03     # Increase from 0.02 (faster recovery)
neurons.b[:N_exc] = 0.25     # Increase from 0.2 (higher sensitivity)

# Slower neurons → lower frequencies
neurons.a[:N_exc] = 0.01     # Decrease (slower recovery)
neurons.b[:N_exc] = 0.15     # Decrease (lower sensitivity)
```

### **Network Synchronization (Affects Rhythm Clarity)**

```python
# Higher synchronization → clearer rhythms
syn_exc.connect(p=0.25)      # Increase from 0.15 (more connections)
syn_exc.w = 1.2              # Increase weights (from 0.8)

# Lower synchronization → more high frequencies
syn_exc.connect(p=0.08)      # Fewer connections
syn_exc.w = 0.5              # Lower weights
```

### **External Drive (Shifts Frequency Bands)**

```python
# Higher drive → higher frequencies (beta/gamma)
neurons.I_ext = 12 + 4 * np.random.randn()   # Increase from 8

# Lower drive → lower frequencies (alpha/theta)
neurons.I_ext = 5 + 2 * np.random.randn()    # Decrease to 5
```

## Frequency Band Reference

| **Rhythm** | **Frequency** | **Primary Parameter** | **Additional Settings** |
|------------|---------------|----------------------|------------------------|
| **Delta** | 0.5-4 Hz | `alpha_freq = 2` | Low `I_ext`, slow neurons |
| **Theta** | 4-8 Hz | `alpha_freq = 6` | Moderate `I_ext = 6` |
| **Alpha** | 8-13 Hz | `alpha_freq = 10` | Balanced `I_ext = 8` |
| **Beta** | 13-30 Hz | `alpha_freq = 20` | High `I_ext = 10-12` |
| **Gamma** | 30-100 Hz | Separate modulation | Very high `I_ext = 15+` |

## Advanced Frequency Control

### **Multiple Simultaneous Rhythms:**
```python
@network_operation(dt=1*ms)
def multi_rhythm_modulation():
    t = float(defaultclock.t / second)
    
    # Alpha (10 Hz) - primary
    alpha_mod = 3 * np.sin(2 * np.pi * 10 * t)
    
    # Theta (6 Hz) - background
    theta_mod = 1 * np.sin(2 * np.pi * 6 * t)
    
    # Beta (20 Hz) - weak
    beta_mod = 0.5 * np.sin(2 * np.pi * 20 * t)
    
    total_mod = alpha_mod + theta_mod + beta_mod
    neurons.I_ext[:N_exc] += total_mod
```

### **Dynamic Frequency Changes:**
```python
@network_operation(dt=100*ms)
def dynamic_frequency():
    global alpha_freq
    t = float(defaultclock.t / second)
    
    # Frequency varies over time (8-12 Hz)
    alpha_freq = 10 + 2 * np.sin(2 * np.pi * 0.1 * t)  # Slow changes
```

## Key Parameters That Control EEG Signal

### 🧠 **Network Structure**

```python
N_TOTAL = 1000        # Total number of neurons
EXC_RATIO = 0.8       # 80% excitatory, 20% inhibitory
DURATION = 60.0       # Recording duration (seconds)
```

**Effects:**
- **More neurons** → smoother, more stable EEG[Uploading eeg_signal_fcz.txt…]()

- **Higher excitatory ratio** → more excitable network
- **Longer duration** → better spectral resolution

### ⚡ **Neuron Model Properties**

#### Excitatory Neurons (Regular Spiking)
```python
neurons.a[:N_exc] = 0.02                    # Recovery time constant
neurons.b[:N_exc] = 0.2                     # Sensitivity to voltage
neurons.c[:N_exc] = -65 + 15 * r_exc**2     # Reset voltage (-65 to -50 mV)
neurons.d[:N_exc] = 8 - 6 * r_exc**2        # Reset recovery (2 to 8)
```

#### Inhibitory Neurons (Fast Spiking)
```python
neurons.a[N_exc:] = 0.02 + 0.08 * r_inh     # Faster recovery (0.02 to 0.10)
neurons.b[N_exc:] = 0.25 - 0.05 * r_inh     # Different sensitivity (0.20 to 0.25)
neurons.c[N_exc:] = -65                     # Fixed reset voltage
neurons.d[N_exc:] = 2                       # Lower reset recovery
```

**Parameter Effects:**
- **Lower `a`** → slower recovery, more sustained activity
- **Higher `b`** → more sensitive to voltage changes
- **`c` values** → control spike shape and baseline
- **`d` values** → affect post-spike dynamics

### 🔌 **Network Connectivity**

```python
# Excitatory connections
syn_exc.connect(p=0.15)                     # 15% connection probability
syn_exc.w = 0.8 + 0.3 * np.random.randn()  # Synaptic weights (mean ± std)

# Inhibitory connections
syn_inh.connect(p=0.25)                     # 25% connection probability
syn_inh.w = 1.2 + 0.4 * np.random.randn()  # Inhibitory weights
```

**Effects:**
- **Higher connectivity** → more synchronized activity
- **Stronger excitatory weights** → higher amplitude oscillations
- **Stronger inhibitory weights** → better rhythm control
- **E/I balance** → controls network stability

### 🎛️ **External Drive & Noise**

```python
neurons.I_ext = 8 + 3 * np.random.randn(N_total)  # External current (μA)
neurons.I_noise = 2.0 * np.random.randn(N_total)  # Noise current (μA)
```

**Effects:**
- **Higher I_ext** → more active network, higher firing rates
- **More I_noise** → more irregular, realistic patterns
- **Balance** → determines baseline activity level

### 🌊 **Rhythm Generation**

```python
# Alpha rhythm modulation
alpha_freq = 10                              # Alpha frequency (Hz)
alpha_amplitude = 3                          # Alpha modulation strength
modulation = alpha_amplitude * sin(2π * alpha_freq * t)

# Theta rhythm modulation  
theta_freq = 6                               # Theta frequency (Hz)
theta_amplitude = 1.5                        # Theta modulation strength
```

**Effects:**
- **alpha_freq (8-13 Hz)** → peak frequency in spectrum
- **alpha_amplitude** → strength of alpha rhythm in EEG
- **theta_freq (4-8 Hz)** → secondary peak frequency
- **theta_amplitude** → theta/alpha power ratio

### 📊 **Signal Processing**

```python
# EEG amplitude scaling
scaling_factor = 500000.0                    # Membrane potential → μV conversion
eeg_raw = (membrane_sum / N_exc) / scaling_factor

# Digital filtering
highpass_freq = 0.5      # Hz - removes DC drift
lowpass_freq = 100       # Hz - removes high-frequency noise  
notch_freq = 50          # Hz - removes power line interference
noise_amplitude = 0.05   # Relative noise level
```

**Effects:**
- **Smaller scaling_factor** → larger EEG amplitudes
- **Filter frequencies** → shape frequency spectrum
- **Noise level** → signal-to-noise ratio

## Parameter Tuning Guide

### 🎯 **To Increase Alpha Rhythm Strength:**
```python
alpha_amplitude = 5        # Increase from 3
alpha_freq = 10           # Keep at 10 Hz for peak alpha
syn_exc.w = 1.0          # Increase excitatory weights
```

### 📈 **To Increase EEG Amplitude:**
```python
scaling_factor = 250000   # Decrease from 500000 (doubles amplitude)
neurons.I_ext = 10 + 4 * np.random.randn()  # Increase drive
```

### 🔧 **To Make Signal More/Less Noisy:**
```python
# More realistic (noisier)
neurons.I_noise = 3.0
noise_amplitude = 0.1

# Cleaner signal
neurons.I_noise = 1.0  
noise_amplitude = 0.02
```

### ⚖️ **To Balance Network Activity:**
```python
# More active network
neurons.I_ext = 10        # Increase external drive
syn_exc.w = 1.0          # Stronger excitation

# More controlled network  
syn_inh.w = 1.5          # Stronger inhibition
syn_inh.connect(p=0.3)   # More inhibitory connections
```

### 🎵 **To Change Dominant Frequency:**
```python
# Beta rhythm (13-30 Hz)
alpha_freq = 20
alpha_amplitude = 4

# Theta rhythm (4-8 Hz)  
alpha_freq = 6
theta_amplitude = 0.5    # Reduce competing theta
```

## Signal Characteristics

### **Normal Output:**
- **Amplitude:** 50-100 μV (realistic human scalp EEG)
- **Dominant frequency:** ~10 Hz (alpha rhythm)
- **Sampling rate:** 1000 Hz (original), 500 Hz (Fcz format)
- **Duration:** 60 seconds
- **Signal-to-noise ratio:** ~20-30 dB

### **Frequency Bands:**
- **Delta (0.5-4 Hz):** Low power
- **Theta (4-8 Hz):** Moderate power  
- **Alpha (8-13 Hz):** **Dominant peak** 🎯
- **Beta (13-30 Hz):** Moderate power
- **Gamma (30-100 Hz):** Low power

## Troubleshooting

### **Problem: No alpha rhythm visible**
```python
# Increase alpha modulation
alpha_amplitude = 5       # Increase from 3
alpha_freq = 10          # Ensure it's in alpha band
```

### **Problem: Signal too noisy/chaotic**
```python
# Reduce excitation or increase inhibition
syn_exc.w = 0.6          # Reduce from 0.8
syn_inh.w = 1.5          # Increase from 1.2
neurons.I_noise = 1.0    # Reduce noise
```

### **Problem: Signal too flat/no activity**
```python
# Increase network drive
neurons.I_ext = 10 + 4 * np.random.randn()  # Increase drive
syn_exc.connect(p=0.2)   # More connections
syn_exc.w = 1.0          # Stronger weights
```

### **Problem: Amplitude too high/low**
```python
# For amplitude adjustment (most direct)
scaling_factor = 250000   # Decrease for higher amplitude
scaling_factor = 1000000  # Increase for lower amplitude
```

### **Problem: Wrong frequency dominance**
```python
# For specific frequency bands, use the recipes above
# Most important: change alpha_freq to desired frequency
alpha_freq = 8    # For theta-alpha border
alpha_freq = 12   # For alpha-beta border  
alpha_freq = 25   # For beta rhythm
```

## Advanced Modifications

### **Create Beta Rhythm Dominance:**
```python
alpha_freq = 20           # Move to beta range
alpha_amplitude = 4       # Increase strength
theta_amplitude = 0.5     # Reduce theta
```

### **Simulate Pathological Activity:**
```python
# Increase synchronization (epileptic-like)
syn_exc.connect(p=0.3)    # More connections
syn_exc.w = 1.5           # Stronger weights
syn_inh.w = 0.8           # Weaker inhibition
```

### **Create More Realistic Variability:**
```python
# Add random modulation
@network_operation(dt=100*ms)
def variable_modulation():
    global alpha_amplitude
    alpha_amplitude = 3 + 0.5 * np.random.randn()  # Varying alpha strength
```

## Technical Details

- **Neuron Model:** Izhikevich (2003) with mixed regular-spiking and fast-spiking types
- **Integration Method:** Euler method with dt = 0.1 ms
- **Connectivity:** Random sparse connectivity with realistic E/I ratios
- **EEG Generation:** Sum of excitatory membrane potentials with filtering
- **Framework:** Brian2 spiking neural network simulator

## References

- Izhikevich, E.M. (2003). Simple model of spiking neurons. IEEE Transactions on Neural Networks, 14(6), 1569-1572.
- Davelaar, E.J. (2018). Mechanisms of Neurofeedback: A Computation-theoretic Approach. Neuroscience, 378, 175-188.

## File Structure

```
/
├── 01_izhikevich_model.py     # Main script
├── README.md                  # This documentation
└── eeg_output/               # Generated output files
    ├── eeg_signal_fcz.txt    # Main EEG data (timestamp format)
    ├── eeg_signal.csv        # EEG data (CSV format)  
    ├── eeg_spectrum.csv      # Spectral analysis
    ├── eeg_summary.csv       # Statistics and band powers
    └── eeg_analysis.png      # Visualization plots
```

Perfect for research, algorithm development, and educational purposes! 🧠✨
