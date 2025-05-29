# EEG Signal Architecture: Component Breakdown

This document provides a detailed schematic breakdown of how the EEG signal is generated from multiple components in the Izhikevich neural network model.

## 🧠 **Signal Generation Architecture**

### **Core Question: What Creates the Final EEG Signal?**

The resulting EEG signal is **NOT just** from Izhikevich neurons - it's a combination of multiple signal sources that are mathematically combined.

---

## 📊 **Signal Component Breakdown**

### **1. 🔋 Foundation: Izhikevich Neural Network**
```python
# 1000 neurons with membrane potentials
neurons.v  # Membrane potentials (-65 to +30 mV)
↓
membrane_sum = np.sum(neurons.v[:N_exc])  # Sum of excitatory neurons
```
**Purpose:** Provides the biological foundation and natural network oscillations

### **2. 🎛️ External Drive (Baseline Excitation)**
```python
neurons.I_ext = 8 + 3 * np.random.randn(N_total)  # Tonic excitation
```
**Effect:** Sets baseline network activity level

### **3. 🎵 Rhythmic Generators (ADDITIONAL Wave Sources)**

#### **Alpha Wave Generator (10 Hz):**
```python
@network_operation(dt=1*ms)
def alpha_modulation():
    t = float(defaultclock.t / second)
    alpha_freq = 10  # Hz
    modulation = 3 * np.sin(2 * np.pi * alpha_freq * t)  # 🎵 PURE SINE WAVE!
    neurons.I_ext[:N_exc] += modulation  # Added to neurons
```

#### **Theta Wave Generator (6 Hz):**
```python
@network_operation(dt=1*ms)
def theta_modulation():
    t = float(defaultclock.t / second)
    theta_freq = 6  # Hz
    modulation = 1.5 * np.sin(2 * np.pi * theta_freq * t)  # 🎵 PURE SINE WAVE!
    neurons.I_ext[:N_exc] += modulation  # Added to neurons
```

**Important:** These are **separate sine wave generators** that modulate the neural network!

### **4. 🔀 Dynamic Noise**
```python
@network_operation(dt=10*ms)
def update_noise():
    neurons.I_noise = 2.0 * np.random.randn(len(neurons))  # Random noise
```

### **5. 🔗 Synaptic Connections**
```python
# Excitatory synapses
syn_exc: I_syn += w  # Adds current from other neurons

# Inhibitory synapses  
syn_inh: I_syn -= w  # Subtracts current (inhibition)
```

### **6. 📡 Signal Processing**
```python
# Bandpass filter (0.5-100 Hz)
filtered_signal = apply_filters(raw_signal)

# Notch filter (50 Hz - removes power line noise)
notched_signal = remove_power_line_noise(filtered_signal)
```

### **7. 📏 Amplitude Scaling**
```python
# Convert mV → μV
eeg_signal = (membrane_sum / N_exc) / 500000.0 * 1e6
```

---

## 🏗️ **SCHEMATIC DIAGRAM OF SIGNAL GENERATION**

```
┌─────────────────────────────────────────────────────────────┐
│                    SIGNAL SOURCES                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  🧠 IZHIKEVICH NEURONS (1000 units)                        │
│     ├── 800 excitatory (v: -65 to +30 mV)                 │
│     └── 200 inhibitory (v: -65 to +30 mV)                 │
│                      ⬇                                     │
│  🔗 SYNAPTIC CONNECTIONS                                    │
│     ├── Excitatory: E→E, E→I (weights: ~0.8)              │
│     └── Inhibitory: I→E, I→I (weights: ~1.2)              │
│                      ⬇                                     │
│  🎛️ EXTERNAL CURRENTS                                      │
│     ├── I_ext: 8±3 μA (baseline drive)                    │
│     ├── I_noise: 2.0 μA (random noise)                    │
│     └── I_syn: variable (from synapses)                    │
│                      ⬇                                     │
│  🎵 RHYTHMIC GENERATORS (Additional!)                       │
│     ├── Alpha: 3×sin(2π×10×t) → added to I_ext            │
│     ├── Theta: 1.5×sin(2π×6×t) → added to I_ext           │
│     └── Periodic noise updates (every 10 ms)               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ⬇
┌─────────────────────────────────────────────────────────────┐
│                   SIGNAL PROCESSING                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  📊 POTENTIAL SUMMATION                                     │
│     membrane_sum = Σ(v_excitatory_neurons)                 │
│                      ⬇                                     │
│  📏 AMPLITUDE SCALING                                       │
│     eeg_raw = (membrane_sum / 800) / 500000                 │
│                      ⬇                                     │
│  🔄 DIGITAL FILTERING                                       │
│     ├── Bandpass: 0.5-100 Hz                               │
│     ├── Notch: 50 Hz (power line removal)                  │
│     └── Realistic noise addition                            │
│                      ⬇                                     │
│  📈 FINAL EEG SIGNAL                                        │
│     50-100 μV, dominant frequency ~10 Hz                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 **KEY WAVE SOURCES**

### **1. 🧠 Natural Network Oscillations:**
- **Source:** E/I neuron interactions through synapses
- **Frequency:** 5-15 Hz (natural network frequency)
- **Amplitude:** Depends on synaptic weights

### **2. 🎵 Artificial Generators (ADDITIONAL!):**
```python
# These are SEPARATE sinusoidal generators!
Alpha Generator: 3 × sin(2π × 10 × t)    # 10 Hz, amplitude 3
Theta Generator: 1.5 × sin(2π × 6 × t)   # 6 Hz, amplitude 1.5
```
**These generators are ADDED to the baseline neuron current!**

### **3. 🔀 Noise Components:**
- **Gaussian noise:** `neurons.I_noise`
- **Periodic noise updates:** every 10 ms
- **Filter noise:** added after filtering

---

## ⚡ **MATHEMATICAL FORMULA OF RESULTING SIGNAL**

```python
# Complete EEG generation formula:
EEG(t) = SCALE × FILTER × [
    Σ(V_neurons(t)) +           # Foundation: membrane potentials
    ALPHA_GEN(t) +              # 3×sin(2π×10×t) 
    THETA_GEN(t) +              # 1.5×sin(2π×6×t)
    NOISE(t)                    # Random noise
]

# Where:
V_neurons(t) = f(I_ext, I_syn, I_noise, Izhikevich_dynamics)
SCALE = 1/500000 × 1e6          # Convert mV → μV  
FILTER = Bandpass(0.5-100Hz) + Notch(50Hz)
```

---

## 🎛️ **CONTROLLABLE COMPONENTS**

### **Direct Wave Generators:**
```python
alpha_freq = 10        # Alpha generator frequency
alpha_amplitude = 3    # Alpha generator strength
theta_freq = 6         # Theta generator frequency  
theta_amplitude = 1.5  # Theta generator strength
```

### **Indirect (Through Network):**
```python
neurons.I_ext          # Baseline activity level
syn_exc.w             # Excitatory synaptic weights
syn_inh.w             # Inhibitory synaptic weights
neurons.a, b, c, d    # Izhikevich neuron parameters
```

---

## 🔍 **Signal Flow Analysis**

### **Step 1: Neural Dynamics**
```python
# Each neuron follows Izhikevich equations:
dv/dt = 0.04*v² + 5*v + 140 - u + I_total
du/dt = a*(b*v - u)

# Where I_total includes:
I_total = I_ext + I_syn + I_noise + I_modulation
```

### **Step 2: Modulation Addition**
```python
# External sine wave generators modulate the network:
I_modulation = alpha_amplitude * sin(2π * alpha_freq * t) + 
               theta_amplitude * sin(2π * theta_freq * t)
```

### **Step 3: Signal Extraction**
```python
# Sum excitatory membrane potentials:
raw_eeg = Σ(v_excitatory[i] for i in range(800))
```

### **Step 4: Processing Pipeline**
```python
# Apply realistic EEG processing:
processed_eeg = filter(scale(normalize(raw_eeg)))
```

---

## 📈 **Frequency Content Origins**

| **Frequency Band** | **Primary Source** | **Secondary Sources** |
|-------------------|-------------------|---------------------|
| **Delta (0.5-4 Hz)** | Slow network dynamics | Low-frequency noise |
| **Theta (4-8 Hz)** | **Theta generator** + network | Synaptic dynamics |
| **Alpha (8-13 Hz)** | **Alpha generator** + network | Natural oscillations |
| **Beta (13-30 Hz)** | Network interactions | Higher harmonics |
| **Gamma (30-100 Hz)** | Fast synaptic dynamics | Noise, artifacts |

---

## 🔧 **Modification Examples**

### **Pure Network Oscillations (Remove Artificial Generators):**
```python
# Comment out the generators:
# @network_operation(dt=1*ms)
# def alpha_modulation():
#     ...

# @network_operation(dt=1*ms)  
# def theta_modulation():
#     ...
```

### **Add Custom Frequency Generator:**
```python
@network_operation(dt=1*ms)
def beta_modulation():
    t = float(defaultclock.t / second)
    beta_freq = 20  # Hz
    modulation = 2 * np.sin(2 * np.pi * beta_freq * t)
    neurons.I_ext[:N_exc] += modulation
```

### **Variable Frequency Generator:**
```python
@network_operation(dt=1*ms)
def variable_alpha():
    t = float(defaultclock.t / second)
    # Frequency sweeps from 8 to 12 Hz
    freq = 10 + 2 * np.sin(2 * np.pi * 0.1 * t)
    modulation = 3 * np.sin(2 * np.pi * freq * t)
    neurons.I_ext[:N_exc] += modulation
```

---

## ✅ **CONCLUSION**

**YES, there are ADDITIONAL WAVE GENERATORS beyond Izhikevich neurons!**

1. **🧠 Foundation:** Izhikevich neural network (natural oscillations)
2. **🎵 Additional generators:** 
   - Alpha sine wave (10 Hz)
   - Theta sine wave (6 Hz)  
   - Periodic noise updates
3. **📊 Processing:** Filtering + scaling

**Resulting signal = Biological network + Artificial rhythms + Noise + Processing** 🎯

---

## 🔗 **Related Files**
- `01_izhikevich_model.py` - Main implementation
- `README.md` - General documentation and parameter guide
- `EEG_Signal_Architecture.md` - This document (signal architecture)

The EEG signal is a sophisticated combination of biological neural dynamics enhanced with artificial rhythm generators to produce realistic brain-like oscillations.
