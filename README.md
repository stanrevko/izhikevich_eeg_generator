# Izhikevich EEG Generator

This project implements a computational model of EEG generation based on a network of Izhikevich spiking neurons. It was developed for the PhD thesis of Stanislav Revko, based on the computational framework described by Davelaar (2018).

## Researcher Profile
- **Name**: Stanislav Revko
- **Institution**: Lesya Ukrainka Volyn National University
- **Status**: PhD student

## Overview

The core of this application is the `IzhikevichEEGGenerator` module, which simulates a cortical column consisting of:
- **800 Excitatory Neurons**: Modeled as Regular Spiking (RS) neurons.
- **200 Inhibitory Neurons**: Modeled as Fast Spiking (FS) neurons.

The neurons are connected in an all-to-all topology with specific synaptic weights and delays. The "EEG" signal is derived from the sum of membrane potentials of all excitatory neurons, processed through a **1-50 Hz bandpass filter** to simulate the field potential recorded by an EEG electrode.

### Theoretical Basis
The model is based on:
> Davelaar, E. J. (2018). **Mechanisms of Neurofeedback: A Computation-theoretic Approach**. *Neuroscience*.

This paper describes how such a network can generate alpha oscillations (8-12 Hz) and how these oscillations can be modulated by thalamic input and neurofeedback training.

## Features

- **Real-time Simulation**: Runs the spiking neural network in real-time.
- **Interactive UI**: Control parameters on the fly.
- **Alpha Drive**: Inject oscillatory drive to entrain alpha frequencies.
- **Reproducibility**: Set random seeds to generate identical signal traces.
- **Visualization**: Live plotting of the time-domain EEG signal and its Power Spectral Density (PSD).

## Quick Start

### Prerequisites
The project requires Python 3 and the following libraries:
- `numpy`
- `matplotlib`
- `scipy`
- `tkinter` (usually included with Python)

### Installation

1.  Clone the repository or navigate to the project directory.
2.  Create a virtual environment:
    ```bash
    python3 -m venv venv
    ```
3.  Activate the virtual environment:
    - On macOS/Linux:
      ```bash
      source venv/bin/activate
      ```
    - On Windows:
      ```bash
      venv\Scripts\activate
      ```
4.  Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

To launch the interactive UI:

```bash
python eeg_ui.py
```

## Usage Guide

The User Interface allows you to manipulate the neural network in real-time:

### Controls
- **Start/Stop**: Toggles the simulation.
- **Reset**: Resets the network state and clears the plots.
- **Random Seed**: Enter an integer (e.g., `42`) and click **Apply Seed** to restart with a specific seed. Leave empty for random behavior.

### Parameters
- **Input (Exc/Inh)**: Controls the mean background drive to the neurons. Higher input generally leads to higher frequency oscillations.
- **Noise Level**: Adds measurement noise to the output signal.
- **Alpha Drive Amplitude**: Injects a sinusoidal current at the alpha frequency. Increase this to artificially enhance alpha power (similar to the "blue plot" in Davelaar's analysis).
- **Alpha Drive Freq**: Sets the frequency of the alpha drive (8-12 Hz).

## Project Structure

- `eeg_generator.py`: The core simulation engine implementing the Izhikevich network.
- `eeg_ui.py`: The graphical user interface for real-time interaction.
