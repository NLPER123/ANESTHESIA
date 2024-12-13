#Preprocessing and PSD Computation
pip install mne pycatch22 sklearn
import numpy as np
import pandas as pd
import mne
from mne.io import read_raw_edf
from mne.time_frequency import psd_welch
import matplotlib.pyplot as plt

# Load EEG Data (Assume you have the EDF file path)
def load_eeg_data(edf_file):
    raw_data = read_raw_edf(edf_file, preload=True)
    return raw_data

# Preprocess EEG Data (e.g., filtering)
def preprocess_eeg_data(raw_data, l_freq=0.5, h_freq=40):
    raw_data.filter(l_freq=l_freq, h_freq=h_freq)
    return raw_data

# Compute Power Spectral Density (PSD)
def compute_psd(raw_data, epochs, fmin=0.5, fmax=40):
    psd, freqs = psd_welch(raw_data, fmin=fmin, fmax=fmax, n_fft=2048)
    return psd, freqs

# Plot PSD for a given epoch
def plot_psd(psd, freqs, title='Power Spectral Density'):
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, np.mean(psd, axis=0))
    plt.title(title)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (uV^2/Hz)')
    plt.show()

# Example for loading and processing data
edf_file = 'path_to_your_eeg_data.edf'  # Replace with your actual file path
raw_data = load_eeg_data(edf_file)
raw_data = preprocess_eeg_data(raw_data)
psd, freqs = compute_psd(raw_data, epochs=None)  # epochs parameter can be adjusted if needed
plot_psd(psd, freqs)
