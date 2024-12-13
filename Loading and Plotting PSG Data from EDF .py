#Loading and Plotting PSG Data from EDF
import pyedflib
import numpy as np
import matplotlib.pyplot as plt

def load_psg_data(edf_file):
    # Open the EDF file
    edf = pyedflib.EdfReader(edf_file)
    
    # Get the number of channels
    num_channels = edf.signals_in_file
    
    # Read the signals (EEG, EOG, EMG, etc.)
    psg_data = []
    for i in range(num_channels):
        signal = edf.readSignal(i)
        psg_data.append(signal)
    
    # Return signals as a NumPy array (shape: channels x samples)
    return np.array(psg_data), edf.getSampleFrequency(0)  # assuming same frequency for all channels

# Load PSG data from an EDF file
psg_data, sampling_rate = load_psg_data('your_psg_file.edf')

# Plot the first channel (usually EEG)
plt.figure(figsize=(10, 4))
plt.plot(np.arange(psg_data.shape[1]) / sampling_rate, psg_data[0])
plt.title("EEG Data - Channel 1")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.show()
