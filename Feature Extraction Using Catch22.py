import numpy as np
from mne.io import read_raw_edf
from catch22 import catch22_all

# Function to extract Catch22 features
def extract_catch22_features(eeg_data, channel_idx=0):
    """
    Extract Catch22 features from EEG data.
    :param eeg_data: MNE Raw object containing EEG data.
    :param channel_idx: The index of the channel to use for feature extraction.
    :return: A NumPy array of Catch22 features.
    """
    # Get data from the specified channel
    data, times = eeg_data[channel_idx, :]  # shape: (1, n_samples)
    data = data.flatten()  # Flatten to 1D array

    # Extract Catch22 features
    features = catch22_all(data)['values']
    return features

# Example usage
edf_file_path = 'your_file.edf'  # Replace with the path to your .edf file
raw_data = read_raw_edf(edf_file_path, preload=True)  # Load .edf file

# Extract features for a specific channel (e.g., the first channel)
channel_index = 0  # Adjust this for the desired channel
catch22_features = extract_catch22_features(raw_data, channel_idx=channel_index)

print("Catch22 Features for Channel", channel_index, ":", catch22_features)

