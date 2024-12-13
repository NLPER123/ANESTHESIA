#Loading .mat Files in Python
import scipy.io

# Load .mat file
data = scipy.io.loadmat('split2250_bipolarRerefType1_lineNoiseRemoved_postPuffpreStim.mat')

# Explore the contents of the file
print(data.keys())  # This will show the variable names stored in the file

# Access specific data (assuming data is stored in a variable named 'psg_data')
psg_data = data['psg_data']  # Adjust the key name based on your file

# Example: Plot the first channel (assuming it's a 2D array)
import matplotlib.pyplot as plt

plt.plot(psg_data[0, :])  # Plot the first channel (rows=channels, columns=samples)
plt.title("EEG Data")
plt.xlabel("Samples")
plt.ylabel("Amplitude")
plt.show()
