import numpy as np
from scipy.signal import butter, filtfilt, correlate
import librosa

# Bandpass filter: 500 Hz to 3 kHz
def bandpass_filter(data, sr, lowcut=500, highcut=3000, order=4):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Cross-correlation to compute Time Difference of Arrival (TDoA)
def cross_correlation(signal1, signal2):
    corr = correlate(signal1, signal2, mode='full')
    lag = np.argmax(corr) - len(signal1)
    return lag

# Calculate TDoA using cross-correlation
def calculate_tdoa(file1, file2):
    # Load the audio files
    y1, sr1 = librosa.load(file1, sr=44100)
    y2, sr2 = librosa.load(file2, sr=44100)

    # Apply bandpass filter to both signals
    y1_filtered = bandpass_filter(y1, sr1)
    y2_filtered = bandpass_filter(y2, sr2)

    # Calculate TDoA between the two signals
    lag = cross_correlation(y1_filtered, y2_filtered)
    time_delay = lag / sr1
    return time_delay

# Example usage: Calculate TDoA between two test files
file1 = '/datasets/test/test_gunshot_1.wav'
file2 = '/datasets/test/test_gunshot_2.wav'
tdoa = calculate_tdoa(file1, file2)
print(f"Time difference between {file1} and {file2}: {tdoa} seconds")
