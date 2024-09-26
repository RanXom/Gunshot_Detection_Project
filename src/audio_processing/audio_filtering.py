from scipy.signal import butter, filtfilt
import librosa

# Bandpass filter design: 500 Hz to 3 kHz
def bandpass_filter(data, sr, lowcut=500, highcut=3000, order=4):
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def process_audio_file(file_path):
    y, sr = librosa.load(file_path, sr=44100)
    filtered_y = bandpass_filter(y, sr)
    return filtered_y