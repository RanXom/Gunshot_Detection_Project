import sys
sys.path.append('scripts/')

import librosa
import numpy as np
import os

def extract_features(file_path, n_mfcc=40):
    """Extracts MFCC features from an audio file."""
    audio, sample_rate = librosa.load(file_path, duration=3)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)  # Averaging across time steps

def process_data(data_dir):
    """Processes all audio files in a directory and extracts features."""
    features = []
    labels = []
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                label = 1 if 'gunshot' in file else 0  # Assuming gunshot files contain 'gunshot' in name
                file_path = os.path.join(root, file)
                mfcc_features = extract_features(file_path)
                features.append(mfcc_features)
                labels.append(label)
    
    return np.array(features), np.array(labels)

# Example usage
train_features, train_labels = process_data('data/train/')
test_features, test_labels = process_data('data/test/')

# Save extracted features
np.save('features/train_features.npy', train_features)
np.save('features/train_labels.npy', train_labels)
np.save('features/test_features.npy', test_features)
np.save('features/test_labels.npy', test_labels)