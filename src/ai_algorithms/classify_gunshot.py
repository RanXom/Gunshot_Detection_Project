import librosa
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained gunshot classifier model
model = load_model('/models/gunshot_classifier.h5')

# Function to classify audio using the trained model
def classify_gunshot(audio_path):
    # Load the audio file
    y, sr = librosa.load(audio_path, sr=44100)

    # Extract MFCC features for classification
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=13)
    mfcc_scaled = np.mean(mfcc.T, axis=0)

    # Predict using the pre-trained model
    prediction = model.predict(np.expand_dims(mfcc_scaled, axis=0))

    # Return the class label
    return "gunshot" if prediction > 0.5 else "ambient_noise"

# Test the function on a file
audio_file = '/datasets/test/test_gunshot_1.wav'
result = classify_gunshot(audio_file)
print(f"Detected sound in {audio_file}: {result}")
