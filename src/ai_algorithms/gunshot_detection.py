# AI-based gunshot detection algorithm
import librosa
from keras.models import load_model
import numpy as np

model = load_model('/models/gunshot_classifier.h5')

def detect_gunshot(audio_path):
    y, sr = librosa.load(audio_path, sr=44100)
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=13)
    mfcc_scaled = np.mean(mfcc.T, axis=0)
    prediction = model.predict(np.expand_dims(mfcc_scaled, axis=0))
    return "gunshot" if prediction > 0.5 else "ambient_noise"

result = detect_gunshot('/datasets/raw_audio/mic1_gunshot.wav')
print(f"Detected sound: {result}")
