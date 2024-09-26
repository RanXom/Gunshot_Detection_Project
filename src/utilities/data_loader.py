import librosa

def load_audio_file(file_path, sr=44100):
    return librosa.load(file_path, sr=sr)
