# Gunshot Detection System

## Overview
This project aims to develop an AI-based gunshot detection system that processes audio inputs, detects gunshot sounds, and determines the origin of the sound. The system utilizes **Machine Learning** techniques with **MFCC (Mel Frequency Cepstral Coefficients)** for feature extraction and a neural network for classification.

## Features
- **Audio Preprocessing**: Converts raw audio files into MFCC features.
- **Model Training**: A feedforward neural network classifies the audio signals.
- **Gunshot Detection**: Identifies gunshot sounds from other environmental noises.
- **Time Difference of Arrival (TDoA)**: Detects the location of the gunshot using multiple microphones.
- **Real-time Inference**: Capable of processing audio in real-time for gunshot detection.

## Folder Structure
```bash
Gunshot_Detection_Project/
│
├── data/                     # Dataset: Audio files for training and testing
│   ├── train/
│   └── test/
├── features/                 # Extracted features from audio files
├── models/                   # Saved models after training
├── notebooks/                # Jupyter notebooks for development
├── scripts/                  # Python scripts for preprocessing and training
├── results/                  # Logs and evaluation results
└── requirements.txt          # Project dependencies
```

## Installation

1. **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/gunshot-detection.git
    ```

2. **Navigate into the project directory**:
    ```bash
    cd gunshot-detection
    ```

3. **Install required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. **Data Preprocessing**
   Extract features from the audio files:
   ```bash
   python scripts/data_preprocessing.py
   ```

### 2. **Model Training**
   Train the neural network model:
   ```bash
   python scripts/model_training.py
   ```

### 3. **Inference**
   Use the trained model to detect gunshot sounds from new audio inputs.

## Algorithms Used
- **MFCC** (Mel Frequency Cepstral Coefficients) – For feature extraction from audio files.
- **Feedforward Neural Network** – For classifying audio as either gunshot or non-gunshot.
- **Adam Optimizer** – For optimizing the model during training.
- **Binary Cross-Entropy Loss** – For calculating loss during training.
- **TDoA (Time Difference of Arrival)** – For detecting the direction of gunshots using multiple microphones.

## Technologies Used
- **Python**
- **Librosa** for audio processing
- **Keras/TensorFlow** for model building
- **Numpy** for handling numerical data
- **Matplotlib** for visualization

## Results
You can find the accuracy and evaluation logs in the `results/` folder.

## License
This project is licensed under the MIT License.

---
