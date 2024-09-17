# test_live.py
import numpy as np
import tensorflow as tf
import sounddevice as sd
import librosa
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("wake_word_model.h5")

# Define the feature extraction parameters
N_MFCC = 13
SAMPLING_RATE = 16000  # Adjust if necessary

def extract_features(audio):
    # Extract MFCC features from the audio
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLING_RATE, n_mfcc=N_MFCC)
    return mfcc.T  # Transpose to fit the expected input shape

def detect_wake_word():
    print("Listening for wake word...")
    while True:
        # Record audio for 1 second
        audio = sd.rec(int(SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1, dtype='float32')
        sd.wait()
        
        # Extract features from the recorded audio
        features = extract_features(audio.flatten())
        
        # Ensure the shape is compatible with Conv1D
        if features.shape[0] < 32:  # If the sequence length is less than required
            features = np.pad(features, ((0, 32 - features.shape[0]), (0, 0)), mode='constant')  # Pad if needed
        
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        
        # Verify that the shape matches the input expected by Conv1D
        features = np.expand_dims(features, axis=-1)  # Add channel dimension
        
        # Predict using the trained model
        prediction = model.predict(features)
        predicted_class = np.argmax(prediction, axis=1)[0]
        
        # Output result
        if predicted_class == 0:
            print("Wake word detected!")
        else:
            print("No wake word detected.")

if __name__ == "__main__":
    detect_wake_word()
