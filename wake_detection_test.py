import numpy as np
import tensorflow as tf
import librosa

# Constants
RATE = 16000
RECORD_SECONDS = 2

print("Loading model...")
model = tf.keras.models.load_model('wake_word_model.h5')
print("Model loaded successfully")

def preprocess_audio(audio):
    mfccs = librosa.feature.mfcc(y=audio, sr=RATE, n_mfcc=40)
    mfccs = np.pad(mfccs, ((0, 0), (0, 98 - mfccs.shape[1])), mode='constant')
    return mfccs.reshape(1, 40, 98, 1)

def detect_wake_word(audio):
    processed_audio = preprocess_audio(audio)
    prediction = model.predict(processed_audio)
    return prediction[0][0]

def test_wake_word_detection():
    print("Testing wake word detection...")
    
    # Generate a random audio signal
    random_audio = np.random.rand(RATE * RECORD_SECONDS)
    
    # Test with random audio
    result = detect_wake_word(random_audio)
    print(f"Random audio detection score: {result:.4f}")
    
    # Generate a simple sine wave to simulate a potential wake word
    t = np.linspace(0, RECORD_SECONDS, RATE * RECORD_SECONDS, endpoint=False)
    simulated_wake_word = np.sin(2 * np.pi * 1000 * t)
    
    # Test with simulated wake word
    result = detect_wake_word(simulated_wake_word)
    print(f"Simulated wake word detection score: {result:.4f}")
    
    # Test with silence
    silence = np.zeros(RATE * RECORD_SECONDS)
    result = detect_wake_word(silence)
    print(f"Silence detection score: {result:.4f}")

    # Print model summary
    model.summary()

if __name__ == "__main__":
    test_wake_word_detection()