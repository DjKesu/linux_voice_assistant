import pyaudio
import numpy as np
import librosa
import tensorflow as tf
from ctypes import *

# Constants
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 2

# Load the trained model
model = tf.keras.models.load_model('wake_word_model.h5')

def preprocess_audio(audio):
    mfccs = librosa.feature.mfcc(y=audio, sr=RATE, n_mfcc=40)
    mfccs = np.pad(mfccs, ((0, 0), (0, 98 - mfccs.shape[1])), mode='constant')
    return mfccs.reshape(1, 40, 98, 1)

def detect_wake_word(audio):
    processed_audio = preprocess_audio(audio)
    prediction = model.predict(processed_audio)
    return prediction[0][0] > 0.5

def listen_for_wake_word():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Listening for wake word...")

    while True:
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        audio = np.frombuffer(b''.join(frames), dtype=np.float32)
        
        if detect_wake_word(audio):
            stream.stop_stream()
            stream.close()
            p.terminate()
            return 1
    
    return 0

# Create a C-compatible function
@CFUNCTYPE(c_int)
def wake_word_detected():
    result = listen_for_wake_word()
    return result

# This will be called when the library is loaded
def init_wake_word_detection():
    # Any initialization code can go here
    pass