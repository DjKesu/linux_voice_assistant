import pyaudio
import wave
import numpy as np
import os
import time

# Constants
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 2
WAVE_OUTPUT_FOLDER = "wake_word_data"
NUM_WAKE_WORD_SAMPLES = 50
NUM_NOT_WAKE_WORD_SAMPLES = 50

def record_audio(filename, duration):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print(f"Recording {filename}...")

    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Done recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def collect_wake_word_samples():
    print(f"\nCollecting {NUM_WAKE_WORD_SAMPLES} wake word samples.")
    print("Please say your wake word clearly each time you're prompted.")
    
    for i in range(NUM_WAKE_WORD_SAMPLES):
        input(f"Press Enter and say the wake word (Sample {i+1}/{NUM_WAKE_WORD_SAMPLES})...")
        record_audio(f"{WAVE_OUTPUT_FOLDER}/wakeword_{i:03d}.wav", RECORD_SECONDS)
        time.sleep(0.5)  # Short pause between recordings

def collect_not_wake_word_samples():
    print(f"\nCollecting {NUM_NOT_WAKE_WORD_SAMPLES} non-wake word samples.")
    print("These should be a mix of silence, background noise, and random words or phrases.")
    
    for i in range(NUM_NOT_WAKE_WORD_SAMPLES):
        input(f"Press Enter for the next non-wake word sample (Sample {i+1}/{NUM_NOT_WAKE_WORD_SAMPLES})...")
        record_audio(f"{WAVE_OUTPUT_FOLDER}/not_wakeword_{i:03d}.wav", RECORD_SECONDS)
        time.sleep(0.5)  # Short pause between recordings

def main():
    os.makedirs(WAVE_OUTPUT_FOLDER, exist_ok=True)
    
    print("Wake Word Data Collection")
    print("=========================")
    print(f"This script will guide you through collecting {NUM_WAKE_WORD_SAMPLES} wake word samples")
    print(f"and {NUM_NOT_WAKE_WORD_SAMPLES} non-wake word samples.")
    print("\nEach recording will be 2 seconds long.")
    print("Please ensure you're in a quiet environment for best results.")
    
    input("\nPress Enter to begin...")
    
    collect_wake_word_samples()
    collect_not_wake_word_samples()
    
    print("\nData collection complete!")
    print(f"All samples have been saved in the '{WAVE_OUTPUT_FOLDER}' directory.")

if __name__ == "__main__":
    main()