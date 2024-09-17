# process_data.py
import os
import numpy as np
import librosa
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Define paths
DATA_DIR = "wake_word_data/"
SAMPLE_RATE = 16000
N_MFCC = 13

# Function to load and extract MFCC features from an audio file
def extract_features(file_path, sr=SAMPLE_RATE, n_mfcc=N_MFCC):
    # Load the audio file
    audio, _ = librosa.load(file_path, sr=sr)
    
    # Extract MFCC features (fix the function call)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Take the mean across the time axis to get a fixed-size input
    mfcc = np.mean(mfcc.T, axis=0)
    
    return mfcc

# Function to load the dataset and extract features for wake word and non-wake word data
def load_data(data_dir):
    X, y = [], []
    
    for filename in os.listdir(data_dir):
        if filename.startswith("wakeword"):
            label = 1  # Wake word
        elif filename.startswith("not_wakeword"):
            label = 0  # Non-wake word
        else:
            continue
        
        file_path = os.path.join(data_dir, filename)
        features = extract_features(file_path)
        X.append(features)
        y.append(label)
    
    return np.array(X), np.array(y)

# Main function to process data and split into training and validation sets
def process_data():
    X, y = load_data(DATA_DIR)
    
    # Split data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # One-hot encode the labels
    y_train_cat = to_categorical(y_train, num_classes=2)
    y_val_cat = to_categorical(y_val, num_classes=2)
    
    # Save processed data
    np.save("X_train.npy", X_train)
    np.save("X_val.npy", X_val)
    np.save("y_train_cat.npy", y_train_cat)
    np.save("y_val_cat.npy", y_val_cat)
    
    print("Data processing complete! Files saved as 'X_train.npy', 'X_val.npy', 'y_train_cat.npy', and 'y_val_cat.npy'.")

if __name__ == "__main__":
    process_data()
