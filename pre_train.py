# build_model.py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the preprocessed training and validation data
X_train = np.load("X_train.npy")
X_val = np.load("X_val.npy")
y_train_cat = np.load("y_train_cat.npy")
y_val_cat = np.load("y_val_cat.npy")

# Reshape the input data to match input shape of the model
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]

# Function to build the CNN + GRU model with Conv1D
def build_model(input_shape):
    model = models.Sequential()
    
    # 1. Conv1D layer to process the input features (MFCC in this case)
    model.add(layers.Conv1D(32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    
    # 2. GRU layer to capture sequential features
    model.add(layers.GRU(128, return_sequences=False, activation='relu'))
    
    # 3. Fully connected layer for classification (wake word or not)
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))  # 2 classes: wake word, non-wake word
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Build and train the model
input_shape = (X_train.shape[1], 1)  # Update to match the shape (sequence length, 1 channel)
model = build_model(input_shape)

# Print the model summary
model.summary()

# Train the model
history = model.fit(X_train, y_train_cat, epochs=10, batch_size=32, validation_data=(X_val, y_val_cat))

# Save the trained model
model.save("wake_word_model.h5")
print("Model training complete and saved as 'wake_word_model.h5'.")
