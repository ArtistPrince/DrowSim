import numpy as np
import joblib
import pygame
import time
import tensorflow as tf
import random

# Load trained model
model = tf.keras.models.load_model("NLP_model.h5")

# Load the scaler
scaler = joblib.load("scaler.pkl")

# Initialize Pygame for sound alerts
pygame.mixer.init()
alarm_sound = "alarm.wav"  # Ensure you have an alert sound file

# Function to simulate real-time Heart Rate (HR) & Steering Behavior
def get_real_time_inputs():
    hr = random.randint(50, 110)  # Simulated heart rate
    steering_variability = random.uniform(0.1, 0.9)  # Simulated steering deviations
    return np.array([[hr, steering_variability]])

while True:
    # Get simulated real-time inputs
    X_real_time = get_real_time_inputs()

    # Normalize inputs
    X_real_time = scaler.transform(X_real_time)

    # Predict drowsiness (0 = Alert, 1 = Drowsy)
    prediction = model.predict(X_real_time)
    drowsiness_label = "DROWSY" if prediction[0][0] > 0.5 else "ALERT"

    # Print real-time values
    print(f"Heart Rate: {X_real_time[0][0]:.1f} BPM | Steering: {X_real_time[0][1]:.2f} | Status: {drowsiness_label}")

    # Play alarm if drowsy
    if drowsiness_label == "DROWSY":
        pygame.mixer.music.load(alarm_sound)
        pygame.mixer.music.play()
    else:
        pygame.mixer.music.stop()

    time.sleep(1)  # Simulate real-time processing
