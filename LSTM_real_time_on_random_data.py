import numpy as np
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Load trained LSTM model with custom loss
custom_objects = {"mse": MeanSquaredError()}
model = load_model("lstm_model.h5", custom_objects=custom_objects)
print("Model input shape:", model.input_shape)  # Debugging: Print expected input shape

# Define feature columns based on training data
feature_columns = [
    'BIOPAC/ECG', 'BIOPAC/Poz', 'BIOPAC/Fz', 'BIOPAC/Cz', 'BIOPAC/C3', 'BIOPAC/C4',
    'BIOPAC/F3', 'BIOPAC/F4', 'BIOPAC/P3', 'BIOPAC/P4', 'BIOPAC/HR',
    'Empatica/Acceleration(X)', 'Empatica/Acceleration(Y)', 'Empatica/Acceleration(Z)',
    'Empatica/BVP', 'Empatica/EDA', 'Empatica/Temperature', 'Empatica/SCL', 'Empatica/SCR'
]

EXPECTED_FEATURES = 21  # Your model expects 21 features

# Function to simulate real-time physiological data
def generate_random_data():
    """Generates a random physiological sample."""
    input_data = np.random.normal(loc=0, scale=1, size=(1, len(feature_columns)))  # (1, 19)

    # Fix shape mismatch: Add 2 extra features if missing
    if input_data.shape[1] < EXPECTED_FEATURES:
        input_data = np.pad(input_data, ((0, 0), (0, EXPECTED_FEATURES - input_data.shape[1])), mode='constant')

    # Reshape for LSTM (batch_size=1, time_steps=1, features=21)
    input_data = np.reshape(input_data, (1, 1, EXPECTED_FEATURES))
    
    return input_data

# Real-time inference loop
print("Starting real-time LSTM inference... Press Ctrl+C to stop.")
try:
    while True:
        # Generate random physiological data
        input_data = generate_random_data()

        # Debugging: Print input shape
        print("Input data shape:", input_data.shape)  # Should be (1, 1, 21)

        # Predict using the LSTM model
        prediction = model.predict(input_data)[0][0]

        # Display result
        print(f"Predicted HR (Empatica): {prediction:.2f}")

        # Simulate real-time delay (adjust as needed, e.g., 1Hz sampling rate)
        time.sleep(2)  
except KeyboardInterrupt:
    print("Real-time prediction stopped.")
