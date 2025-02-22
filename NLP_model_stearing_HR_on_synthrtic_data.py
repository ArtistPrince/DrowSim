import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


# Generate synthetic data (100000 samples)
num_samples = 100000
heart_rates = np.random.randint(50, 110, num_samples)  # Heart rate (BPM)
steering_variability = np.random.uniform(0.1, 0.9, num_samples)  # Steering deviation

# Label as "Drowsy" if HR < 55 or Steering Variability > 0.7
drowsiness_labels = np.array([1 if (hr < 55 or steer > 0.7) else 0 for hr, steer in zip(heart_rates, steering_variability)])

# Create a DataFrame
df = pd.DataFrame({"HeartRate": heart_rates, "Steering": steering_variability, "Drowsiness": drowsiness_labels})

# Save to CSV
df.to_csv("synthetic_drowsiness_data.csv", index=False)

print("Synthetic data created & saved.")


# Load the synthetic dataset
df = pd.read_csv("synthetic_drowsiness_data.csv")
X = df[["HeartRate", "Steering"]].values  # Features
y = df["Drowsiness"].values  # Labels

# Normalize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the scaler for real-time use
joblib.dump(scaler, "scaler.pkl")

# Split into train (80%) & test (20%)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the Neural Network model
model = Sequential([
    Dense(64, activation="relu", input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")  # Binary Classification (Drowsy vs Alert)
])

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Save the trained model
model.save("NLP_model.h5")
print("Model training complete & saved as 'NLP.h5'.")

# Evaluate model on test data
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Drowsiness Model Accuracy: {accuracy:.2f}")
