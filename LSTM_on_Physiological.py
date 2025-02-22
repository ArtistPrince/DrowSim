import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Load EEG_ECG and PPG_EDA datasets
EEG_ECG_PATH = "merged_EEG_ECG.csv"
PPG_EDA_PATH = "merged_PPG_EDA.csv"

# Read datasets
eeg_ecg_df = pd.read_csv(EEG_ECG_PATH)
ppg_eda_df = pd.read_csv(PPG_EDA_PATH)

# Ensure 'time' column is in datetime format
eeg_ecg_df['time'] = pd.to_datetime(eeg_ecg_df['time'], errors='coerce')
ppg_eda_df['time'] = pd.to_datetime(ppg_eda_df['time'], errors='coerce')

# Drop rows where 'time' is NaN
eeg_ecg_df = eeg_ecg_df.dropna(subset=['time'])
ppg_eda_df = ppg_eda_df.dropna(subset=['time'])

# Merge datasets on 'time' column using nearest match
merged_df = pd.merge_asof(
    eeg_ecg_df.sort_values('time'),
    ppg_eda_df.sort_values('time'),
    on='time',
    direction='nearest'
)

# Drop 'time' column as it is not needed for training
merged_df.drop(columns=['time'], inplace=True)

# Handle remaining missing values
merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')

# Split features and target (assuming Empatica/HR as target)
X = merged_df.drop(columns=['Empatica/HR'])
y = merged_df['Empatica/HR']

# Normalize features
X = (X - X.mean()) / X.std()

# Reshape for LSTM input (samples, timesteps, features)
X = np.expand_dims(X.values, axis=1)  # Assuming a single timestep

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dropout(0.2),
    Dense(1, activation='linear')
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Save model
model.save("lstm_model.h5")

# Evaluate model
loss, mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}, Test MAE: {mae}")
