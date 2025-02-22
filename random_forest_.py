import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("merged_drowsiness_data.csv")

# Print available columns
#print("Dataset Columns:", df.columns.tolist())

# Feature selection (removing highly correlated or redundant features)
features = ["EAR", "PERCLOS_x", "Blink_x", "PupilDiameter", "HeadPitch", "HeadRoll"]
X = df[features]
y = df["Label"]

# Split data with Stratification for balanced classes
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Apply Cross-Validation to check generalization
rf_model = RandomForestClassifier(
    n_estimators=200,  # More trees = better generalization
    max_depth=10,  # Prevent overfitting
    min_samples_split=10,  # Minimum samples needed to split a node
    min_samples_leaf=5,  # Prevent small leaf nodes
    random_state=42
)

# Train with Cross-Validation
cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5)
print(f"Cross-Validation Accuracy: {np.mean(cv_scores):.2f}")

# Train Final Model
rf_model.fit(X_train, y_train)

# Evaluate Model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

# Save Model
joblib.dump(rf_model, "drowsiness_ml_model.pkl")
print("Trained ML model saved as 'drowsiness_ml_model.pkl'")
