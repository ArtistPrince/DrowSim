import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split

# Paths
data_dir = "extracted_frames"  # Main folder containing Face_cropped1, Face_cropped2, etc.
img_size = 64  # Resize images to 64x64

# Initialize empty lists for images and labels
X, y = [], []
labels = {"alert": 0, "drowsy": 1}

# Iterate over all Face_cropped directories
for i in range(1, 28):  # Assuming directories are named Face_cropped1 to Face_cropped27
    data_dir1 = os.path.join(data_dir, f"Face_cropped{i}")

    for label, class_id in labels.items():
        class_path = os.path.join(data_dir1, label)

        # Check if the label folder exists
        if not os.path.exists(class_path):
            print(f"Skipping missing folder: {class_path}")
            continue

        # Load images
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (img_size, img_size)) / 255.0  # Normalize
            X.append(img)
            y.append(class_id)

# Convert to numpy arrays
X = np.array(X).reshape(-1, img_size, img_size, 1)  # Reshape for CNN input
y = np.array(y)

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Define CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(img_size, img_size, 1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')  # Output layer for binary classification
])

# Compile Model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val))

# Save Model
model.save("drowsiness_cnn_model.h5")
print("Model saved as 'drowsiness_cnn_model.h5'")
