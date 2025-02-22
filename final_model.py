import cv2
import numpy as np
import tensorflow as tf
import dlib
import joblib
import pygame
import random
import time
from scipy.spatial import distance
from deepface import DeepFace
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.saving import register_keras_serializable

@register_keras_serializable()
def mse(y_true, y_pred):
    return MeanSquaredError()(y_true, y_pred)

# Load LSTM model with custom loss function
lstm_model = tf.keras.models.load_model("lstm_model.h5", custom_objects={"mse": mse})

print("✅ LSTM model loaded successfully!")
# Load trained models
cnn_model = tf.keras.models.load_model("drowsiness_cnn_model.h5")
rf_model = joblib.load("drowsiness_ml_model.pkl")
#lstm_model = tf.keras.models.load_model("lstm_model.h5")
nlp_model = tf.keras.models.load_model("NLP_model.h5")

# Load scaler for feature normalization
scaler = joblib.load("scaler.pkl")

# Initialize Dlib's face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize Pygame for sound alerts
pygame.mixer.init()
alarm_sound = "alarm.wav"

# EAR & MAR thresholds
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.5
drowsy_frames = 0
yawning_frames = 0

# Physiological Data Simulation (Real Sensors Needed for Deployment)
EXPECTED_FEATURES = 21

# Define Eye & Mouth Landmarks
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
MOUTH = list(range(60, 68))

def eye_aspect_ratio(eye):
    """Compute Eye Aspect Ratio (EAR)."""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    """Compute Mouth Aspect Ratio (MAR) for yawning detection."""
    if len(mouth) < 10:
        return 0
    A = distance.euclidean(mouth[2], mouth[10])
    B = distance.euclidean(mouth[4], mouth[8])
    C = distance.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

def generate_random_data():
    """Simulate physiological sensor readings (replace with real sensor input)."""
    input_data = np.random.normal(loc=0, scale=1, size=(1, EXPECTED_FEATURES))
    input_data = np.reshape(input_data, (1, 1, EXPECTED_FEATURES))  # Reshape for LSTM
    return input_data

def get_real_time_inputs():
    """Simulate real-time Heart Rate (HR) & Steering Behavior."""
    hr = random.randint(50, 110)
    steering_variability = random.uniform(0.1, 0.9)
    return np.array([[hr, steering_variability]])

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        x, y, w, h = max(0, x), max(0, y), max(1, w), max(1, h)

        if (y + h) >= gray.shape[0] or (x + w) >= gray.shape[1]:
            continue

        # Detect facial landmarks
        landmarks = predictor(gray, face)
        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE]
        mouth = [(landmarks.part(n).x, landmarks.part(n).y) for n in MOUTH]

        # Compute EAR & MAR
        avg_EAR = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # CNN Prediction
        face_img = gray[y:y+h, x:x+w]
        if face_img.shape[0] > 0 and face_img.shape[1] > 0:
            face_img_resized = cv2.resize(face_img, (64, 64)) / 255.0
            face_img_resized = face_img_resized.reshape(1, 64, 64, 1)
            cnn_prediction = cnn_model.predict(face_img_resized)
            cnn_label = np.argmax(cnn_prediction)
        else:
            cnn_label = 0

        # Random Forest Prediction
        ml_features = np.array([[avg_EAR, 50, 2, 3.5, 0.0, 0.0]])
        ml_prediction = rf_model.predict(ml_features)[0]

        # LSTM Physiological Data Prediction
        input_data = generate_random_data()
        lstm_prediction = lstm_model.predict(input_data)[0][0]

        # NLP Model Prediction
        X_real_time = get_real_time_inputs()
        X_real_time = scaler.transform(X_real_time)
        nlp_prediction = nlp_model.predict(X_real_time)[0][0]

        # Final Decision: Drowsy if any model predicts drowsiness
        if cnn_label == 1 or ml_prediction == 1 or avg_EAR < EAR_THRESHOLD or lstm_prediction > 0.5 or nlp_prediction > 0.5:
            final_label = "DROWSY"
            color = (0, 0, 255)
        else:
            final_label = "ALERT"
            color = (0, 255, 0)

        # Yawning Detection
        yawning_label = "Yawning" if mar > MAR_THRESHOLD else "No Yawn"

        # Emotion Detection
        try:
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = results[0]['dominant_emotion']
        except:
            emotion = "Unknown"

        # Display Labels
        cv2.putText(frame, f"Status: {final_label}", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Yawning: {yawning_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, f"Emotion: {emotion}", (x, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Play alarm if drowsy
        if final_label == "DROWSY":
            pygame.mixer.music.load(alarm_sound)
            pygame.mixer.music.play()
        else:
            pygame.mixer.music.stop()

    cv2.imshow("Drowsiness, Yawning & Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
