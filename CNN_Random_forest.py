import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import dlib
from scipy.spatial import distance
import pygame
import joblib
from deepface import DeepFace

# Load trained models
cnn_model = load_model("drowsiness_cnn_model.h5")
rf_model = joblib.load("drowsiness_ml_model.pkl")
feature_count = rf_model.n_features_in_
print(f"The model expects {feature_count} features.")

# Initialize Dlib's face & landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# Initialize Pygame for sound alerts
pygame.mixer.init()
alarm_sound = "alarm.wav"  # Ensure you have an alert sound file

# Define EAR threshold for drowsiness detection
EAR_THRESHOLD = 0.25
drowsy_frames = 0
perclos_x = 50.0  # Placeholder value, replace with real-time computation
blink_x = 2  # Placeholder value, replace with real-time computation
pupil_diameter = 3.5  # Placeholder value, replace with real-time computation
head_pitch = 0.0  # Replace with actual computed value if available
head_roll = 0.0  #

# Define Eye Landmarks
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))

# Function to compute Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

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
        face_img = gray[y:y+h, x:x+w]  # Crop face
        frame_color = (255, 255, 0)  # Yellow frame
        thickness = 3
        cv2.rectangle(frame, (x, y), (x + w, y + h), frame_color, thickness)
        if face_img.shape[0] == 0 or face_img.shape[1] == 0:
            print("Warning: Empty face image detected, skipping frame.")
            continue

        # Detect facial landmarks
        landmarks = predictor(gray, face)

        # Extract eye landmarks
        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE]

        # Compute EAR
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        # Preprocess image for CNN
        face_img_resized = cv2.resize(face_img, (64, 64)) / 255.0
        face_img_resized = face_img_resized.reshape(1, 64, 64, 1)

        # Predict using CNN
        cnn_prediction = cnn_model.predict(face_img_resized)
        cnn_label = np.argmax(cnn_prediction)  # 0 = Alert, 1 = Drowsy
        cnn_confidence = np.max(cnn_prediction)

        # Predict using Random Forest Model (Physiological + Behavioral Features)
        ml_features = np.array([[avg_EAR, perclos_x, blink_x, pupil_diameter, head_pitch, head_roll]])  # Dummy values for missing features
        ml_prediction = rf_model.predict(ml_features)[0]

        # Final Decision: If either CNN or ML predicts drowsiness, classify as Drowsy
        final_label = "DROWSY" if cnn_label == 1 or ml_prediction == 1 or avg_EAR < EAR_THRESHOLD else "ALERT"
        color = (0, 0, 255) if final_label == "DROWSY" else (0, 255, 0)

        # Perform Emotion Detection using DeepFace
        try:
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = results[0]['dominant_emotion']
        except:
            emotion = "Unknown"

        # Display labels
        cv2.putText(frame, f"{final_label} (CNN: {cnn_confidence:.2f})", (x, y - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Play alarm if drowsy
        if final_label == "DROWSY":
            drowsy_frames += 1
            if drowsy_frames >= 3:  # Trigger alarm after 5 consecutive drowsy frames
                pygame.mixer.music.load(alarm_sound)
                pygame.mixer.music.play()
        else:
            drowsy_frames = 0
            pygame.mixer.music.stop()

        cv2.imshow("Drowsiness Detection (CNN + ML)", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
