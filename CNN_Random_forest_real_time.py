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
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Initialize Pygame for sound alerts
pygame.mixer.init()
alarm_sound = "alarm.wav"

# EAR and Yawning Thresholds
EAR_THRESHOLD = 0.25
MAR_THRESHOLD = 0.5  # Lowered for better yawning detection
drowsy_frames = 0
yawning_frames = 0

# Define Eye & Mouth Landmarks
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
MOUTH = list(range(60, 68))  # Inner mouth landmarks

# Function to compute Eye Aspect Ratio (EAR)
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Function to compute Mouth Aspect Ratio (MAR) for yawning
def mouth_aspect_ratio(mouth):
    if len(mouth) < 10:  # Ensure enough mouth points detected
        return 0
    A = distance.euclidean(mouth[2], mouth[10])  
    B = distance.euclidean(mouth[4], mouth[8])  
    C = distance.euclidean(mouth[0], mouth[6])  
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

        # Draw a frame around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 3)

        # Detect facial landmarks
        landmarks = predictor(gray, face)

        # Extract eye and mouth landmarks
        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE]
        mouth = [(landmarks.part(n).x, landmarks.part(n).y) for n in MOUTH]
        
        # Compute EAR (Eye Aspect Ratio)
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        # Compute MAR (Mouth Aspect Ratio) for yawning detection
        mar = mouth_aspect_ratio(mouth)

        # Debug: Print MAR values to verify yawning detection
        print(f"MAR: {mar:.2f}")

        # Predict Drowsiness using CNN
        face_img = gray[y:y+h, x:x+w]
        if face_img.shape[0] > 0 and face_img.shape[1] > 0:
            face_img_resized = cv2.resize(face_img, (64, 64)) / 255.0
            face_img_resized = face_img_resized.reshape(1, 64, 64, 1)
            cnn_prediction = cnn_model.predict(face_img_resized)
            cnn_label = np.argmax(cnn_prediction)  
            cnn_confidence = np.max(cnn_prediction)
        else:
            cnn_label = 0  
            cnn_confidence = 1.0

        # Predict using Random Forest Model
        ml_features = np.array([[avg_EAR, 50, 2, 3.5, 0.0, 0.0]])  
        ml_prediction = rf_model.predict(ml_features)[0]

        # Final Decision: Drowsy if either CNN, ML, or EAR threshold detects drowsiness
        final_label = "DROWSY" if cnn_label == 1 or ml_prediction == 1 or avg_EAR < EAR_THRESHOLD else "ALERT"
        color = (0, 0, 255) if final_label == "DROWSY" else (0, 255, 0)

        # Check if yawning is detected
        yawning_label = "Yawning" if mar > MAR_THRESHOLD else "No Yawn"

        # Perform Emotion Detection using DeepFace
        try:
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = results[0]['dominant_emotion']
        except:
            emotion = "Unknown"

        # Display Labels
        cv2.putText(frame, f"{final_label} (CNN: {cnn_confidence:.2f})", (x, y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Yawning: {yawning_label}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)  
        cv2.putText(frame, f"Emotion: {emotion}", (x, y - 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)  
        cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (x, y + h + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Play alarm if drowsy or yawning
        if final_label == "DROWSY":
            drowsy_frames += 1
        else:
            drowsy_frames = 0

        if mar > MAR_THRESHOLD:
            yawning_frames += 1
        else:
            yawning_frames = 0

        if drowsy_frames >= 3 or yawning_frames >= 2:  
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
