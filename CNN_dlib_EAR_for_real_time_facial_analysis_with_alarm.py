import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import dlib
from scipy.spatial import distance
import pygame
from deepface import DeepFace

# Load trained CNN model for drowsiness detection
model = load_model("drowsiness_cnn_model.h5")

# Initialize Dlib's face & landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

# Initialize Pygame for sound alerts
pygame.mixer.init()
alarm_sound = "alarm.wav"  # Place an alert sound file in the same directory

# Define EAR threshold for drowsiness
EAR_THRESHOLD = 0.25
drowsy_frames = 0  # Track consecutive drowsy frames

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
        face_img = gray[y:y+h, x:x+w]  # Crop face

        # Detect facial landmarks
        landmarks = predictor(gray, face)

        # Extract eye landmarks
        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in LEFT_EYE]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in RIGHT_EYE]

        # Compute EAR
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        # Preprocess the image for CNN
        face_img_resized = cv2.resize(face_img, (64, 64)) / 255.0
        face_img_resized = face_img_resized.reshape(1, 64, 64, 1)

        # Predict Drowsiness using CNN
        prediction = model.predict(face_img_resized)
        class_label = np.argmax(prediction)  # 0 = Alert, 1 = Drowsy
        confidence = np.max(prediction)

        # Assign label based on EAR & CNN
        if avg_EAR < EAR_THRESHOLD or class_label == 1:
            drowsiness_label = "DROWSY"
            drowsiness_color = (0, 0, 255)
            drowsy_frames += 1
        else:
            drowsiness_label = "ALERT"
            drowsiness_color = (0, 255, 0)
            drowsy_frames = 0

        # Perform Emotion Detection using DeepFace
        try:
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = results[0]['dominant_emotion']
        except:
            emotion = "Unknown"

        # Display labels
        cv2.putText(frame, f"{drowsiness_label} ({confidence:.2f})", (x, y - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, drowsiness_color, 2)
        cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(frame, f"EAR: {avg_EAR:.2f}", (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), drowsiness_color, 2)

        # Play alarm if drowsy for consecutive frames
        if drowsy_frames >= 5:
            pygame.mixer.music.load(alarm_sound)
            pygame.mixer.music.play()
        else:
            pygame.mixer.music.stop()

    # Show video feed
    cv2.imshow("Real-Time Drowsiness & Emotion Detection (EAR + CNN)", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
