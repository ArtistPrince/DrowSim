import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import dlib
from deepface import DeepFace

# Load trained CNN model for drowsiness detection
model = load_model("drowsiness_cnn_model.h5")

# Initialize Dlib's face detector
detector = dlib.get_frontal_face_detector()

# Define image size (same as training)
img_size = 64

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

        # Preprocess the image for CNN (Drowsiness Detection)
        face_img_resized = cv2.resize(face_img, (img_size, img_size)) / 255.0
        face_img_resized = face_img_resized.reshape(1, img_size, img_size, 1)

        # Predict Drowsiness using CNN
        prediction = model.predict(face_img_resized)
        print(prediction)
        class_label = np.argmax(prediction)  
        print(class_label)# 0 = Alert, 1 = Drowsy
        confidence = np.max(prediction)
        print(confidence)

        # Assign label & color
        drowsiness_label = "DROWSY" if class_label == 1 else "ALERT"
        drowsiness_color = (0, 0, 255) if class_label == 1 else (0, 255, 0)

        # Perform Emotion Detection using DeepFace
        try:
            results = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = results[0]['dominant_emotion']
        except:
            emotion = "Unknown"

        # Display labels
        cv2.putText(frame, f"{drowsiness_label} ({confidence:.2f})", (x, y - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, drowsiness_color, 2)
        cv2.putText(frame, f"Emotion: {emotion}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), drowsiness_color, 2)

    # Show video feed
    cv2.imshow("Real-Time Drowsiness & Emotion Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
