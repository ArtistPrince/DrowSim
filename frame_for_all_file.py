import cv2
import os
import numpy as np
import pandas as pd
import dlib

# Define base directories
eye_tracking_dir = "data/eye tracking"
video_dir = "data/video"
output_frames_base_dir = "extracted_frames"
output_csv_dir = "processed_data"

# Create necessary directories
os.makedirs(output_frames_base_dir, exist_ok=True)
os.makedirs(output_csv_dir, exist_ok=True)

# Initialize face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Ensure this file exists

def eye_aspect_ratio(eye_points):
    """Compute Eye Aspect Ratio (EAR) for drowsiness detection"""
    return (abs(eye_points[1][1] - eye_points[5][1]) + abs(eye_points[2][1] - eye_points[4][1])) / \
           (2.0 * abs(eye_points[0][0] - eye_points[3][0]))

# Loop through all 27 files
for i in range(1, 28):
    # Paths to input files
    eye_tracking_file = os.path.join(eye_tracking_dir, f"EyeTracking{i}.txt")
    video_path = os.path.join(video_dir, f"Face_cropped{i}.mp4")
    output_frames_dir = os.path.join(output_frames_base_dir, f"Face_cropped{i}")
    output_csv_file = os.path.join(output_csv_dir, f"processed_drowsiness_data_{i}.csv")

    # Ensure paths exist before proceeding
    if not os.path.exists(eye_tracking_file) or not os.path.exists(video_path):
        print(f"Skipping {i}: Missing file -> {eye_tracking_file} or {video_path}")
        continue

    # Create a separate directory for extracted frames of this video
    os.makedirs(output_frames_dir, exist_ok=True)

    # Load Eye Tracking Data
    eye_tracking_df = pd.read_csv(eye_tracking_file, delimiter=",")
    
    # Ensure required columns exist
    required_columns = ["time", "PERCLOS", "Blink", "PupilDiameter", "LeftPupilDiameter", "RightPupilDiameter"]
    missing_cols = [col for col in required_columns if col not in eye_tracking_df.columns]
    
    if missing_cols:
        print(f"Skipping {i}: Missing columns in {eye_tracking_file} -> {missing_cols}")
        continue

    eye_tracking_df = eye_tracking_df[required_columns]
    eye_tracking_df["time"] = pd.to_numeric(eye_tracking_df["time"], errors="coerce")

    # Extract frames from video
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    frame_skip = 10  # Extract every 10th frame for efficiency
    frame_times = []

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        if frame_count % frame_skip == 0:
            frame_filename = os.path.join(output_frames_dir, f"frame_{frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_times.append(frame_count / cap.get(cv2.CAP_PROP_FPS))  # Store frame timestamps
        frame_count += 1

    cap.release()

    if not frame_times:
        print(f"Skipping {i}: No frames extracted from {video_path}")
        continue

    # Synchronize eye-tracking data with frames
    num_frames = len(frame_times)
    max_time = eye_tracking_df["time"].max()
    frame_times = np.linspace(0, max_time, num_frames)  # Adjust frame timestamps
    eye_tracking_df["frame_index"] = np.searchsorted(frame_times, eye_tracking_df["time"])

    # Process each frame and detect faces/eyes
    facial_features = []
    for frame_file in sorted(os.listdir(output_frames_dir)):
        frame_path = os.path.join(output_frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faces = detector(gray)
        for face in faces:
            landmarks = predictor(gray, face)

            # Extract eye landmarks (left & right)
            left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
            right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]
            
            # Compute Eye Aspect Ratio (EAR)
            left_EAR = eye_aspect_ratio(left_eye)
            right_EAR = eye_aspect_ratio(right_eye)
            avg_EAR = (left_EAR + right_EAR) / 2.0

            facial_features.append([frame_file, avg_EAR])

    # Convert to DataFrame
    facial_features_df = pd.DataFrame(facial_features, columns=["frame_file", "EAR"])
    facial_features_df["frame_index"] = facial_features_df.index  # Align frame indices

    # Merge with eye-tracking data
    merged_data = pd.merge(facial_features_df, eye_tracking_df, on="frame_index", how="inner")

    # Save processed data
    merged_data.to_csv(output_csv_file, index=False)
    print(f"Processed data saved as '{output_csv_file}'")

