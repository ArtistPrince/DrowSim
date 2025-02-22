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

# Drowsiness thresholds
PERCLOS_THRESHOLD = 50  # PERCLOS > 50% means drowsy
BLINK_THRESHOLD = 2      # More than 2 blinks per interval means drowsy
EAR_THRESHOLD = 0.2      # EAR < 0.2 indicates drowsiness

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

    # Create labeled output directories for extracted frames
    alert_dir = os.path.join(output_frames_dir, "alert")
    drowsy_dir = os.path.join(output_frames_dir, "drowsy")
    os.makedirs(alert_dir, exist_ok=True)
    os.makedirs(drowsy_dir, exist_ok=True)

    # Load Eye Tracking Data
    eye_tracking_df = pd.read_csv(eye_tracking_file, delimiter=",")

    # Ensure required columns exist
    required_columns = ["time", "PERCLOS", "Blink", "PupilDiameter", "LeftPupilDiameter", "RightPupilDiameter"]
    missing_cols = [col for col in required_columns if col not in eye_tracking_df.columns]
    
    if missing_cols:
        print(f"Skipping {i}: Missing columns in {eye_tracking_file} -> {missing_cols}")
        continue

    # Fill missing PERCLOS values (if needed)
    eye_tracking_df["PERCLOS"] = eye_tracking_df["PERCLOS"].fillna(eye_tracking_df["Blink"] * 20)  # Approximate

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
            frame_filename = f"frame_{frame_count}.jpg"
            frame_times.append((frame_filename, frame, frame_count / cap.get(cv2.CAP_PROP_FPS)))  # Store frame timestamps
        frame_count += 1

    cap.release()

    if not frame_times:
        print(f"Skipping {i}: No frames extracted from {video_path}")
        continue

    # Synchronize eye-tracking data with frames
    num_frames = len(frame_times)
    max_time = eye_tracking_df["time"].max()
    
    frame_times1 = np.array([item[2] for item in frame_times], dtype=np.float32)

    frame_timestamps = np.linspace(0, max_time, num_frames)  # Adjust frame timestamps
    eye_tracking_df["frame_index"] = np.searchsorted(frame_timestamps, eye_tracking_df["time"])

    # Process each frame and detect faces/eyes
    facial_features = []
    for frame_filename, frame, frame_time in frame_times:
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

            # Find the closest eye-tracking data point
            closest_row = eye_tracking_df.iloc[(eye_tracking_df["time"] - frame_time).abs().idxmin()]
            perclos = closest_row["PERCLOS"]
            blink = closest_row["Blink"]

            # Label frame as "drowsy" or "alert"
            label = "drowsy" if (perclos > PERCLOS_THRESHOLD or blink > BLINK_THRESHOLD or avg_EAR < EAR_THRESHOLD) else "alert"

            # Save frame in corresponding labeled folder
            output_frame_path = os.path.join(output_frames_dir, label, frame_filename)
            cv2.imwrite(output_frame_path, frame)

            # Store extracted data
            facial_features.append([frame_filename, avg_EAR, perclos, blink, label])

    # Convert to DataFrame
    facial_features_df = pd.DataFrame(facial_features, columns=["frame_file", "EAR", "PERCLOS", "Blink", "Label"])
    '''print("Facial Features Columns:", facial_features_df.columns)
    print("Eye Tracking Columns:", eye_tracking_df.columns)
    '''
    facial_features_df["frame_file"] = facial_features_df["frame_file"].str.extract(r'(\d+)').astype(int)
    # Merge with eye-tracking data
    merged_data = pd.merge(facial_features_df, eye_tracking_df, left_on="frame_file", right_on="frame_index", how="inner")

    # Save processed data
    merged_data.to_csv(output_csv_file, index=False)
    print(f"Processed data saved as '{output_csv_file}' and labeled frames stored.")
