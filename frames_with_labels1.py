import cv2
import os
import numpy as np
import pandas as pd
import dlib



# Define base directories
eye_tracking_dir = r"data\eye tracking"
video_dir = r"data\video"
output_frames_dir = "extracted_frames"

# Loop through all 27 files
for i in range(1, 28):
    # Create paths for eye-tracking files
    eye_tracking_file = os.path.join(eye_tracking_dir, f"eyetracking{i}.txt")
    
    # Create paths for video files
    video_path = os.path.join(video_dir, f"face_cropped{i}", "video.mp4")
    # Paths to input files
     
    # Create a directory for extracted frames
    os.makedirs(output_frames_dir, exist_ok=True)

        # Load Eye Tracking Data
    eye_tracking_df = pd.read_csv(eye_tracking_file, delimiter=",")
    eye_tracking_df = eye_tracking_df[["time", "PERCLOS", "Blink", "PupilDiameter", "LeftPupilDiameter", "RightPupilDiameter"]]
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

    # Synchronize eye-tracking data with frames
    num_frames = len(frame_times)
    max_time = eye_tracking_df["time"].max()
    frame_times = np.linspace(0, max_time, num_frames)  # Adjust frame timestamps
    eye_tracking_df["frame_index"] = np.searchsorted(frame_times, eye_tracking_df["time"])

    # Initialize face detector and landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download from dlib website

    def eye_aspect_ratio(eye_points):
        """Compute Eye Aspect Ratio (EAR) for drowsiness detection"""
        return (abs(eye_points[1][1] - eye_points[5][1]) + abs(eye_points[2][1] - eye_points[4][1])) / \
            (2.0 * abs(eye_points[0][0] - eye_points[3][0]))

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
    merged_data.to_csv("processed_drowsiness_data.csv", index=False)
    print("Processed data saved as 'processed_drowsiness_data.csv'")

