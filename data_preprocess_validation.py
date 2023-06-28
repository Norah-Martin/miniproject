import cv2
import numpy as np
import os


# Set the path to the training directory
val_dir = "C:\Mini - Project\Drowsiness_detection\dataset/val"

# Set the target size for the frames
target_size = (224, 224)

# Loop over each video in the training directory
for filename in os.listdir(val_dir):
    if not filename.endswith(".mov"):
        continue

    # Create a new directory for the frames
    frames_dir = os.path.join(val_dir, os.path.splitext(filename)[0])
    os.makedirs(frames_dir, exist_ok=True)

    # Open the video file
    video_path = os.path.join(val_dir, filename)
    cap = cv2.VideoCapture(video_path)

    # Initialize a frame count
    frame_count = 0

    # Loop over each frame in the video
    while True:
        # Read a frame from the video
        ret, frame = cap.read()

        # If the frame couldn't be read, we've reached the end of the video
        if not ret:
            break

        # Resize the frame to the target size
        resized_frame = cv2.resize(frame, target_size)

        # Save the frame as an image file in the frames directory
        frame_path = os.path.join(frames_dir, f"frame{frame_count}.jpg")
        cv2.imwrite(frame_path, resized_frame)

        # Increment the frame count
        frame_count += 1

    # Release the video capture object
    cap.release()

# Set the path to the training directory
