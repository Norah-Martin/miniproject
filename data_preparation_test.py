import os
import shutil
import random

# Set the path to the extracted dataset folder
dataset_path = "C:\Mini - Project\Drowsiness_detection\dataset"

# Set the percentage of videos to use for testing and validation (default is 20% each)
test_percent = 20
val_percent = 20

# Get a list of all the video files in the dataset folder
videos = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) if f.endswith('.mov')]

# Shuffle the list of videos
random.shuffle(videos)

# Calculate the number of videos to use for testing and validation
num_test_videos = int(len(videos) * (test_percent / 100.0))
num_val_videos = int(len(videos) * (val_percent / 100.0))

# Create the testing directory and move the videos to it
test_dir = os.path.join(dataset_path, "test")
os.makedirs(test_dir, exist_ok=True)
for i in range(num_test_videos):
    shutil.move(videos[i], os.path.join(test_dir, os.path.basename(videos[i])))

# Create the validation directory and move the videos to it
val_dir = os.path.join(dataset_path, "val")
os.makedirs(val_dir, exist_ok=True)
for i in range(num_test_videos, num_test_videos + num_val_videos):
    shutil.move(videos[i], os.path.join(val_dir, os.path.basename(videos[i])))

# Create the training directory and move the remaining videos to it
train_dir = os.path.join(dataset_path, "train")
os.makedirs(train_dir, exist_ok=True)
for i in range(num_test_videos + num_val_videos, len(videos)):
    shutil.move(videos[i], os.path.join(train_dir, os.path.basename(videos[i])))
