import cv2
import os
import random

def extract_random_frame(video_path, output_dir):
    # Check if the output directory exists, create if not
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return

    # Get total number of video frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Select a random frame
    random_frame_number = random.randint(0, total_frames - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, random_frame_number)
    
    success, frame = cap.read()
    if success:
        # Construct the output path
        base_name = os.path.basename(video_path)
        file_name = os.path.splitext(base_name)[0] + "_frame.jpeg"
        output_path = os.path.join(output_dir, file_name)

        # Save the frame
        cv2.imwrite(output_path, frame)
        print(f"Random frame extracted and saved to {output_path}")
    else:
        print(f"Failed to extract frame from {video_path}")
    
    # Release the video capture object
    cap.release()

def extract_random_frame_from_directory(directory_path, output_dir):
    for filename in os.listdir(directory_path):
        if filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
            video_path = os.path.join(directory_path, filename)
            extract_random_frame(video_path, output_dir)

# Example usage
source_directory = '/Users/zephyrwang/Desktop/Work/DeepLabCut/jCSDS/jCSDS_4-30_videos/mp4'
output_directory = '/Users/zephyrwang/Desktop/Work/DeepLabCut/jCSDS/jCSDS_4-30_videos/frames'
extract_random_frame_from_directory(source_directory, output_directory)
