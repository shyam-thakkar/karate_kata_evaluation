import cv2
import os
import csv

def extract_frames(video_path, output_folder):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Create CSV file for labeling
    frame_number = 0

    while True:
            # Read a frame from the video
            ret, frame = cap.read()

            if not ret:
                break  # Break if the video is finished

            # Save the frame as an image
            frame_filename = f"{frame_number:04d}.jpg"
            frame_path = os.path.join(output_folder, frame_filename)
            cv2.imwrite(frame_path, frame)

            # Write the frame filename and label to the CSV file
           
            frame_number += 1

    # Release the video capture and close the CSV file
    cap.release()

# Example usage:
fname=3

video_path = 'C:/miniproject/videos/10.mp4'
output_folder = 'C:/miniproject/videos/data/10'


extract_frames(video_path, output_folder)