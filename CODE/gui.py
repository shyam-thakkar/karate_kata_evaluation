import tkinter as tk
import math
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
from collections import deque
import ctypes

# Get the screen dimensions
user32 = ctypes.windll.user32
screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)

# Load the MoveNet model from TensorFlow Hub
model = hub.load("https://www.kaggle.com/models/google/movenet/TensorFlow2/singlepose-thunder/4")
movenet = model.signatures['serving_default']

# Load the trained pose classification model
pose_model_path = "C:/miniproject/pose_class"
pose_model = load_model(pose_model_path)

# Load label encoder classes
label_encoder_classes_path = "C:/miniproject/pose_class/label_encoder_classes.npy"
label_encoder_classes = np.load(label_encoder_classes_path, allow_pickle=True)

# Initialize a deque to store the last 15 predicted values
prediction_queue = deque(maxlen=15)
predictionsss = []
# Threshold for movement
movement_threshold = 0.0000000000000001

# Function to calculate difference in keypoints
def calculate_keypoints_difference(prev_keypoints, curr_keypoints):
    # Exclude the confidence scores (every third value)
    prev_keypoints = prev_keypoints[::3]
    curr_keypoints = curr_keypoints[::3]

    # Calculate absolute difference between keypoints
    keypoints_difference = np.sum(np.abs(prev_keypoints - curr_keypoints))

    return keypoints_difference

# Function to process each frame and perform pose classification
def process_frame(frame, prev_keypoints):
    # Make a copy of the original frame
    processed_frame = frame.copy()

    # Resize frame to 256x256
    frame = cv2.resize(frame, (256, 256))
    # Convert frame to RGB (MoveNet expects RGB format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Expand dimensions to match the shape [1, height, width, 3]
    frame_rgb = np.expand_dims(frame_rgb, axis=0).astype(np.uint8)

    # Convert frame to int32
    frame_rgb = tf.cast(frame_rgb, tf.int32)

    # Run inference using MoveNet to extract keypoints
    outputs = movenet(tf.constant(frame_rgb))
    keypoints = outputs['output_0'].numpy().flatten()

    # If prev_keypoints is None, set it to the current keypoints
    if prev_keypoints is None:
        prev_keypoints = keypoints

    # Calculate difference in keypoints
    keypoints_difference = calculate_keypoints_difference(prev_keypoints, keypoints)

    # If movement is below threshold, perform pose classification
    if keypoints_difference < movement_threshold:
        # Prepare feature vector for pose classification
        feature_vector = keypoints[0:51].reshape(1, -1)

        # Perform pose classification
        predictions = pose_model.predict(feature_vector)
        predicted_class_index = np.argmax(predictions)
        predicted_class = label_encoder_classes[predicted_class_index]

        # Add the predicted class to the queue
        prediction_queue.append(predicted_class)

        # Get the most frequent prediction from the queue
        most_frequent_prediction = max(set(prediction_queue), key=prediction_queue.count)
        predictionsss.append(most_frequent_prediction)
        # Draw predicted class on the top left corner
        cv2.putText(processed_frame, f"Predicted Pose Class: {most_frequent_prediction}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 0), 4)

    return processed_frame, most_frequent_prediction

def create_chunks(lst):
    chunk_size = len(lst) / 20
    chunks = [lst[i:i + int(chunk_size)] for i in range(0, len(lst), int(chunk_size))]
    return chunks

# Function to process video file
def process_video_file(video_path):
    # Open video capture
    video_capture = cv2.VideoCapture(video_path)
    prev_keypoints = None
    while True:
        # Read frame-by-frame
        ret, frame = video_capture.read()

        # Break the loop if no frame is captured
        if not ret:
            print("Error: Unable to capture frame.")
            break

        # Process the frame and perform pose classification
        processed_frame, predicted = process_frame(frame, prev_keypoints)

        # Resize frame for display
        processed_frame = cv2.resize(processed_frame, (1080, 720))  # Adjust the size as needed

        # Display the frame
        cv2.imshow("Pose Classification", processed_frame)

        # Stop the app if 'Esc' key is pressed
        if cv2.waitKey(1) == 27:
            break

    # Release the video capture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()
    chunks = create_chunks(predictionsss)
    final_user = []
    grade = 0
    predefined_sequence = ['L_L_GB', 'L_R_S', 'R_R_GB', 'R_L_S', 'F_L_GB', 'F_R_S', 'F_L_S', 'F_R_S', 'R_L_GB',
                           'R_R_S', 'L_R_GB', 'L_L_S', 'B_L_GB', 'B_R_S', 'B_L_S', 'B_R_S', 'L_L_GB', 'L_R_S',
                           'R_R_GB', 'R_L_S']
    for i in range(0,20):
       grade_count= chunks[i].count(predefined_sequence[i])
       grade_count=grade_count/20
       grade=grade+grade_count
    print(math.ceil(grade/2)) # Return the calculated grade
    return(math.ceil(grade/2)) # Return the calculated grade

# GUI class
class PoseClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Pose Classifier")
        self.root.geometry("1080x720")  # Adjust the size of the initial dialog box

        self.image_path = None

        self.label = tk.Label(root, text="Select a video for classification:")
        self.label.pack(pady=10)

        self.browse_button = tk.Button(root, text="Browse", command=self.browse_video)
        self.browse_button.pack()

        self.predict_button = tk.Button(root, text="Predict", command=self.predict_video)
        self.predict_button.pack(pady=10)

        self.result_label = tk.Label(root, text="")
        self.result_label.pack()

    def browse_video(self):
        file_path = filedialog.askopenfilename(initialdir="/", title="Select Video",
                                               filetypes=(("Video files", "*.mp4;*.avi"), ("all files", "*.*")))
        if file_path:
            self.video_path = file_path

    def predict_video(self):
        if hasattr(self, 'video_path') and self.video_path:
            grade = process_video_file(self.video_path)
            self.result_label.config(text=f"Grade: {grade}")
        else:
            self.result_label.config(text="Please select a video first.")

def main():
    root = tk.Tk()
    gui = PoseClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
