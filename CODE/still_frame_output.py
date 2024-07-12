import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
from collections import deque

# Load the Pose Classification Model
pose_model_path = "C:/miniproject/pose_class"
pose_model = load_model(pose_model_path)

# Load label encoder classes
label_encoder_classes_path = "C:/miniproject/pose_class/label_encoder_classes.npy"
label_encoder_classes = np.load(label_encoder_classes_path, allow_pickle=True)

# Load the MoveNet model from TensorFlow Hub
model = hub.load("https://www.kaggle.com/models/google/movenet/TensorFlow2/singlepose-thunder/4")
movenet = model.signatures['serving_default']

# Initialize a deque to store the last 15 predicted values
prediction_queue = deque(maxlen=15)

# Threshold for movement
movement_threshold = 0.05

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

        # Draw predicted class on the top left corner
        cv2.putText(processed_frame, f"Predicted Pose Class: {most_frequent_prediction}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return processed_frame, keypoints

# Main function to process video file
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
        processed_frame, prev_keypoints = process_frame(frame, prev_keypoints)

        # Display the frame
        cv2.imshow("Pose Classification", processed_frame)

        # Stop the app if 'Esc' key is pressed
        if cv2.waitKey(1) == 27:
            break

    # Release the video capture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Path to the video file
    video_path = 'C:/Users/SHYAM THAKKAR/Downloads/0123.mp4'
    process_video_file(video_path)
