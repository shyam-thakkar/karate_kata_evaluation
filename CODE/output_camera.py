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
movenet = hub.load("https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/singlepose-thunder/versions/4").signatures['serving_default']

# Initialize a deque to store the last 20 predicted values
prediction_queue = deque(maxlen=20)

# Function to process each frame and perform pose classification
def process_frame(frame):
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

    return frame, most_frequent_prediction

# Main function to process live camera feed
def process_camera():
    # Open camera capture
    video_capture = cv2.VideoCapture(0)  # Use 0 for the default camera
    
    while True:
        # Read frame-by-frame
        ret, frame = video_capture.read()

        # Process the frame and perform pose classification
        processed_frame, predicted_pose = process_frame(frame)

        # Draw a rectangle for the predicted class
        cv2.rectangle(processed_frame, (0, processed_frame.shape[0] - 50), (processed_frame.shape[1], processed_frame.shape[0]), (0, 0, 0), -1)
        cv2.putText(processed_frame, f"Predicted Pose Class: {predicted_pose}", (10, processed_frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Display the frame
        cv2.imshow("Pose Classification", processed_frame)

        # Stop the app if 'Esc' key is pressed
        if cv2.waitKey(1) == 27:
            break

    # Release the camera capture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_camera()
