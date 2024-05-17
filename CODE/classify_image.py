# classify_image.py
import os
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from keypoint_extraction import extract_keypoints

# Rest of your code related to image classification...
@tf.function
def extract_keypoints(image_path):
    # Load the input image.
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.expand_dims(image, axis=0)
    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)

    # Run model inference.
    outputs = movenet(image)
    # Output is a [1, 1, 17, 3] tensor.
    keypoints = outputs['output_0']

    return keypoints

# Load the model from TF Hub outside the loop
model = hub.load("https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/singlepose-thunder/versions/4")
movenet = model.signatures['serving_default']

# Path to the new image
new_image_path = "C:/Users/SHYAM THAKKAR/Downloads/0342.jpg"


# Call the function to extract keypoints
keypoints = tf.py_function(extract_keypoints, [new_image_path], Tout=tf.float32)

# Flatten the keypoints array for easier indexing
keypoints_flat = keypoints.numpy().flatten()

# Prepare the feature vector for classification
feature_vector = keypoints_flat[0:51]  # Assuming 17 keypoints with (y, x, score) for each

# Load the trained pose classification model
pose_model_path = "C:/miniproject/pose_class"
pose_model = load_model(pose_model_path)

# Reshape the feature vector to match the input shape expected by the model
feature_vector = np.reshape(feature_vector, (1, -1))

# Make predictions
predictions = pose_model.predict(feature_vector)

# Decode predictions using LabelEncoder
label_encoder_classes_path = "C:/miniproject/pose_class/label_encoder_classes.npy"
label_encoder_classes = np.load(label_encoder_classes_path, allow_pickle=True)

# Decode predictions
predicted_class_index = np.argmax(predictions)
predicted_class = label_encoder_classes[predicted_class_index]

print(f"Predicted Pose Class: {predicted_class}")
