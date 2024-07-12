# keypoint_extraction.py
import os
import tensorflow as tf
import tensorflow_hub as hub

# Download the model from TF Hub outside the loop
model = hub.load("https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/singlepose-thunder/versions/4")
movenet = model.signatures['serving_default']

# Function to extract keypoints from an image
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
