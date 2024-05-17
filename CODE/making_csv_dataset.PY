import os
import tensorflow as tf
import tensorflow_hub as hub
import csv

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

# Path to the root directory of the dataset
dataset_root = "C:/miniproject/dataset"

# Output CSV file
output_csv_file = "C:/miniproject/keypoints_dataset.csv"

# Define header and keypoint names
header = ["Image", "Class", "Nose_X", "Nose_Y", "Nose_Confidence", "Left_Eye_X", "Left_Eye_Y", "Left_Eye_Confidence", "Right_Eye_X", "Right_Eye_Y", "Right_Eye_Confidence", "Left_Ear_X", "Left_Ear_Y", "Left_Ear_Confidence", "Right_Ear_X", "Right_Ear_Y", "Right_Ear_Confidence", "Left_Shoulder_X", "Left_Shoulder_Y", "Left_Shoulder_Confidence", "Right_Shoulder_X", "Right_Shoulder_Y", "Right_Shoulder_Confidence", "Left_Elbow_X", "Left_Elbow_Y", "Left_Elbow_Confidence", "Right_Elbow_X", "Right_Elbow_Y", "Right_Elbow_Confidence", "Left_Wrist_X", "Left_Wrist_Y", "Left_Wrist_Confidence", "Right_Wrist_X", "Right_Wrist_Y", "Right_Wrist_Confidence", "Left_Hip_X", "Left_Hip_Y", "Left_Hip_Confidence", "Right_Hip_X", "Right_Hip_Y", "Right_Hip_Confidence", "Left_Knee_X", "Left_Knee_Y", "Left_Knee_Confidence", "Right_Knee_X", "Right_Knee_Y", "Right_Knee_Confidence", "Left_Ankle_X", "Left_Ankle_Y", "Left_Ankle_Confidence", "Right_Ankle_X", "Right_Ankle_Y", "Right_Ankle_Confidence"]

# Open CSV file in write mode
with open(output_csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write the header to the CSV file
    writer.writerow(header)

    # Iterate through subdirectories (each subdirectory represents a class)
    for class_name in os.listdir(dataset_root):
        class_path = os.path.join(dataset_root, class_name)

        # Check if it's a directory
        if os.path.isdir(class_path):
            # Iterate through images in the class directory
            for image_name in os.listdir(class_path):
                if image_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_path, image_name)

                    # Call the function to extract keypoints
                    keypoints = tf.py_function(extract_keypoints, [image_path], Tout=tf.float32)

                    # Flatten the keypoints array for easier indexing
                    keypoints_flat = keypoints.numpy().flatten()

                    # Write image absolute path and class name to the CSV file
                    row = [os.path.abspath(image_path), class_name]

                    # Iterate through keypoints and append to the CSV row
                    for i in range(17):  # Assuming 17 keypoints
                        row.extend(keypoints_flat[i * 3: (i + 1) * 3])

                    writer.writerow(row)

                    # Print an update after each image is processed
                    print(f"Image processed: {image_path}")
