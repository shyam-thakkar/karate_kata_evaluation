import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Load the keypoints dataset from the CSV file
dataset_path = "C:/miniproject/keypoints_dataset.csv"
df = pd.read_csv(dataset_path)

# Extract features (keypoints and image paths) and labels from the dataset
keypoints = df.iloc[:, 2:].values  # Exclude the first two columns (Image and Class)
image_paths = df["Image"].values
y = df["Class"]

# Encode class labels using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
keypoints_train, keypoints_test, image_paths_train, image_paths_test, y_train, y_test = train_test_split(
    keypoints, image_paths, y_encoded, test_size=0.15, random_state=88
)

# Define the Dense Neural Network model for keypoints
dense_input = layers.Input(shape=(keypoints_train.shape[1],), name="dense_input")
dense_output = layers.Dense(128, activation='relu')(dense_input)
dense_output = layers.Dropout(0.3)(dense_output)
dense_output = layers.Dense(64, activation='relu')(dense_output)
dense_output = layers.Dense(len(label_encoder.classes_), activation='softmax')(dense_output)

# Define the CNN model for images
image_input = layers.Input(shape=(256, 256, 3), name="image_input")
cnn_output = layers.Conv2D(32, (3, 3), activation='relu')(image_input)
cnn_output = layers.MaxPooling2D((2, 2))(cnn_output)
cnn_output = layers.Conv2D(64, (3, 3), activation='relu')(cnn_output)
cnn_output = layers.MaxPooling2D((2, 2))(cnn_output)
cnn_output = layers.Flatten()(cnn_output)

# Concatenate the outputs from both models
concatenated = layers.concatenate([dense_output, cnn_output])

# Additional Dense layers for combined processing
combined_output = layers.Dense(128, activation='relu')(concatenated)
combined_output = layers.Dropout(0.3)(combined_output)
combined_output = layers.Dense(64, activation='relu')(combined_output)
final_output = layers.Dense(len(label_encoder.classes_), activation='softmax')(combined_output)

# Create the combined model using the functional API
combined_model = models.Model(inputs=[dense_input, image_input], outputs=final_output)

# Print the summary of the combined model
combined_model.summary()

# Compile the combined model
combined_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Create callbacks
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(256, 256))  # Adjust target_size as needed
    img_array = img_to_array(img)
    img_array /= 255.0  # Normalize pixel values to [0, 1]
    return img_array

# Train the combined model
combined_model.fit(
    {
        "dense_input": keypoints_train,
        "image_input": np.array([load_and_preprocess_image(img_path) for img_path in image_paths_train]),
    },
    y_train,
    epochs=100,
    batch_size=128,
    validation_split=0.2,
    callbacks=[tensorboard_callback, early_stopping],
)
# Evaluate the combined model on the test set
test_loss, test_accuracy = combined_model.evaluate(
    {"dense_input": keypoints_test, "image_input": np.array([load_and_preprocess_image(img_path) for img_path in image_paths_test])},
    y_test
)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the combined model
combined_model.save("C:/miniproject/combined_pose_model")

# Save label encoder classes
np.save("C:/miniproject/combined_pose_model/label_encoder_classes.npy", label_encoder.classes_)

# Evaluate the combined model using the evaluation function
evaluate_model(
    combined_model,
    {"dense_input": keypoints_test, "image_input": np.array([load_and_preprocess_image(img_path) for img_path in image_paths_test])},
    y_test
)
