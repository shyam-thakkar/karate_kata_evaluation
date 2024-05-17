import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import TensorBoard
import datetime  # for timestamp in log directory
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np


# Load the keypoints dataset from the CSV file
dataset_path = "C:/miniproject/keypoints_dataset.csv"
df = pd.read_csv(dataset_path)

# Extract features (keypoints) and labels from the dataset
X = df.iloc[:, 2:].values  # Exclude the first column (image paths)
y = df["Image"].apply(lambda x: os.path.basename(os.path.dirname(x)))  # Extract class names from image paths

# Encode class labels using LabelEncoder
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)



# Define the neural network model
model = models.Sequential([
    layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(label_encoder.classes_), activation='softmax')  # Output layer with softmax activation for multi-class classification
])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler

# Define a learning rate schedule
def lr_schedule(epoch):
    initial_lr = 0.001
    drop = 0.5
    epochs_drop = 10.0
    lr = initial_lr * (drop ** (epoch / epochs_drop))
    return lr

# Create a learning rate scheduler
lr_scheduler = LearningRateScheduler(lr_schedule)
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model with the TensorBoard callback
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")


model.save("C:/miniproject/pose_class")

import numpy as np

# Assuming label_encoder is your trained LabelEncoder instance
np.save("C:/miniproject/pose_class/label_encoder_classes.npy", label_encoder.classes_)


# Function to calculate and print various evaluation metrics
def evaluate_model(model, X_test, y_test):
    # Predict class probabilities
    y_prob = model.predict(X_test)

    # Predict classes
    y_pred = tf.argmax(y_prob, axis=1).numpy()

    # Convert one-hot encoded labels to integers
    y_true = y_test

    # Calculate accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    # Calculate precision, recall, and F1 score
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')

    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1 Score: {f1 * 100:.2f}%")

    # Confusion matrix
    confusion_mat = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(confusion_mat)

# Call the function to evaluate the model
evaluate_model(model, X_test, y_test)
