import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import datetime

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

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=16, validation_split=0.2, callbacks=[early_stopping])

# Save the model
model.save("C:/miniproject/pose_class")

# Save label encoder classes
np.save("C:/miniproject/pose_class/label_encoder_classes.npy", label_encoder.classes_)

# Function to calculate and print various evaluation metrics
def evaluate_model(model, X_test, y_test):
    # Predict class probabilities
    y_prob = model.predict(X_test)

    # Predict classes
    y_pred = np.argmax(y_prob, axis=1)

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

    # Plot confusion matrix
    plot_confusion_matrix(confusion_mat, label_encoder.classes_)

# Function to plot confusion matrix
def plot_confusion_matrix(confusion_mat, classes):
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_mat, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

# Load the saved model
loaded_model = tf.keras.models.load_model("C:/miniproject/pose_class")

# Load the label encoder
label_encoder = LabelEncoder()
label_encoder.classes_ = np.load("C:/miniproject/pose_class/label_encoder_classes.npy",allow_pickle=True,)

# Evaluate the loaded model on the test set
evaluate_model(loaded_model, X_test, y_test)
