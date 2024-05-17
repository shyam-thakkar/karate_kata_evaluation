import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.callbacks import EarlyStopping
# Define a simple learning rate scheduler
def lr_scheduler(epoch, lr):
    if epoch % 10 == 0 and epoch > 0:
        lr = lr * 0.5  # adjust the learning rate as needed
    return lr
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Load the CSV file
csv_path = "C:/miniproject/keypoints_dataset.csv"
df = pd.read_csv(csv_path)

# Convert class labels to numerical labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['Class'])

# Define image size and channels
img_size = (224, 224)
channels = 3  # Assuming RGB images

# Load and preprocess the images
def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img)
    img_array /= 255.0  # Normalize pixel values to be between 0 and 1
    return img_array

# Apply the preprocessing function to all images
df['image'] = df['Image'].apply(load_and_preprocess_image)

# Convert the images and labels to numpy arrays
X = np.stack(df['image'])
y = df['label'].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D, Dropout, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(layers.Dropout(0.7))  # Increased dropout rate
model.add(layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
model.add(layers.Dense(len(label_encoder.classes_), activation='softmax'))

# Compile the CNN model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=100,batch_size=128, validation_split=0.1, validation_data=(X_test, y_test),callbacks=[LearningRateScheduler(lr_scheduler),early_stopping])

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc}')


model.save("C:/miniproject/pose_class_cnn")

np.save("C:/miniproject/pose_class_cnn/label_encoder_classes.npy", label_encoder.classes_)
