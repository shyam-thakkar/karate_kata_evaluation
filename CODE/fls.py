from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
from collections import deque
import numpy as np
import cv2
import math
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['PROCESSED_FOLDER']):
    os.makedirs(app.config['PROCESSED_FOLDER'])

# Load the MoveNet model from TensorFlow Hub
model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
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

# Function to process each frame and perform pose classification
def process_frame(frame):
    processed_frame = frame.copy()
    frame = cv2.resize(frame, (256, 256))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb = np.expand_dims(frame_rgb, axis=0).astype(np.uint8)
    frame_rgb = tf.cast(frame_rgb, tf.int32)

    outputs = movenet(tf.constant(frame_rgb))
    keypoints = outputs['output_0'].numpy().flatten()

    feature_vector = keypoints[0:51].reshape(1, -1)
    predictions = pose_model.predict(feature_vector)
    predicted_class_index = np.argmax(predictions)
    predicted_class = label_encoder_classes[predicted_class_index]

    prediction_queue.append(predicted_class)
    most_frequent_prediction = max(set(prediction_queue), key=prediction_queue.count)
    predictionsss.append(most_frequent_prediction)

    cv2.putText(processed_frame, f"Predicted Pose Class: {most_frequent_prediction}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.25, (0, 0, 0), 4)

    return processed_frame, most_frequent_prediction

def create_chunks(lst):
    chunk_size = len(lst) / 20
    chunks = [lst[i:i + int(chunk_size)] for i in range(0, len(lst), int(chunk_size))]
    return chunks

def process_video_file(video_path, output_path):
    video_capture = cv2.VideoCapture(video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        processed_frame, predicted = process_frame(frame)
        out.write(processed_frame)

    video_capture.release()
    out.release()

    chunks = create_chunks(predictionsss)
    grade = 0
    predefined_sequence = ['L_L_GB', 'L_R_S', 'R_R_GB', 'R_L_S', 'F_L_GB', 'F_R_S', 'F_L_S', 'F_R_S', 'R_L_GB',
                           'R_R_S', 'L_R_GB', 'L_L_S', 'B_L_GB', 'B_R_S', 'B_L_S', 'B_R_S', 'L_L_GB', 'L_R_S',
                           'R_R_GB', 'R_L_S']
    for i in range(20):
        grade_count = chunks[i].count(predefined_sequence[i])
        grade_count = grade_count / 20
        grade += grade_count
    grade = math.ceil(grade / 2)
    return grade, output_path

@app.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    output_path = os.path.join(app.config['PROCESSED_FOLDER'], f"processed_{filename}")
    grade, processed_video_path = process_video_file(file_path, output_path)

    return jsonify({"grade": grade, "processed_video_filename": f"processed_{filename}"})

@app.route('/video/<filename>')
def get_video(filename):
    return send_file(os.path.join(app.config['PROCESSED_FOLDER'], filename))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
