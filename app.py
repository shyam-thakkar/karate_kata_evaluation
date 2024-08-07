import os
import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.models import load_model
from collections import deque
import math
import streamlit as st

# Setup the directories for upload
UPLOAD_FOLDER = './uploads'
PROCESSED_FOLDER = './processed'
DEMO_VIDEO_PATH = './demo_videos'
pose_model_path = "./pose_class/posemodel.keras"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PROCESSED_FOLDER):
    os.makedirs(PROCESSED_FOLDER)

@st.cache_resource
def load_model_movenet():
    model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
    moveneta = model.signatures['serving_default']
    return moveneta

movenet = load_model_movenet()

@st.cache_resource
def load_model_pose():
    pose_modela = load_model(pose_model_path)
    return pose_modela

pose_model = load_model_pose()

# Load label encoder classes
label_encoder_classes_path = "./pose_class/label_encoder_classes.npy"
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

def process_video_file(video_path, temp_output_path):
    video_capture = cv2.VideoCapture(video_path)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_capture.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(temp_output_path, fourcc, fps, (frame_width, frame_height))

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
    return grade, temp_output_path

def convert_to_mp4(input_path, output_path):
    command = f'ffmpeg -y -i  "{input_path}" -vcodec libx264 -crf 23 "{output_path}"'
    os.system(command)

st.title('Karate Kata Evaluation🥋')

uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
demo_button = st.button("Use Demo Video")

# Function to delete previous files
def delete_previous_files():
    for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER]:
        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)
            if os.path.isfile(file_path):
                os.remove(file_path)

# State to keep track of whether the demo button has been clicked
if 'demo_button_clicked' not in st.session_state:
    st.session_state.demo_button_clicked = False

# Check if the demo button was clicked
if demo_button:
    st.session_state.demo_button_clicked = True

# Processing the uploaded file
if uploaded_file is not None:
    delete_previous_files()
    file_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
    with open(file_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())

    temp_output_path = os.path.join(PROCESSED_FOLDER, f"temp_{os.path.basename(file_path)}.avi")
    final_output_path = os.path.join(PROCESSED_FOLDER, f"processed_{os.path.basename(file_path)}.mp4")

    grade, temp_processed_video_path = process_video_file(file_path, temp_output_path)
    convert_to_mp4(temp_processed_video_path, final_output_path)

    st.write(f"Grade: {grade}")

    if os.path.exists(final_output_path):
        with open(final_output_path, "rb") as file:
            mp4_bytes = file.read()
            st.video(mp4_bytes)
    else:
        st.error("Failed to convert video to MP4 format.")

# Display the selectbox if the demo button has been clicked
if st.session_state.demo_button_clicked:
    # List files in demo videos folder and add a default option
    files = os.listdir(DEMO_VIDEO_PATH)
    files.insert(0, 'Select a file...')

    selected_file = st.selectbox('Choose a Demo Video:', files)
    
    # Processing the demo video file
    if selected_file != "Select a file...":
        delete_previous_files()
        demo_temp_output_path = os.path.join(PROCESSED_FOLDER, "temp_demo_video.avi")
        demo_final_output_path = os.path.join(PROCESSED_FOLDER, "processed_demo_video.mp4")
        file_path = os.path.join(DEMO_VIDEO_PATH, selected_file)
        grade, demo_temp_processed_video_path = process_video_file(file_path, demo_temp_output_path)
        convert_to_mp4(demo_temp_processed_video_path, demo_final_output_path)

        st.write(f"Grade: {grade}")

        if os.path.exists(demo_final_output_path):
            with open(demo_final_output_path, "rb") as file:
                mp4_bytes = file.read()
                st.video(mp4_bytes)
        else:
            st.error("Failed to convert demo video to MP4 format.")
