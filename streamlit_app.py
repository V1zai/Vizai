import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import time
import numpy as np

# Load model once
model = YOLO("runs/detect/yolov8_cpu2/weights/best.pt")

# Streamlit page config
st.title("Real-time YOLOv8 Webcam Detection")
stframe = st.empty()

# Function to capture frames from webcam and process
def process_camera():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame.")
            break

        # Run YOLOv8 on the frame
        results = model(frame, imgsz=640, verbose=False)[0]

        # Render the results
        annotated_frame = results.plot()

        # Convert BGR (OpenCV) to RGB (Streamlit)
        stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        # Optional: Break loop using Streamlit button
        if st.button("Stop"):
            break

    cap.release()

# Start button
if st.button("Start Webcam Detection"):
    process_camera()
