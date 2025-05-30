import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 model
model = YOLO("runs/detect/yolov8_cpu2/weights/best.pt")

# UI setup
st.title("🔍 Real-time YOLOv8 Webcam Detection")
run = st.checkbox("Start Webcam")

# Placeholder for video frames
frame_placeholder = st.empty()

if run:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Could not open webcam.")
    else:
        while run:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to grab frame.")
                break

            # Run detection
            results = model(frame, imgsz=640, verbose=False)[0]

            # Draw results on frame
            annotated_frame = results.plot()

            # Show frame in Streamlit
            frame_placeholder.image(annotated_frame, channels="BGR", use_column_width=True)

            # Update checkbox state (this is what keeps Streamlit from rerunning incorrectly)
            run = st.session_state.get("Start Webcam", False)

        cap.release()
