from ultralytics import YOLO
import cv2
import os

# Paths (edit as needed)
model_path = "Vizai/yolov8_cpu4/weights/best.pt"
video1_path = "data/video/PXL_20250423_073320994.TS.mp4"
video2_path = "data/video/PXL_20250423_073418424.TS.mp4"
video3_path = "data/video/PXL_20250423_073622320.TS.mp4"

output_video_path = "data/video/output_detected.mp4"

# Load model
model = YOLO(model_path)

cap = cv2.VideoCapture(video2_path) # Change video path here
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = cap.get(cv2.CAP_PROP_FPS)

# Read the first frame to determine rotated size
ret, frame = cap.read()
if not ret:
    print("Failed to read video")
    cap.release()
    exit()

# Fix orientation
frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # Adjust as needed

height, width = frame.shape[:2]
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

print("Running inference on video...")

while True:
    # Use the already-read and rotated first frame
    results = model(frame)
    annotated_frame = results[0].plot()
    out.write(annotated_frame)

    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # Adjust as needed

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video with detections saved to {output_video_path}")