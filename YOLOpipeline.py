# Vizai Sprint 2
# YOLOv8 Training Pipeline Script with ClearML
# Not running yet, just a script to test the pipeline

from ultralytics import YOLO
from clearml import Task
import os

print("Starting YOLOv8 training pipeline...")

# ClearML Setup
print("Initializing ClearML task...")
task = Task.init(project_name="Vizai", task_name="YOLOv8 Training Pipeline", task_type=Task.TaskTypes.training)
print("ClearML task initialized successfully")

# Configurations
print("Loading configurations...")
data_yaml_path = "data.yaml" 
pretrained_model = "yolov8n.pt" 
imgsz = 640
batch_size = 16
epochs = 50
project_output = "runs/train/vizai"
device = "cpu"

print(f"Configuration loaded: data_yaml={data_yaml_path}, model={pretrained_model}, device={device}")

# Load and Train Model
print("Loading YOLO model...")
model = YOLO(pretrained_model)
print("Model loaded successfully")

print("Starting training...")
model.train(
    data=data_yaml_path,
    epochs=epochs,
    imgsz=imgsz,
    batch=batch_size,
    project=project_output,
    device=device
)
print("Training completed")

# Evaluate on Validation Set
print("Evaluating model...")
metrics = model.val()
print("Validation Results:", metrics)

# Save Model
model_path = os.path.join(project_output, "weights", "best.pt")
print(f"Best model saved to: {model_path}")
