# base_train_task.py
from clearml import Task
from ultralytics import YOLO
import torch
import yaml
import os

task = Task.init(project_name="Vizai", task_name="YOLOv8 Base Train Task", task_type=Task.TaskTypes.training)

# Get parameters (connected from HPO)
num_epochs = int(task.get_parameter("num_epochs", 100))
batch_size = int(task.get_parameter("batch_size", 8))
learning_rate = float(task.get_parameter("learning_rate", 0.001))
weight_decay = float(task.get_parameter("weight_decay", 1e-5))
data_yaml_path = task.get_parameter("data_yaml_path", "path/to/data.yaml")

# Load data config if needed
with open(data_yaml_path) as f:
    data_cfg = yaml.safe_load(f)

device = 0 if torch.cuda.is_available() else "cpu"

model = YOLO("yolov8n.pt")

model.train(
    data=data_yaml_path,
    epochs=num_epochs,
    batch=batch_size,
    imgsz=320,
    device=device,
    name="yolov8",
    project="Vizai",
    augment=True,
    lr0=learning_rate,
    weight_decay=weight_decay,
)

metrics = model.val(
    data=data_yaml_path,
    batch=batch_size,
    imgsz=320,
    device=device,
    split='val'
)

logger = task.get_logger()
logger.report_scalar("Validation Metrics", "mAP50", metrics.box.map50, 0)
logger.report_scalar("Validation Metrics", "mAP50-95", metrics.box.map, 0)

best_model_path = os.path.join("runs/detect/yolov8/weights/best.pt")
task.upload_artifact("best.pt", best_model_path)

print("Training complete.")
