# step3_yolo_train.py
import os
from collections import Counter
import glob
from clearml import Task
from ultralytics import YOLO

# Step 3: Model training
def train_yolo_model():
    task = Task.init(project_name="Vizai", task_name="Pipeline Step 3 - YOLOv8 Training")

    # Use ClearML artifacts or local path
    dataset_path = os.path.abspath("data")
    yaml_path = os.path.join(dataset_path, "data.yaml")

    task.connect({"dataset_path": dataset_path, "yaml_path": yaml_path})

    # Load YOLOv8 model (nano version for speed)
    model = YOLO("yolov8n.pt")
    task.connect(model)

    # Train model
    results = model.train(
        data=yaml_path,
        epochs=50,
        batch=8,
        imgsz=320,
        device="cpu",
        name="yolov8_cpu",
        project="Vizai",
        augment=True
    )

    # Evaluate
    metrics = model.val(
        data=yaml_path,
        batch=8,
        imgsz=320,
        device='cpu',
        split='val'
    )

    logger = task.get_logger()
    logger.report_scalar("Validation Metrics", "mAP50", metrics.box.map50, 0)
    logger.report_scalar("Validation Metrics", "mAP50-95", metrics.box.map, 0)

    # Class distribution
    label_files = glob.glob(os.path.join(dataset_path, "train/labels/*.txt"))
    class_counts = Counter()
    for file in label_files:
        with open(file) as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1

    for cid, count in class_counts.items():
        logger.report_scalar("Class Distribution", f"class_{cid}", count, 0)

    # Upload model
    trained_model_path = os.path.join("runs/detect/yolov8_cpu/weights/best.pt")
    task.upload_artifact(name="best.pt", artifact_object=trained_model_path)

    print("YOLOv8 training and validation complete.")

if __name__ == "__main__":
    train_yolo_model()