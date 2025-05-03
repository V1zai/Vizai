# yolo_pipeline_decorator.py
import os
import yaml
from collections import Counter
import glob
from clearml import Dataset, Task
from clearml.automation.controller import PipelineDecorator
from ultralytics import YOLO
import torch


@PipelineDecorator.component(cache=False, execution_queue='default')
def upload_dataset():
    task = Task.current_task()
    workspace_path = os.path.abspath(".")
    dataset_path = os.path.join(workspace_path, "data")

    # Upload dataset to ClearML
    dataset = Dataset.create(
        dataset_name="Vizai Dataset",
        dataset_project="Vizai",
        dataset_tags=["YOLOv8", "training"]
    )
    dataset.add_files(path=dataset_path)
    dataset.upload()
    dataset.finalize()

    # Save data.yaml
    class_names = ['barrier', 'bicycle', 'car', 'crosswalk', 'dog', 'person',
                   'pole', 'shutters', 'signboard', 'trash_can', 'tree', 'truck']
    data_yaml = {
        "path": dataset_path,
        "train": os.path.join(dataset_path, "train/images"),
        "val": os.path.join(dataset_path, "valid/images"),
        "names": class_names
    }
    yaml_path = os.path.join(dataset_path, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    task.upload_artifact("data_yaml", yaml_path)
    return yaml_path


@PipelineDecorator.component(cache=False, execution_queue='default')
def train_model(data_yaml_path):
    task = Task.current_task()
    task.connect({"data_yaml_path": data_yaml_path})

    model = YOLO("yolov8n.pt")
    task.connect(model)

    device = 0 if torch.cuda.is_available() else "cpu"

    # Train
    model.train(
        data=data_yaml_path,
        epochs=100,
        batch=8,
        imgsz=320,
        device=device,
        name="yolov8",
        project="Vizai",
        augment=True
    )

    # Validate
    metrics = model.val(
        data=data_yaml_path,
        batch=8,
        imgsz=320,
        device=device,
        split='val'
    )

    logger = task.get_logger()
    logger.report_scalar("Validation Metrics", "mAP50", metrics.box.map50, 0)
    logger.report_scalar("Validation Metrics", "mAP50-95", metrics.box.map, 0)

    # Class distribution
    dataset_path = os.path.dirname(data_yaml_path)
    label_files = glob.glob(os.path.join(dataset_path, "train/labels/*.txt"))
    class_counts = Counter()
    for file in label_files:
        with open(file) as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1

    for cid, count in class_counts.items():
        logger.report_scalar("Class Distribution", f"class_{cid}", count, 0)

    best_model_path = os.path.join("runs/detect/yolov8_cpu/weights/best.pt")
    task.upload_artifact("best.pt", best_model_path)

    print("Training complete. Model saved.")


@PipelineDecorator.pipeline(name='YOLOv8 Pipeline with Decorator', project='Vizai')
def full_pipeline():
    yaml_path = upload_dataset()
    train_model(yaml_path)


if __name__ == "__main__":
    PipelineDecorator.run_locally()  # Run controller locally instead of services queue
    full_pipeline()
