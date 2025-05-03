import os
import yaml
from ultralytics import YOLO
from clearml import Task, Dataset, Logger
from collections import Counter
import glob

print("Starting YOLOv8 training pipeline...")

def train_model():
    # Initialize ClearML task
    task = Task.init(
        project_name="Vizai",
        task_name="YOLOv8 Training Pipeline",
        task_type=Task.TaskTypes.training
    )

    # Get absolute path to the workspace
    workspace_path = os.path.abspath(".")
    dataset_path = workspace_path + "/data"

    # Register dataset with ClearML
    dataset = Dataset.create(
        dataset_name="Vizai Dataset",
        dataset_project="Vizai",
        dataset_tags=["YOLOv8", "training"]
    )
    dataset.add_files(path=dataset_path)
    dataset.upload()
    dataset.finalize()

    # Path to labels
    label_dir = os.path.join(dataset_path, "train/labels")

    print("Analyzing label files...")
    # Get all unique class IDs from label files
    class_ids = set()
    for label_file in os.listdir(label_dir):
        if label_file.endswith(".txt"):
            with open(os.path.join(label_dir, label_file), 'r') as f:
                for line in f.readlines():
                    if line.strip():  # Skip empty lines
                        try:
                            class_id = int(line.split()[0])  # First value is class_id
                            class_ids.add(class_id)
                        except (IndexError, ValueError):
                            continue

    # Sort class IDs and assign names 
    class_names = ['barrier', 'bicycle', 'car', 'crosswalk', 'dog', 'person', 'pole', 'shutters', 'signboard', 'trash_can', 'tree', 'truck']

    # Create data.yaml
    data_yaml = {
        "path": dataset_path,
        "train": os.path.join(dataset_path, "train/images"),
        "val": os.path.join(dataset_path, "valid/images"),
        "names": class_names
    }

    # Save to file
    output_path = os.path.join(dataset_path, "data.yaml")
    with open(output_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    print(f"Generated data.yaml at {output_path} with {len(class_names)} classes:")
    print(class_names)

    # Log configuration to ClearML
    task.connect_configuration({
        "dataset_path": dataset_path,
        "class_names": class_names,
        "data_yaml": data_yaml
    })

    print("Loading YOLOv8 model...")
    # Load model (YOLOv8 Nano is lightest for CPU)
    model = YOLO("yolov8n.pt")

    # Log model architecture
    task.connect(model)

    print("Starting training...")
    # Train on CPU
    results = model.train(
        data=output_path,  # Your dataset YAML
        epochs=50,        # Changed back to 50 epochs
        batch=8,              # Reduce batch size (CPU memory is limited)
        imgsz=320,            # Smaller image size = faster on CPU
        device="cpu",         # Force CPU training
        #device=0,
        name="yolov8_cpu",    # Save results to 'runs/detect/yolov8_cpu'
        project="Vizai",       # Project name for ClearML
        augment=True
    )

    print("Training completed. Starting validation...")
    # Evaluate the model
    metrics = model.val(
        data=output_path,
        batch=8,          
        imgsz=320,        
        device='cpu',     
        split='val'       
    )

    # Log validation metrics to ClearML
    task.get_logger().report_scalar(
        title="Validation Metrics",
        series="mAP50",
        value=metrics.box.map50,
        iteration=0
    )
    task.get_logger().report_scalar(
        title="Validation Metrics",
        series="mAP50-95",
        value=metrics.box.map,
        iteration=0
    )

    print("Validation metrics:", metrics)

    print("Analyzing class distribution...")
    # Count instances per class in training set

    label_files = glob.glob(os.path.join(dataset_path, "train/labels/*.txt"))
    class_counts = Counter()

    for file in label_files:
        with open(file) as f:
            for line in f:
                class_id = int(line.split()[0])
                class_counts[class_id] += 1

    # Log class distribution to ClearML
    for class_id, count in class_counts.items():
        task.get_logger().report_scalar(
            title="Class Distribution",
            series=f"class_{class_id}",
            value=count,
            iteration=0
        )

    print("Training set class distribution:", class_counts)
    print("Training pipeline completed!")

    # Upload model to ClearML
    task.upload_artifact(
        name="best.pt",
        artifact_object=os.path.join("runs/detect/yolov8_cpu/weights/best.pt")
    )

    # Close the task
    task.close()
    pass

if __name__ == "__main__":
    train_model()

