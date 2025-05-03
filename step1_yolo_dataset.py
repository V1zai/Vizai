# step1_yolo_dataset.py
import os
import yaml
from clearml import Task, Dataset

# Step 1: Dataset preparation and upload
def upload_yolo_dataset():
    task = Task.init(project_name="Vizai", task_name="Pipeline Step 1 - YOLOv8 Dataset Upload")

    # Get workspace path and dataset location
    workspace_path = os.path.abspath(".")
    dataset_path = os.path.join(workspace_path, "data")

    # Register and upload dataset
    dataset = Dataset.create(
        dataset_name="Vizai Dataset",
        dataset_project="Vizai",
        dataset_tags=["YOLOv8", "training"]
    )
    dataset.add_files(path=dataset_path)
    dataset.upload()
    dataset.finalize()

    # Path to label files
    label_dir = os.path.join(dataset_path, "train", "labels")

    # Scan for class IDs
    class_ids = set()
    for file in os.listdir(label_dir):
        if file.endswith(".txt"):
            with open(os.path.join(label_dir, file)) as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_ids.add(int(parts[0]))

    # Define class names
    class_names = ['barrier', 'bicycle', 'car', 'crosswalk', 'dog', 'person',
                'pole', 'shutters', 'signboard', 'trash_can', 'tree', 'truck']

    # Save data.yaml
    data_yaml = {
        "path": dataset_path,
        "train": os.path.join(dataset_path, "train/images"),
        "val": os.path.join(dataset_path, "test/images"),
        "names": class_names
    }
    yaml_path = os.path.join(dataset_path, "data.yaml")
    with open(yaml_path, 'w') as f:
        yaml.dump(data_yaml, f, sort_keys=False)

    task.upload_artifact(name="data_yaml", artifact_object=yaml_path)
    task.upload_artifact(name="class_names", artifact_object=class_names)

    print(f"Data.yaml saved at {yaml_path}")
    print("Dataset upload step completed.")

if __name__ == "__main__":
    upload_yolo_dataset()