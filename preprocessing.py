import shutil
import random
from glob import glob
import os

# Paths for unsplit data
workspace_path = os.path.abspath(".")
dataset_path = workspace_path + "/data"
unsplit_images_dir = os.path.join(dataset_path, "images")
unsplit_labels_dir = os.path.join(dataset_path, "labels")

# Paths for split data
split_dirs = {
    "train": {
        "images": os.path.join(dataset_path, "train/images"),
        "labels": os.path.join(dataset_path, "train/labels"),
    },
    "valid": {
        "images": os.path.join(dataset_path, "valid/images"),
        "labels": os.path.join(dataset_path, "valid/labels"),
    }
}

# Create split directories if they don't exist
for split in split_dirs.values():
    os.makedirs(split["images"], exist_ok=True)
    os.makedirs(split["labels"], exist_ok=True)

# List all images
image_paths = glob(os.path.join(unsplit_images_dir, "*.jpg")) + glob(os.path.join(unsplit_images_dir, "*.png"))

# Shuffle and split
random.seed(42)
random.shuffle(image_paths)
split_idx = int(0.8 * len(image_paths))  # 80% train, 20% valid
train_images = image_paths[:split_idx]
valid_images = image_paths[split_idx:]

splits = [("train", train_images), ("valid", valid_images)]

for split_name, images in splits:
    for img_path in images:
        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(unsplit_labels_dir, base + ".txt")
        # Copy image
        shutil.copy(img_path, split_dirs[split_name]["images"])
        # Copy label if exists
        if os.path.exists(label_path):
            shutil.copy(label_path, split_dirs[split_name]["labels"])

print("Dataset split complete: {} train, {} valid".format(len(train_images), len(valid_images)))