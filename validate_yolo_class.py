import os
import glob
from collections import Counter

# Configuration
NUM_CLASSES = 5
LABEL_DIRS = [
    "clean_combined_dataset/train/labels",
    "clean_combined_dataset/test/labels"
]
AUTO_FIX = False  # Set to True to auto-remove invalid lines

def check_label_files():
    print(f"🔍 Scanning label files with NUM_CLASSES={NUM_CLASSES}...\n")

    total_valid_labels = 0
    class_counter = Counter()
    invalid_class_counter = Counter()
    errors_found = False

    for label_dir in LABEL_DIRS:
        label_files = glob.glob(os.path.join(label_dir, "*.txt"))
        print(f"📂 Scanning directory: {label_dir} — Found {len(label_files)} .txt files")

        for file_path in label_files:
            with open(file_path, "r") as f:
                lines = f.readlines()

            valid_lines = []
            invalid_lines = []

            for i, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        raise ValueError(f"Invalid format: {line.strip()}")

                    class_id = int(parts[0])
                    print(f"  ↪️ Line {i+1} in {file_path}: class_id = {class_id}")

                    if 0 <= class_id < NUM_CLASSES:
                        valid_lines.append(line)
                        class_counter[class_id] += 1
                        total_valid_labels += 1
                    else:
                        invalid_lines.append((i + 1, class_id, line.strip()))
                        invalid_class_counter[class_id] += 1
                except Exception as e:
                    invalid_lines.append((i + 1, "parse_error", line.strip()))
                    print(f"⚠️  Parse error in {file_path} line {i+1}: {e}")

            if invalid_lines:
                errors_found = True
                print(f"\n❌ Invalid entries in: {file_path}")
                for line_num, class_id, content in invalid_lines:
                    print(f"  Line {line_num}: class_id={class_id} ➜ \"{content}\"")

                if AUTO_FIX and valid_lines:
                    with open(file_path, "w") as f:
                        f.writelines(valid_lines)
                    print(f"  ✅ Fixed: Kept {len(valid_lines)} valid lines, removed {len(invalid_lines)} bad ones.")

    # Summary
    print("\n📈 Label Summary")
    print(f"➡️  Total valid labels: {total_valid_labels}")
    print("➡️  Class frequency:")
    for class_id in range(NUM_CLASSES):
        count = class_counter[class_id]
        print(f"  Class {class_id:2}: {count}")

    if invalid_class_counter:
        print("\n⚠️  Invalid entries:")
        for class_id, count in invalid_class_counter.items():
            print(f"  Invalid class {class_id}: {count} occurrence(s)")
    else:
        print("✅ All labels are within valid class range.")

    if not errors_found:
        print("\n🎉 No issues found in any label file!")

if __name__ == "__main__":
    check_label_files()
