from pathlib import Path
from collections import Counter

PROJECT_ROOT=Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data"

SPLITS  =["train","valid","test"]
IMAGE_EXTENSIONS={".jpg",".jpeg"}

def read_labels(label_path):
    boxes = []

    with label_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"{label_path}, line {line_number}: expected 5 values"
                )

            class_id = int(parts[0])
            xc, yc, width, height = map(float, parts[1:])

            if class_id < 0:
                raise ValueError(
                    f"{label_path}, line {line_number}: negative class ID"
                )

            coords = (xc, yc, width, height)

            if not all(0 <= value <= 1 for value in coords):
                raise ValueError(
                    f"{label_path}, line {line_number}: "
                    "coordinates must be in [0, 1]"
                )

            if width <= 0 or height <= 0:
                raise ValueError(
                    f"{label_path}, line {line_number}: "
                    "box width and height must be positive"
                )

            boxes.append((class_id, xc, yc, width, height))

    return boxes
def validate_split(split):
    image_dir = DATASET_DIR / split / "images"
    label_dir = DATASET_DIR / split / "labels"

    print(f"\n{'=' * 45}")
    print(f"Checking split: {split}")
    print(f"{'=' * 45}")

    if not image_dir.is_dir() or not label_dir.is_dir():
        print(f"ERROR: Missing images or labels directory for {split}")
        return

    images = sorted(
        path for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    label_files = sorted(label_dir.glob("*.txt"))

    image_stems = {image.stem for image in images}
    label_stems = {label.stem for label in label_files}

    missing_labels = image_stems - label_stems
    missing_images = label_stems - image_stems

    print(f"Images: {len(images)}")
    print(f"Label files: {len(label_files)}")
    print(f"Images without label files: {len(missing_labels)}")
    print(f"Labels without images: {len(missing_images)}")

    class_counts = Counter()
    total_boxes = 0
    empty_labels = 0
    invalid_labels = 0

    for label_path in label_files:
        try:
            boxes = read_labels(label_path)

            if not boxes:
                empty_labels += 1

            for box in boxes:
                class_counts[box[0]] += 1
                total_boxes += 1

        except (ValueError, OSError) as error:
            invalid_labels += 1
            print(f"Invalid label: {error}")

    print(f"Total bounding boxes: {total_boxes}")
    print(f"Empty label files: {empty_labels}")
    print(f"Invalid label files: {invalid_labels}")
    print(f"Bounding boxes by class ID: {dict(class_counts)}")


def main():
    for split in SPLITS:
        validate_split(split)


if __name__ == "__main__":
    main()