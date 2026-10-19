from pathlib import Path
import cv2

PROJECT_ROOT=Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "data"

IMAGE_DIR = DATASET_DIR / "train" / "images"
LABEL_DIR = DATASET_DIR / "train" / "labels"
OUTPUT_PATH=DATASET_DIR /"preview_train.jpg"
IMAGE_EXTENSIONS={".jpg",".jpeg"}

def read_labels(label_path):
    boxes = []
    label_path = Path(label_path)
    if not label_path.exists():
        return boxes

    with label_path.open("r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:
                class_id = int(parts[0])
                coords = [float(x) for x in parts[1:]]
                boxes.append((class_id, *coords))
    return boxes

def draw_boxes(image, boxes):
    """Convert normalized coordinates to pixels and draw rectangles."""
    image_height, image_width = image.shape[:2]

    for class_id, xc, yc, bw, bh in boxes:
        # Normalized center/size -> pixel corner coordinates
        x1 = round((xc - bw / 2) * image_width)
        y1 = round((yc - bh / 2) * image_height)
        x2 = round((xc + bw / 2) * image_width)
        y2 = round((yc + bh / 2) * image_height)

        # Clip the box to the image boundaries
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Show the numeric class ID until we verify the class mapping
        cv2.putText(
            image,
            f"class {class_id}",
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    return image


def main():
    if not IMAGE_DIR.is_dir():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    if not LABEL_DIR.is_dir():
        raise FileNotFoundError(f"Label directory not found: {LABEL_DIR}")

    images = sorted(
        path for path in IMAGE_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise FileNotFoundError(f"No images found in {IMAGE_DIR}")

    # Find an image that has a corresponding, non-empty label file
    selected_image = None
    selected_boxes = None

    for image_path in images:
        label_path = LABEL_DIR / f"{image_path.stem}.txt"
        boxes = read_labels(label_path)

        if boxes:
            selected_image = image_path
            selected_boxes = boxes
            break

    if selected_image is None:
        raise RuntimeError(
            "No image with a non-empty matching label file was found."
        )

    image = cv2.imread(str(selected_image))

    if image is None:
        raise RuntimeError(f"Could not read image: {selected_image}")

    print(f"Image: {selected_image.name}")
    print(f"Image dimensions: {image.shape[1]} x {image.shape[0]}")
    print(f"Number of boxes: {len(selected_boxes)}")

    for box in selected_boxes:
        print("Label:", box)

    annotated_image = draw_boxes(image.copy(), selected_boxes)

    if not cv2.imwrite(str(OUTPUT_PATH), annotated_image):
        raise RuntimeError(f"Could not save preview to {OUTPUT_PATH}")

    print(f"Saved annotated image to: {OUTPUT_PATH}")

    cv2.imshow("Ground-truth pothole annotations", annotated_image)
    print("Press any key while the image window is active to close it.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
