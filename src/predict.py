from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from model import potholecnn


IMAGE_SIZE = 128
GRID_SIZE = 32
BOXES_PER_CELL = 3
NUM_CLASSES = 3

CONF_THRESHOLD = 0.75
NMS_THRESHOLD = 0.25
IOM_THRESHOLD = 0.35

CLASS_NAMES = [
    "crocodile crack",
    "longitudinal crack",
    "pothole",
]


def box_metrics(box1, box2):
    """Return (IoU, IoM) for boxes in normalized xyxy format."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    area1 = max(0.0, box1[2] - box1[0]) * max(
        0.0, box1[3] - box1[1]
    )
    area2 = max(0.0, box2[2] - box2[0]) * max(
        0.0, box2[3] - box2[1]
    )

    union = area1 + area2 - intersection
    min_area = min(area1, area2)

    iou = intersection / union if union > 0 else 0.0
    iom = intersection / min_area if min_area > 0 else 0.0

    return iou, iom


def is_duplicate(best_box, candidate_box):
    """Check if candidate_box overlaps or is nested inside best_box."""
    iou, iom = box_metrics(best_box, candidate_box)
    return iou >= NMS_THRESHOLD or iom >= IOM_THRESHOLD


def nms(detections):
    """Greedy NMS with containment suppression for each predicted class."""
    kept = []

    for class_id in range(NUM_CLASSES):
        candidates = [
            det for det in detections
            if det["class_id"] == class_id
        ]
        candidates.sort(
            key=lambda det: det["score"],
            reverse=True,
        )

        while candidates:
            best = candidates.pop(0)
            kept.append(best)

            candidates = [
                det for det in candidates
                if not is_duplicate(best["box"], det["box"])
            ]

    return sorted(
        kept, key=lambda det: det["score"], reverse=True
    )


def predict_on_image(model, image_path, label_path, output_dir, device):
    """Run inference on a single image and save the annotated result."""
    orig_image = Image.open(image_path).convert("RGB")

    resized_image = orig_image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )

    array = np.asarray(resized_image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(
        array.transpose(2, 0, 1).copy()
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(tensor)[0]

    detections = []

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            for slot in range(BOXES_PER_CELL):
                raw = predictions[row, col, slot]

                objectness = torch.sigmoid(raw[4]).item()
                class_scores = torch.softmax(raw[5:], dim=0)
                class_id = int(class_scores.argmax().item())
                class_score = class_scores[class_id].item()

                score = objectness * class_score

                if score < CONF_THRESHOLD:
                    continue

                box = torch.sigmoid(raw[:4])

                cx = (col + box[0].item()) / GRID_SIZE
                cy = (row + box[1].item()) / GRID_SIZE
                w = box[2].item()
                h = box[3].item()

                x1 = max(0.0, cx - w / 2)
                y1 = max(0.0, cy - h / 2)
                x2 = min(1.0, cx + w / 2)
                y2 = min(1.0, cy + h / 2)

                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append({
                    "box": [x1, y1, x2, y2],
                    "class_id": class_id,
                    "score": score,
                })

    final_detections = nms(detections)

    draw = ImageDraw.Draw(orig_image)
    width, height = orig_image.size

    for det in final_detections:
        x1, y1, x2, y2 = det["box"]

        coords = [
            round(x1 * width),
            round(y1 * height),
            round(x2 * width),
            round(y2 * height),
        ]

        label = (
            f'{CLASS_NAMES[det["class_id"]]} '
            f'{det["score"]:.2f}'
        )

        draw.rectangle(coords, outline="red", width=3)
        draw.text(
            (coords[0], max(0, coords[1] - 14)),
            label,
            fill="red",
        )

    ground_truth_count = 0

    if label_path.exists():
        for line in label_path.read_text().splitlines():
            if not line.strip():
                continue

            values = [float(v) for v in line.split()]
            if len(values) != 5:
                continue

            class_id, cx, cy, w, h = values
            class_id = int(class_id)

            x1 = (cx - w / 2) * width
            y1 = (cy - h / 2) * height
            x2 = (cx + w / 2) * width
            y2 = (cy + h / 2) * height

            coords = [
                round(x1), round(y1), round(x2), round(y2)
            ]

            draw.rectangle(coords, outline="lime", width=3)
            draw.text(
                (coords[0], min(height - 14, coords[3] + 2)),
                f"GT: {CLASS_NAMES[class_id]}",
                fill="lime",
            )
            ground_truth_count += 1

    output_path = output_dir / image_path.name
    orig_image.save(output_path)

    return len(detections), final_detections, ground_truth_count


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = potholecnn().to(device)

    model.load_state_dict(
        torch.load(
            "pothole_detector.pth",
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    image_dir = Path("data/test/images")
    label_dir = Path("data/test/labels")
    output_dir = Path("prediction_results")
    output_dir.mkdir(exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in extensions
    )

    if not images:
        raise RuntimeError(f"No images found in {image_dir}")

    print(f"Device: {device}")
    print(f"Found {len(images)} test images. Running predictions...\n")

    total_gt = 0
    total_pred = 0

    for idx, image_path in enumerate(images, start=1):
        label_path = label_dir / f"{image_path.stem}.txt"

        raw_count, final_detections, gt_count = predict_on_image(
            model, image_path, label_path, output_dir, device
        )

        total_gt += gt_count
        total_pred += len(final_detections)

        print(
            f"[{idx}/{len(images)}] {image_path.name} -> "
            f"Preds: {len(final_detections)} (from {raw_count} raw) | "
            f"GT: {gt_count}"
        )

        for det in final_detections:
            print(
                f'    - {CLASS_NAMES[det["class_id"]]} '
                f'score={det["score"]:.3f}, '
                f'box={[round(v, 3) for v in det["box"]]}'
            )

    print("\n" + "=" * 50)
    print("Batch Prediction Complete!")
    print(f"Total Images Processed : {len(images)}")
    print(f"Total Ground-Truth Boxes: {total_gt}")
    print(f"Total Predicted Boxes   : {total_pred}")
    print(f"All images saved to     : {output_dir.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main() 