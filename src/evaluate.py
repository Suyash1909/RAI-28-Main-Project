
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from model import potholecnn


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_SIZE = 128
GRID_SIZE = 32
BOXES_PER_CELL = 3
NUM_CLASSES = 3

CONF_THRESHOLDS = [0.25, 0.50, 0.75]
NMS_THRESHOLD = 0.45
IOU_THRESHOLD = 0.50

MAX_CANDIDATES = 300
MAX_DETECTIONS = 100

CLASS_NAMES = [
    "crocodile crack",
    "longitudinal crack",
    "pothole",
]

CHECKPOINT = Path("pothole_detector.pth")
IMAGE_DIR = Path("data/valid/images")
LABEL_DIR = Path("data/valid/labels")


# --------------------------------------------------
# Bounding-box utilities
# --------------------------------------------------

def box_iou(box1, box2):
    """IoU between normalized xyxy boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = (
        max(0.0, x2 - x1) *
        max(0.0, y2 - y1)
    )

    area1 = (
        max(0.0, box1[2] - box1[0]) *
        max(0.0, box1[3] - box1[1])
    )
    area2 = (
        max(0.0, box2[2] - box2[0]) *
        max(0.0, box2[3] - box2[1])
    )

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0


def nms(detections):
    """Greedy NMS separately for each class."""
    kept = []

    for class_id in range(NUM_CLASSES):
        candidates = [
            d for d in detections
            if d["class_id"] == class_id
        ]

        candidates.sort(
            key=lambda d: d["score"],
            reverse=True,
        )

        while candidates:
            best = candidates.pop(0)
            kept.append(best)

            candidates = [
                d for d in candidates
                if box_iou(best["box"], d["box"])
                < NMS_THRESHOLD
            ]

    kept.sort(key=lambda d: d["score"], reverse=True)
    return kept[:MAX_DETECTIONS]


# --------------------------------------------------
# Data loading
# --------------------------------------------------

def load_ground_truth(label_path):
    """Read YOLO labels and convert to normalized xyxy."""
    boxes = []

    if not label_path.exists():
        return boxes

    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue

        values = [float(v) for v in line.split()]

        if len(values) != 5:
            raise ValueError(
                f"Invalid label in {label_path}: {line}"
            )

        class_value, cx, cy, w, h = values
        class_id = int(class_value)

        if (
            class_value != class_id
            or not 0 <= class_id < NUM_CLASSES
            or not (0 <= cx <= 1 and 0 <= cy <= 1)
            or not (0 < w <= 1 and 0 < h <= 1)
        ):
            raise ValueError(
                f"Invalid annotation in {label_path}: {line}"
            )

        boxes.append({
            "class_id": class_id,
            "box": [
                max(0.0, cx - w / 2),
                max(0.0, cy - h / 2),
                min(1.0, cx + w / 2),
                min(1.0, cy + h / 2),
            ],
        })

    return boxes


# --------------------------------------------------
# Model inference
# --------------------------------------------------

def predict_image(model, image_path, device):
    """Decode model outputs into normalized xyxy boxes."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )

    array = np.asarray(image, dtype=np.float32) / 255.0

    tensor = torch.from_numpy(
        array.transpose(2, 0, 1).copy()
    ).unsqueeze(0).to(device)

    with torch.inference_mode():
        # Transfer outputs to CPU once to avoid repeated GPU syncs.
        output = model(tensor)[0].detach().cpu()

    detections = []

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            for slot in range(BOXES_PER_CELL):
                raw = output[row, col, slot]

                objectness = torch.sigmoid(raw[4]).item()

                # Training uses cross_entropy for class logits.
                class_probs = torch.softmax(raw[5:], dim=0)
                class_id = int(class_probs.argmax().item())
                class_score = class_probs[class_id].item()

                score = objectness * class_score

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
                    "class_id": class_id,
                    "score": score,
                    "box": [x1, y1, x2, y2],
                })

    return detections


# --------------------------------------------------
# One-to-one detection matching
# --------------------------------------------------

def match_detections(predictions, ground_truth):
    """
    Match predictions to ground truth by class and IoU.
    Each prediction and GT can be matched at most once.
    """
    result = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "ious": [],
        "per_class": {},
    }

    for class_id in range(NUM_CLASSES):
        preds = [
            p for p in predictions
            if p["class_id"] == class_id
        ]
        preds.sort(key=lambda p: p["score"], reverse=True)

        gts = [
            g for g in ground_truth
            if g["class_id"] == class_id
        ]

        used_gt = set()
        tp = 0
        fp = 0
        matched_ious = []

        for pred in preds:
            best_iou = 0.0
            best_gt_idx = None

            for gt_idx, gt in enumerate(gts):
                if gt_idx in used_gt:
                    continue

                iou = box_iou(pred["box"], gt["box"])

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if (
                best_gt_idx is not None
                and best_iou >= IOU_THRESHOLD
            ):
                used_gt.add(best_gt_idx)
                tp += 1
                matched_ious.append(best_iou)
            else:
                fp += 1

        fn = len(gts) - len(used_gt)

        result["tp"] += tp
        result["fp"] += fp
        result["fn"] += fn
        result["ious"].extend(matched_ious)

        result["per_class"][class_id] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gt": len(gts),
            "ious": matched_ious,
        }

    return result


def precision_recall(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device, flush=True)

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT.resolve()}"
        )

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(
            f"Validation image directory not found: {IMAGE_DIR}"
        )

    model = potholecnn().to(device)

    model.load_state_dict(
        torch.load(
            CHECKPOINT,
            map_location=device,
            weights_only=True,
        )
    )
    model.eval()

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    image_paths = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in extensions
    )

    if not image_paths:
        raise RuntimeError(
            f"No validation images found in {IMAGE_DIR}"
        )

    print("Validation images:", len(image_paths))
    print("IoU threshold:", IOU_THRESHOLD)
    print("NMS threshold:", NMS_THRESHOLD)
    print("Maximum candidates/image:", MAX_CANDIDATES)
    print("Maximum detections/image:", MAX_DETECTIONS)

    # Cache predictions once so every threshold uses the same outputs.
    cached = []
    total_gt = 0
    raw_score_sample = []

    for index, image_path in enumerate(image_paths, start=1):
        label_path = LABEL_DIR / f"{image_path.stem}.txt"

        ground_truth = load_ground_truth(label_path)
        raw_predictions = predict_image(
            model,
            image_path,
            device,
        )

        cached.append((raw_predictions, ground_truth))
        total_gt += len(ground_truth)

        # Collect a sample for score-distribution diagnostics.
        raw_score_sample.extend(
            p["score"] for p in raw_predictions
        )

        if index % 25 == 0 or index == len(image_paths):
            print(
                f"Processed {index}/{len(image_paths)} images",
                flush=True,
            )

    print("\nGround-truth objects:", total_gt)
    print("Raw valid-box predictions:", len(raw_score_sample))

    if raw_score_sample:
        print(
            "Raw score distribution: "
            f"median={np.percentile(raw_score_sample, 50):.3f}, "
            f"p90={np.percentile(raw_score_sample, 90):.3f}, "
            f"max={max(raw_score_sample):.3f}"
        )

    for threshold in CONF_THRESHOLDS:
        aggregate = {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "ious": [],
            "per_class": {
                c: {
                    "tp": 0,
                    "fp": 0,
                    "fn": 0,
                    "gt": 0,
                    "ious": [],
                }
                for c in range(NUM_CLASSES)
            },
        }

        candidates_before_nms = 0
        detections_after_nms = 0

        for raw_predictions, ground_truth in cached:
            filtered = [
                p for p in raw_predictions
                if p["score"] >= threshold
            ]

            # Diagnostic cap: keep highest-confidence candidates.
            filtered.sort(
                key=lambda p: p["score"],
                reverse=True,
            )
            filtered = filtered[:MAX_CANDIDATES]

            candidates_before_nms += len(filtered)

            final_predictions = nms(filtered)
            detections_after_nms += len(final_predictions)

            stats = match_detections(
                final_predictions,
                ground_truth,
            )

            for key in ("tp", "fp", "fn"):
                aggregate[key] += stats[key]

            aggregate["ious"].extend(stats["ious"])

            for class_id in range(NUM_CLASSES):
                class_stats = stats["per_class"][class_id]

                for key in ("tp", "fp", "fn", "gt"):
                    aggregate["per_class"][class_id][key] += (
                        class_stats[key]
                    )

                aggregate["per_class"][class_id]["ious"].extend(
                    class_stats["ious"]
                )

        precision, recall = precision_recall(
            aggregate["tp"],
            aggregate["fp"],
            aggregate["fn"],
        )

        mean_iou = (
            float(np.mean(aggregate["ious"]))
            if aggregate["ious"] else 0.0
        )

        print("\n" + "=" * 64)
        print(f"Confidence threshold: {threshold:.2f}")
        print("Candidates before NMS:", candidates_before_nms)
        print("Detections after NMS:", detections_after_nms)
        print(
            f"TP={aggregate['tp']} | "
            f"FP={aggregate['fp']} | "
            f"FN={aggregate['fn']}"
        )
        print(f"Precision @ IoU 0.50: {precision:.4f}")
        print(f"Recall    @ IoU 0.50: {recall:.4f}")
        print(f"Mean IoU of matched detections: {mean_iou:.4f}")

        for class_id, class_name in enumerate(CLASS_NAMES):
            stats = aggregate["per_class"][class_id]

            class_precision, class_recall = precision_recall(
                stats["tp"],
                stats["fp"],
                stats["fn"],
            )

            class_iou = (
                float(np.mean(stats["ious"]))
                if stats["ious"] else 0.0
            )

            print(
                f"  {class_name}: "
                f"GT={stats['gt']}, "
                f"TP={stats['tp']}, "
                f"FP={stats['fp']}, "
                f"FN={stats['fn']}, "
                f"P={class_precision:.4f}, "
                f"R={class_recall:.4f}, "
                f"IoU={class_iou:.4f}"
            )


if __name__ == "__main__":
    main()
