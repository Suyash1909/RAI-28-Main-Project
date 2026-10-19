
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


IMAGE_SIZE = 128
GRID_SIZE = 32  
BOXES_PER_CELL = 3
NUM_CLASSES = 3
VALUES_PER_CELL = 5 + NUM_CLASSES

CLASS_NAMES = [
    "crocodile crack",
    "longitudinal crack",
    "pothole",
]


class PotholeDataset(Dataset):
    def __init__(self, images_dir, labels_dir):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)

        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        self.image_paths = sorted(
            path for path in self.images_dir.iterdir()
            if path.suffix.lower() in extensions
        )

        if not self.image_paths:
            raise RuntimeError(
                f"No images found in {self.images_dir}"
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]

        # Load and resize the image.
        image = Image.open(image_path).convert("RGB")
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.BILINEAR
        )

        # Convert HWC uint8 image to CHW float tensor [0, 1].
        image_array = np.array(image, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(
            image_array.transpose(2, 0, 1).copy()
        )

        # Target shape: [grid row, grid column, 8 values].
        target = torch.zeros(
            GRID_SIZE,
            GRID_SIZE,
            BOXES_PER_CELL,
            VALUES_PER_CELL,
            dtype=torch.float32
        )

        label_path = self.labels_dir / f"{image_path.stem}.txt"

        if label_path.exists():
            lines = label_path.read_text().splitlines()

            for line in lines:
                if not line.strip():
                    continue

                values = [float(v) for v in line.split()]

                if len(values) != 5:
                    raise ValueError(
                        f"Invalid label in {label_path}: {line}"
                    )

                class_id_float, cx, cy, w, h = values
                class_id = int(class_id_float)

                if (
                    class_id_float != class_id
                    or not 0 <= class_id < NUM_CLASSES
                ):
                    raise ValueError(
                        f"Invalid class ID in {label_path}: {class_id}"
                    )

                if not (
                    0 <= cx <= 1
                    and 0 <= cy <= 1
                    and 0 < w <= 1
                    and 0 < h <= 1
                ):
                    raise ValueError(
                        f"Invalid box coordinates in {label_path}: {line}"
                    )

                # Find the grid cell containing the box centre.
                # Find the grid cell containing the box centre.
                col = min(int(cx * GRID_SIZE), GRID_SIZE - 1)
                row = min(int(cy * GRID_SIZE), GRID_SIZE - 1)
                
                # Find an available slot in this cell.
                cell_targets = target[row, col]
                free_slots = torch.where(cell_targets[:, 4] == 0)[0]
                
                if len(free_slots) == 0:
                    raise ValueError(
                        f"More than {BOXES_PER_CELL} objects occupy grid cell "
                        f"({row}, {col}) in {label_path}. "
                        "Increase BOXES_PER_CELL or use a different detector design."
                    )
                
                slot = free_slots[0].item()
                
                # Centre coordinates relative to the grid cell.
                x_cell = cx * GRID_SIZE - col
                y_cell = cy * GRID_SIZE - row
                
                target[row, col, slot, 0] = x_cell
                target[row, col, slot, 1] = y_cell
                target[row, col, slot, 2] = w
                target[row, col, slot, 3] = h
                target[row, col, slot, 4] = 1.0
                target[row, col, slot, 5 + class_id] = 1.0

        return image_tensor, target


if __name__ == "__main__":
    # Run this file from the project root.
    dataset = PotholeDataset(
        "data/train/images",
        "data/train/labels"
    )

    image, target = dataset[0]

    print("Dataset size:", len(dataset))
    print("Image shape:", image.shape)
    print("Target shape:", target.shape)
    print("Image range:", image.min().item(), image.max().item())
    print("Objects in first image:",
          int(target[:, :, 2].sum().item()))

    # Test batching multiple images.
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    images, targets = next(iter(loader))

    print("Batch image shape:", images.shape)
    print("Batch target shape:", targets.shape)
