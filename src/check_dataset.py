
from pathlib import Path

from dataset import PotholeDataset, GRID_SIZE, BOXES_PER_CELL

dataset = PotholeDataset(
    "data/train/images",
    "data/train/labels",
)

total_objects = 0
total_cells_with_objects = 0
max_objects_in_cell = 0
collision_cells = 0
images_with_collisions = 0

for index in range(len(dataset)):
    _, target = dataset[index]

    # Objectness is value 4 for each of the 3 slots.
    occupancy = target[..., 4].sum(dim=-1)

    image_objects = int(occupancy.sum().item())
    total_objects += image_objects

    occupied = occupancy > 0
    total_cells_with_objects += int(occupied.sum().item())

    image_max = int(occupancy.max().item())
    max_objects_in_cell = max(max_objects_in_cell, image_max)

    image_collisions = int((occupancy > 1).sum().item())
    collision_cells += image_collisions

    if image_collisions:
        images_with_collisions += 1

print("Images:", len(dataset))
print("Objects encoded:", total_objects)
print("Occupied grid cells:", total_cells_with_objects)
print("Maximum objects in one cell:", max_objects_in_cell)
print("Cells containing multiple objects:", collision_cells)
print("Images with collisions:", images_with_collisions)
print("Available slots per cell:", BOXES_PER_CELL)
