from pathlib import Path

import torch
from torch.utils.data import DataLoader

from model import potholecnn
from dataset import PotholeDataset
from loss import DetectionLoss


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Train for one epoch, or evaluate without updating parameters."""
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_samples = 0
    component_totals = {}

    context = torch.enable_grad if training else torch.no_grad

    with context():
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            if training:
                optimizer.zero_grad(set_to_none=True)

            predictions = model(images)
            loss, components = criterion(predictions, targets)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss encountered: {loss.item()}"
                )

            if training:
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            for name, value in components.items():
                component_totals[name] = (
                    component_totals.get(name, 0.0)
                    + value.item() * batch_size
                )

    average_loss = total_loss / total_samples
    average_components = {
        name: value / total_samples
        for name, value in component_totals.items()
    }

    return average_loss, average_components


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)

    train_dataset = PotholeDataset(
        "data/train/images",
        "data/train/labels",
    )

    valid_dataset = PotholeDataset(
        "data/valid/images",
        "data/valid/labels",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    model = potholecnn().to(device)
    criterion = DetectionLoss().to(device)

    epochs = 30

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=1e-5,
    )

    best_valid_loss = float("inf")

    best_path = Path("pothole_detector.pth")
    last_path = Path("pothole_detector_last.pth")

    print("Training images:", len(train_dataset))
    print("Validation images:", len(valid_dataset))
    print("Epochs:", epochs)

    for epoch in range(epochs):
        train_loss, train_components = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
        )

        valid_loss, valid_components = run_epoch(
            model,
            valid_loader,
            criterion,
            device,
        )

        scheduler.step()

        print(
            f"\nEpoch [{epoch + 1}/{epochs}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Validation Loss: {valid_loss:.4f}"
        )

        print(
            "  Train components:",
            {
                k: round(v, 4)
                for k, v in train_components.items()
            },
        )

        print(
            "  Validation components:",
            {
                k: round(v, 4)
                for k, v in valid_components.items()
            },
        )

        # Preserve the most recent epoch separately.
        torch.save(model.state_dict(), last_path)

        # Keep the checkpoint with the lowest validation loss.
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), best_path)
            print("  Best checkpoint updated.")

    print("\nTraining complete.")
    print(f"Best validation loss: {best_valid_loss:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    main()