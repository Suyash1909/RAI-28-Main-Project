
import torch

from loss import DetectionLoss


def main():
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    criterion = DetectionLoss().to(device)

    # Simulate predictions for a batch of two images.
    predictions = torch.randn(
        2, 32, 32, 3, 8,
        device=device,
        requires_grad=True,
    )

    targets = torch.zeros(
        2, 32, 32, 3, 8,
        device=device,
    )

    # Add one pothole to image 1, cell (10, 12), slot 0.
    targets[0, 10, 12, 0] = torch.tensor(
        [0.4, 0.6, 0.2, 0.3, 1, 0, 0, 1],
        device=device,
    )

    # Add one longitudinal crack to image 2.
    targets[1, 20, 18, 0] = torch.tensor(
        [0.2, 0.7, 0.15, 0.1, 1, 0, 1, 0],
        device=device,
    )

    loss, components = criterion(predictions, targets)

    print("Device:", device)
    print("Loss:", loss.item())
    print("Components:")

    for name, value in components.items():
        print(f"  {name}: {value.item():.4f}")

    loss.backward()

    print("Gradients computed:", predictions.grad is not None)
    print(
        "Gradient finite:",
        torch.isfinite(predictions.grad).all().item(),
    )


if __name__ == "__main__":
    main()
