import torch

print("PyTorch version:", torch.version)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version used by PyTorch:", torch.version.cuda)
    # Perform a real computation on the GPU
    a = torch.randn(2000, 2000, device="cuda")
    b = torch.randn(2000, 2000, device="cuda")
    c = a @ b
    print("Result device:", c.device)
    print("Result shape:", c.shape)
    print("GPU test passed:", torch.isfinite(c).all().item())
else:
    print("GPU unavailable. Check the PyTorch installation and driver.")