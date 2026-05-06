import torch

print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    print("Allocated:", torch.cuda.memory_allocated() / 1024**2, "MB")
    print("Reserved:", torch.cuda.memory_reserved() / 1024**2, "MB")
else:
    print("NO GPU — PyTorch is CPU-only")

x = torch.randn((1024,1024), dtype=torch.float16, device="cuda")
print("FP16 works on GPU")

