import sys
import torch

print(f"Python Version: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")

    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    print(f"Total VRAM: {vram_gb:.2f} GB")
    print(f"bfloat16 Supported: {torch.cuda.is_bf16_supported()}")
else:
    print("⚠️ CẢNH BÁO: PyTorch chưa nhận GPU (đang chạy trên CPU)!")
    