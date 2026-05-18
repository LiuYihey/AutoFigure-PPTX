import sys, platform, shutil, subprocess, os

print("python:", sys.executable)
print("version:", sys.version.split()[0], platform.platform())

try:
    import torch
    print("torch:", torch.__version__)
    print("torch.version.cuda:", torch.version.cuda)
    print("torch.backends.cudnn.version():", torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    print("torch.cuda.device_count():", torch.cuda.device_count())
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  device[{i}]: {p.name}  cc={p.major}.{p.minor}  vram={p.total_memory/1024**3:.2f}GiB")
except Exception as e:
    print("torch import failed:", e)

try:
    import torchvision
    print("torchvision:", torchvision.__version__)
except Exception as e:
    print("torchvision import failed:", e)

nvsmi = shutil.which("nvidia-smi")
print("nvidia-smi path:", nvsmi)
if nvsmi:
    try:
        out = subprocess.run([nvsmi, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=15)
        print("nvidia-smi stdout:", out.stdout.strip())
        if out.stderr.strip():
            print("nvidia-smi stderr:", out.stderr.strip())
    except Exception as e:
        print("nvidia-smi failed:", e)

print("CUDA_PATH:", os.environ.get("CUDA_PATH"))
print("CUDA_HOME:", os.environ.get("CUDA_HOME"))
