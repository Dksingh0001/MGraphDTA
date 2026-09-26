"""
Environment and Hardware Verification Script for MGraphDTA
Checks Python version, PyTorch, CUDA, GPU name/VRAM, PyG, and RDKit.
"""

import sys
import platform
import subprocess

def check_env():
    print("=" * 60)
    print("MGraphDTA Environment & Hardware Verification")
    print("=" * 60)

    # 1. Python version
    py_version = sys.version.replace("\n", " ")
    print(f"Python version: {platform.python_version()} ({sys.executable})")
    print(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    print("-" * 60)

    # 2. Check NVIDIA SMI directly for hardware check
    print("NVIDIA System Management Interface (nvidia-smi):")
    try:
        smi_out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
            encoding="utf-8"
        ).strip()
        print(f"Detected GPU(s): {smi_out}")
    except Exception as e:
        print(f"nvidia-smi not available or failed: {e}")
    print("-" * 60)

    # 3. Check PyTorch
    torch_installed = False
    try:
        import torch
        torch_installed = True
        print(f"PyTorch version: {torch.__version__}")
        cuda_avail = torch.cuda.is_available()
        print(f"CUDA available: {cuda_avail}")
        if cuda_avail:
            print(f"PyTorch built with CUDA version: {torch.version.cuda}")
            print(f"GPU device count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                dev_name = torch.cuda.get_device_name(i)
                total_mem = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)
                print(f"  GPU {i}: {dev_name} ({total_mem:.2f} GB VRAM)")
        else:
            print("CUDA is NOT accessible by the installed PyTorch build.")
    except ImportError:
        print("PyTorch: NOT INSTALLED")
    print("-" * 60)

    # 4. Check PyTorch Geometric
    try:
        import torch_geometric
        print(f"PyTorch Geometric version: {torch_geometric.__version__}")
    except ImportError:
        print("PyTorch Geometric: NOT INSTALLED")
    print("-" * 60)

    # 5. Check RDKit
    try:
        import rdkit
        print(f"RDKit version: {rdkit.__version__}")
    except ImportError:
        print("RDKit: NOT INSTALLED")
    print("-" * 60)

    # 6. Check Other Supporting Libraries
    for lib in ["numpy", "pandas", "scipy", "sklearn", "networkx", "matplotlib"]:
        try:
            mod = __import__(lib)
            ver = getattr(mod, "__version__", "installed")
            print(f"{lib}: {ver}")
        except ImportError:
            print(f"{lib}: NOT INSTALLED")
    print("=" * 60)

if __name__ == "__main__":
    check_env()
