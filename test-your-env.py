#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2025 Dennis H. Wuitz <dennis.wuitz@wavelens.io>
#
# SPDX-License-Identifier: MIT

import importlib
import sys


DEPENDENCIES = {
    "jupyter": "jupyter",
    "matplotlib": "matplotlib",
    "termcolor": "termcolor",
    "torch": "torch",
    "torchinfo": "torchinfo",
    "tqdm": "tqdm",
    "umap-learn": "umap",
}


def check_package(display_name, import_name):
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown")
        print(f"[OK] {display_name:<12} | version: {version}")
        return True
    except ImportError:
        print(f"[MISSING] {display_name:<12} | not installed")
        return False


def check_torch_cuda():
    try:
        import torch

        print("\nPyTorch CUDA check:")
        print(f"  torch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("  No CUDA-capable GPU detected.")
    except ImportError:
        print("\nPyTorch not installed; skipping CUDA check.")


def main():
    print("Testing Python environment\n" + "-" * 30)

    any_missing = False
    for display_name, import_name in DEPENDENCIES.items():
        ok = check_package(display_name, import_name)
        if not ok:
            any_missing = True

    check_torch_cuda()

    print("\nSummary:")
    if any_missing:
        print("  Some dependencies are missing.")
        sys.exit(1)
    else:
        print("  All dependencies are installed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
