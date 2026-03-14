"""Minimal Modal smoke test for requesting an L40S GPU.

Usage:
    set -a && source .env && set +a
    pip install modal
    modal token info
    modal run scripts/modal_gpu_check.py
"""

import json

import modal


app = modal.App("omnilex-modal-gpu-check")

image = modal.Image.debian_slim(python_version="3.10").pip_install("torch")


@app.function(gpu="L40S", image=image, timeout=600)
def gpu_info() -> dict[str, object]:
    import subprocess

    import torch

    nvidia_smi = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()

    return {
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_capability": (
            list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
        ),
        "nvidia_smi": nvidia_smi,
    }


@app.local_entrypoint()
def main() -> None:
    print(json.dumps(gpu_info.remote(), indent=2))
