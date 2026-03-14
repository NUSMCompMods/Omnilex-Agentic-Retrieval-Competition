#!/usr/bin/env python3
"""Upload a local GGUF model into the Modal volume used by Omnilex.

Usage:
    set -a && source .env && set +a
    python scripts/upload_modal_model.py models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
from pathlib import Path

import modal

from omnilex.llm.modal_backend import MODAL_MODEL_VOLUME_NAME, ensure_modal_credentials


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload a GGUF model to Modal Volume")
    parser.add_argument("model_path", help="Local path to a GGUF file")
    parser.add_argument(
        "--remote-path",
        default=None,
        help="Optional path inside the Modal volume. Defaults to the local filename.",
    )
    parser.add_argument(
        "--volume",
        default=MODAL_MODEL_VOLUME_NAME,
        help=f"Modal volume name (default: {MODAL_MODEL_VOLUME_NAME})",
    )
    args = parser.parse_args()

    ensure_modal_credentials()

    model_path = Path(args.model_path).expanduser().resolve()
    if not model_path.exists() or not model_path.is_file():
        raise FileNotFoundError(f"Local model file not found: {model_path}")

    remote_path = args.remote_path or model_path.name
    volume = modal.Volume.from_name(args.volume, create_if_missing=True)

    with volume.batch_upload(force=True) as batch:
        batch.put_file(model_path, remote_path)

    print(f"Uploaded {model_path} to Modal volume {args.volume}:{remote_path}")


if __name__ == "__main__":
    main()
