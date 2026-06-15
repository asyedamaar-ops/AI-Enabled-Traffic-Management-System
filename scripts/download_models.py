"""
Model Downloader
================
Downloads pre-trained model weights into the models/ directory.

Usage:
    python scripts/download_models.py

Note: These are placeholder URLs. Replace with your actual hosted weights
or implement training scripts (see docs/training.md for guidance).
"""

import os
import sys
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_REGISTRY = {
    "ssd_vehicle.pb": {
        "description": "SSD MobileNet v2 fine-tuned on COCO vehicle classes",
        "size_mb": 67,
        "url": "https://your-storage.example.com/models/ssd_vehicle.pb",
    },
    "densenet169_ev.h5": {
        "description": "DenseNet-169 emergency vehicle classifier (10k image dataset)",
        "size_mb": 112,
        "url": "https://your-storage.example.com/models/densenet169_ev.h5",
    },
    "cnn_siren.h5": {
        "description": "CNN siren detector trained on MFCC + spectrogram features",
        "size_mb": 23,
        "url": "https://your-storage.example.com/models/cnn_siren.h5",
    },
}


def download_all():
    print("Model Downloader — AI Traffic Management System")
    print("=" * 50)

    for filename, meta in MODEL_REGISTRY.items():
        dest = MODELS_DIR / filename
        if dest.exists():
            print(f"  [SKIP]  {filename} already exists.")
            continue

        print(f"  [INFO]  {filename} (~{meta['size_mb']} MB)")
        print(f"          {meta['description']}")
        print(f"          URL: {meta['url']}")
        print()

        # Uncomment to enable actual download:
        # import urllib.request
        # print(f"  Downloading...")
        # urllib.request.urlretrieve(meta["url"], dest)
        # print(f"  Saved to {dest}")

    print()
    print("Place model files in the models/ directory to enable full inference.")
    print("Without weights, the system runs in mock mode (random scores).")


if __name__ == "__main__":
    download_all()
