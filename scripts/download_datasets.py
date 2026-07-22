"""
AVIS — Dataset Download Script

Downloads FER-2013 from Kaggle and organizes it into the expected folder structure:
    data/fer2013/train/{angry,disgust,fear,happy,neutral,sad,surprise}/
    data/fer2013/test/{angry,disgust,fear,happy,neutral,sad,surprise}/

Prerequisites:
    pip install kaggle
    Set KAGGLE_USERNAME and KAGGLE_KEY environment variables
    OR place ~/.kaggle/kaggle.json

Run:
    python scripts/download_datasets.py
"""
from __future__ import annotations

import os
import sys
import shutil
import zipfile
from pathlib import Path

# Add project root to sys.path to allow importing src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.utils.logger import logger

DATA_DIR = Path("data")
FER_DIR = DATA_DIR / "fer2013"

# FER-2013 class names (standard Kaggle version folder names)
FER_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def download_fer2013() -> None:
    """Download FER-2013 from Kaggle and organize into class folders."""
    if FER_DIR.exists() and any(FER_DIR.iterdir()):
        logger.info(f"FER-2013 already exists at {FER_DIR}. Skipping download.")
        return

    DATA_DIR.mkdir(exist_ok=True)

    try:
        import kaggle  # noqa: requires kaggle credentials
    except ImportError:
        logger.error("kaggle package not installed. Run: pip install kaggle")
        logger.error("Also set KAGGLE_USERNAME and KAGGLE_KEY env variables.")
        return

    logger.info("Downloading FER-2013 from Kaggle …")
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "pankaj4321/fer-2013-facial-expression-dataset",
        path=str(DATA_DIR),
        unzip=True,
        quiet=False,
    )

    # The Kaggle FER-2013 dataset already has train/test split with class folders
    # But some versions download as a flat CSV — handle both
    csv_path = DATA_DIR / "fer2013.csv"
    if csv_path.exists():
        logger.info("CSV version detected — converting to image folders …")
        _convert_csv_to_folders(csv_path)
    else:
        logger.info("Folder version detected — verifying structure …")
        _verify_folder_structure()

    logger.info(f"FER-2013 ready at {FER_DIR}")


def _convert_csv_to_folders(csv_path: Path) -> None:
    """Convert the CSV version of FER-2013 to class image folders."""
    import pandas as pd
    import numpy as np
    from PIL import Image

    LABEL_MAP = {
        0: "angry", 1: "disgust", 2: "fear", 3: "happy",
        4: "neutral", 5: "sad", 6: "surprise"
    }

    df = pd.read_csv(csv_path)
    counters = {split: {cls: 0 for cls in LABEL_MAP.values()} for split in ["train", "test"]}

    for _, row in df.iterrows():
        label = LABEL_MAP[int(row["emotion"])]
        split = "train" if row["Usage"] == "Training" else "test"
        pixels = np.array(row["pixels"].split(), dtype=np.uint8).reshape(48, 48)
        img = Image.fromarray(pixels).convert("RGB").resize((96, 96))

        out_dir = FER_DIR / split / label
        out_dir.mkdir(parents=True, exist_ok=True)
        counters[split][label] += 1
        img.save(out_dir / f"{counters[split][label]:05d}.jpg")

    for split, cls_counts in counters.items():
        total = sum(cls_counts.values())
        logger.info(f"  {split}: {total} images across {len(cls_counts)} classes")


def _verify_folder_structure() -> None:
    for split in ["train", "test"]:
        split_dir = FER_DIR / split
        if not split_dir.exists():
            # Try to find the downloaded folder and rename
            for p in DATA_DIR.iterdir():
                if p.is_dir() and "train" in p.name.lower():
                    p.rename(FER_DIR / "train")
                    break
        count = sum(1 for _ in split_dir.rglob("*.jpg")) if split_dir.exists() else 0
        logger.info(f"  {split}: {count} images found")


if __name__ == "__main__":
    download_fer2013()
    logger.info("Dataset download complete.")
