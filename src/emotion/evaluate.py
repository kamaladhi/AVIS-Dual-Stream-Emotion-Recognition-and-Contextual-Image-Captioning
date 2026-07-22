"""
AVIS — Emotion Model Evaluation
Generates accuracy, weighted F1, per-class report, and confusion matrix.

Run:
    python -m src.emotion.evaluate --data data/fer2013
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from src.emotion.train import FER2013Dataset
from src.emotion.model import build_model
from src.utils.config import config
from src.utils.logger import logger


def evaluate_model(data_root: str = "data/fer2013", model_path: Path | None = None) -> dict:
    """
    Full evaluation: accuracy, weighted F1, per-class report, confusion matrix plot.

    Returns:
        dict with keys: accuracy, weighted_f1, report (str), cm (np.ndarray)
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    model_path = model_path or config.emotion_model_path

    # ── Load model ────────────────────────────────────────────
    model = build_model(freeze_backbone=False).to(device)
    if model_path.exists():
        ckpt = torch.load(str(model_path), map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        logger.info(f"Loaded weights from {model_path}")
    model.eval()

    # ── Dataset ───────────────────────────────────────────────
    test_ds = FER2013Dataset(Path(data_root), "test", use_landmarks=True)
    test_loader = DataLoader(
        test_ds, batch_size=64, shuffle=False, num_workers=config.num_workers
    )

    # ── Inference ─────────────────────────────────────────────
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            landmarks = batch["landmarks"].to(device)
            labels = batch["label"]
            out = model(images, landmarks)
            preds = out["logits"].argmax(dim=1).cpu()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Metrics ───────────────────────────────────────────────
    accuracy = (all_preds == all_labels).mean()
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
    report = classification_report(
        all_labels, all_preds, target_names=config.emotion_classes
    )
    cm = confusion_matrix(all_labels, all_preds)

    logger.info(f"\nAccuracy: {accuracy:.4f} | Weighted F1: {weighted_f1:.4f}\n{report}")

    # ── Confusion matrix plot ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.emotion_classes,
        yticklabels=config.emotion_classes,
        ax=ax,
    )
    ax.set_title("Emotion Recognition — Confusion Matrix", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    out_path = config.results_dir / "confusion_matrix.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info(f"Confusion matrix saved to {out_path}")

    return {
        "accuracy": float(accuracy),
        "weighted_f1": float(weighted_f1),
        "report": report,
        "cm": cm,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fer2013")
    args = parser.parse_args()
    evaluate_model(args.data)
