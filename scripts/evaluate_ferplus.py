"""
AVIS — FER+ Benchmark Evaluation Script

Evaluates the trained dual-stream emotion recognition model against Microsoft's
clean FER+ crowd-sourced ground truth annotations.

Computes:
1. Accuracy on original FER-2013 test set.
2. Accuracy on clean FER+ majority-vote test set.
3. Detailed per-class precision, recall, and F1-score.
4. Comparative Confusion Matrix saved to results/ferplus_confusion_matrix.png.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.emotion.model import build_model
from src.utils.config import config
from src.utils.logger import logger

# Mapping from FER+ column names to our project's 7 classes
FERPLUS_COL_TO_CLASS = {
    "neutral": "neutral",
    "happiness": "happy",
    "surprise": "surprise",
    "sadness": "sad",
    "anger": "angry",
    "disgust": "disgust",
    "fear": "fear",
}

PROJECT_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
CLASS_TO_IDX = {c: i for i, c in enumerate(PROJECT_CLASSES)}


class FERPlusTestDataset(Dataset):
    """
    Dataset loader matching FER-2013 test images directly by row index with FER+ majority vote labels.
    """

    EVAL_TRANSFORM = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    IMG_SIZE = 224

    def __init__(
        self,
        fer2013_root: Path = Path("data/fer2013"),
        ferplus_csv: Path = Path("data/fer2013new.csv"),
        min_votes_threshold: int = 1,
    ):
        self.fer2013_root = fer2013_root
        self.min_votes_threshold = min_votes_threshold
        self.landmark_extractor = None

        df = pd.read_csv(ferplus_csv)
        emotion_cols = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]

        self.samples = []
        test_files = list((fer2013_root / "test").rglob("*.png")) + list((fer2013_root / "test").rglob("*.jpg"))

        for p in test_files:
            try:
                row_idx = int(p.stem)
            except ValueError:
                continue

            if row_idx >= len(df):
                continue

            orig_class = p.parent.name.lower()
            if orig_class not in CLASS_TO_IDX:
                continue

            row = df.iloc[row_idx]

            # Extract crowd votes
            votes = {col: int(row[col]) for col in emotion_cols}
            unknown_votes = int(row.get("unknown", 0))
            nf_votes = int(row.get("NF", 0))

            max_emotion = max(votes, key=votes.get)
            max_votes = votes[max_emotion]

            # Filter out non-faces (NF) and completely ambiguous/unknown samples
            if max_votes < self.min_votes_threshold or nf_votes >= max_votes or unknown_votes >= max_votes:
                continue

            clean_class = FERPLUS_COL_TO_CLASS[max_emotion]
            clean_idx = CLASS_TO_IDX[clean_class]
            orig_idx = CLASS_TO_IDX[orig_class]

            self.samples.append({
                "path": p,
                "clean_label": clean_idx,
                "orig_label": orig_idx,
                "clean_class": clean_class,
                "orig_class": orig_class,
                "votes": votes,
            })

        logger.info(f"Loaded {len(self.samples)} valid samples for FER+ benchmarking (after filtering ambiguous/NF).")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        item = self.samples[idx]
        bgr = cv2.imread(str(item["path"]))
        if bgr is None:
            bgr = np.zeros((self.IMG_SIZE, self.IMG_SIZE, 3), dtype=np.uint8)
        else:
            bgr = cv2.resize(bgr, (self.IMG_SIZE, self.IMG_SIZE))

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        augmented = self.EVAL_TRANSFORM(image=rgb)
        image_tensor = augmented["image"]

        if self.landmark_extractor is None:
            from src.detection.landmark_extractor import LandmarkExtractor
            self.landmark_extractor = LandmarkExtractor()

        lm_vec = self.landmark_extractor.extract(bgr)

        return {
            "image": image_tensor,
            "landmarks": torch.tensor(lm_vec, dtype=torch.float32),
            "clean_label": item["clean_label"],
            "orig_label": item["orig_label"],
        }


def run_benchmark():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = Path("weights/emotion_model_best.pth")

    if not model_path.exists():
        logger.error(f"Model weights not found at {model_path}")
        return

    logger.info(f"Using device: {device}")
    model = build_model(freeze_backbone=False).to(device)
    ckpt = torch.load(str(model_path), map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model.eval()
    logger.info("Loaded model weights successfully.")

    dataset = FERPlusTestDataset()
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)

    preds_list = []
    orig_labels_list = []
    clean_labels_list = []

    logger.info("Running inference across test benchmark ...")
    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            landmarks = batch["landmarks"].to(device)
            out = model(images, landmarks)
            preds = out["logits"].argmax(dim=1).cpu().numpy()

            preds_list.extend(preds)
            orig_labels_list.extend(batch["orig_label"].numpy())
            clean_labels_list.extend(batch["clean_label"].numpy())

    preds = np.array(preds_list)
    orig_labels = np.array(orig_labels_list)
    clean_labels = np.array(clean_labels_list)

    # Calculate metrics
    orig_acc = (preds == orig_labels).mean() * 100
    clean_acc = (preds == clean_labels).mean() * 100

    orig_f1 = f1_score(orig_labels, preds, average="weighted")
    clean_f1 = f1_score(clean_labels, preds, average="weighted")

    print("\n" + "="*70)
    print("           BENCHMARK COMPARISON: FER-2013 vs FER+ (MICROSOFT)")
    print("="*70)
    print(f"Original FER-2013 Test Accuracy : {orig_acc:.2f}%  (Weighted F1: {orig_f1:.4f})")
    print(f"Cleaned FER+ Test Accuracy      : {clean_acc:.2f}%  (Weighted F1: {clean_f1:.4f})")
    print(f"Net Accuracy Gain               : +{clean_acc - orig_acc:.2f}%")
    print("="*70)

    clean_report = classification_report(clean_labels, preds, target_names=PROJECT_CLASSES, digits=4)
    print("\nFER+ Detailed Classification Report:")
    print(clean_report)

    # Save comparative confusion matrix
    cm = confusion_matrix(clean_labels, preds)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Purples",
        xticklabels=PROJECT_CLASSES,
        yticklabels=PROJECT_CLASSES,
        ax=ax,
    )
    ax.set_title(f"FER+ (Clean Ground Truth) Confusion Matrix (Acc: {clean_acc:.2f}%)", fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label (FER+ Majority Vote)")
    ax.set_xlabel("Predicted Label (AVIS Dual-Stream)")
    plt.tight_layout()

    out_cm_path = Path("results/ferplus_confusion_matrix.png")
    out_cm_path.parent.mkdir(exist_ok=True)
    fig.savefig(out_cm_path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved FER+ confusion matrix to {out_cm_path}")

    # Save summary report text file
    summary_txt = Path("results/benchmark_comparison.txt")
    with open(summary_txt, "w") as f:
        f.write("AVIS — BENCHMARK COMPARISON: FER-2013 vs FER+\n")
        f.write("="*60 + "\n")
        f.write(f"Original FER-2013 Test Accuracy : {orig_acc:.2f}%\n")
        f.write(f"Cleaned FER+ Test Accuracy      : {clean_acc:.2f}%\n")
        f.write(f"Accuracy Improvement            : +{clean_acc - orig_acc:.2f}%\n")
        f.write(f"Original Weighted F1            : {orig_f1:.4f}\n")
        f.write(f"Cleaned Weighted F1             : {clean_f1:.4f}\n\n")
        f.write("FER+ Classification Report:\n")
        f.write(clean_report)
    logger.info(f"Saved summary metrics to {summary_txt}")


if __name__ == "__main__":
    run_benchmark()
