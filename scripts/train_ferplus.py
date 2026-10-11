"""
AVIS — Fine-Tuning Dual-Stream Model on Microsoft FER+ Clean Annotations

Loads the pre-trained weights (weights/emotion_model_best.pth) and fine-tunes
on the crowd-sourced clean FER+ majority-vote ground truth for 6 epochs.

Saves the new best checkpoint to:
    weights/emotion_model_ferplus_best.pth
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report, f1_score

from src.emotion.model import build_model
from src.utils.config import config
from src.utils.logger import logger

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


class FERPlusDataset(Dataset):
    """
    Dataset loader mapping FER-2013 images by row index to FER+ majority vote labels.
    """
    AUGMENT = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.08, rotate_limit=8, p=0.4),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    EVAL_TRANSFORM = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    IMG_SIZE = 224

    def __init__(
        self,
        split: str = "train",
        data_root: Path = Path("data/fer2013"),
        csv_path: Path = Path("data/fer2013new.csv"),
    ):
        self.split = split
        self.transform = self.AUGMENT if split == "train" else self.EVAL_TRANSFORM
        self.landmark_extractor = None

        df = pd.read_csv(csv_path)
        emotion_cols = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear"]

        # Map split to folder
        split_dir = data_root / split
        img_files = list(split_dir.rglob("*.png")) + list(split_dir.rglob("*.jpg"))

        self.samples = []
        for p in img_files:
            try:
                row_idx = int(p.stem)
            except ValueError:
                continue

            if row_idx >= len(df):
                continue

            row = df.iloc[row_idx]
            votes = {col: int(row[col]) for col in emotion_cols}
            unknown_votes = int(row.get("unknown", 0))
            nf_votes = int(row.get("NF", 0))

            max_emotion = max(votes, key=votes.get)
            max_votes = votes[max_emotion]

            # Filter out non-faces (NF) and completely ambiguous samples
            if max_votes < 1 or nf_votes >= max_votes or unknown_votes >= max_votes:
                continue

            clean_class = FERPLUS_COL_TO_CLASS[max_emotion]
            clean_idx = CLASS_TO_IDX[clean_class]

            self.samples.append((p, clean_idx))

        logger.info(f"FER+ [{split}]: {len(self.samples)} clean samples loaded.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, label = self.samples[idx]
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            bgr = np.zeros((self.IMG_SIZE, self.IMG_SIZE, 3), dtype=np.uint8)
        else:
            bgr = cv2.resize(bgr, (self.IMG_SIZE, self.IMG_SIZE))

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        augmented = self.transform(image=rgb)
        image_tensor = augmented["image"]

        if self.landmark_extractor is None:
            from src.detection.landmark_extractor import LandmarkExtractor
            self.landmark_extractor = LandmarkExtractor()

        lm_vec = self.landmark_extractor.extract(bgr)

        return {
            "image": image_tensor,
            "landmarks": torch.tensor(lm_vec, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }


def fine_tune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Datasets & Loaders
    train_ds = FERPlusDataset(split="train")
    val_ds = FERPlusDataset(split="val")
    test_ds = FERPlusDataset(split="test")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=0)

    # 2. Model
    model = build_model(freeze_backbone=False).to(device)
    base_ckpt = Path("weights/emotion_model_best.pth")
    if base_ckpt.exists():
        ckpt = torch.load(str(base_ckpt), map_location=device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        logger.info(f"Loaded starting weights from {base_ckpt}")

    # 3. Class Weights (for balance)
    train_labels = [s[1] for s in train_ds.samples]
    class_counts = np.bincount(train_labels, minlength=len(PROJECT_CLASSES))
    class_counts = np.maximum(class_counts, 1)  # avoid div by zero
    total_samples = len(train_labels)
    weights = total_samples / (len(PROJECT_CLASSES) * class_counts)
    weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    logger.info(f"Class Weights: {weights.round(2)}")

    criterion = nn.CrossEntropyLoss(weight=weights_tensor, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=6, eta_min=1e-6)

    best_val_acc = 0.0
    out_weights = Path("weights/emotion_model_ferplus_best.pth")
    out_weights.parent.mkdir(exist_ok=True)

    epochs = 6
    logger.info(f"Starting Fine-Tuning on FER+ for {epochs} epochs ...")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        # Train
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]", leave=False):
            images = batch["image"].to(device)
            landmarks = batch["landmarks"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                out = model(images, landmarks)
                loss = criterion(out["logits"], labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item() * images.size(0)
            preds = out["logits"].argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += images.size(0)

        train_acc = train_correct / train_total
        scheduler.step()

        # Val
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for batch in val_loader:
                images = batch["image"].to(device)
                landmarks = batch["landmarks"].to(device)
                labels = batch["label"].to(device)

                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    out = model(images, landmarks)
                    loss = criterion(out["logits"], labels)

                val_loss += loss.item() * images.size(0)
                preds = out["logits"].argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += images.size(0)

        val_acc = val_correct / val_total
        elapsed = time.time() - start_time

        logger.info(
            f"Epoch {epoch:02d}/{epochs:02d} | "
            f"Train Acc: {train_acc*100:.2f}% | "
            f"Val Acc: {val_acc*100:.2f}% | "
            f"Time: {elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_accuracy": val_acc,
            }, str(out_weights))
            logger.info(f" Saved new best weights to {out_weights} (Val Acc: {val_acc*100:.2f}%)")

    # 4. Final Evaluation on Test Set
    logger.info("Evaluating Best Checkpoint on Test Set ...")
    ckpt = torch.load(str(out_weights), map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            images = batch["image"].to(device)
            landmarks = batch["landmarks"].to(device)
            out = model(images, landmarks)
            test_preds.extend(out["logits"].argmax(dim=1).cpu().numpy())
            test_labels.extend(batch["label"].numpy())

    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    final_test_acc = (test_preds == test_labels).mean() * 100
    final_f1 = f1_score(test_labels, test_preds, average="weighted")

    print("\n" + "="*70)
    print("       FINAL FINE-TUNING RESULT ON CLEAN FER+ TEST SET")
    print("="*70)
    print(f"Final FER+ Test Accuracy: {final_test_acc:.2f}%")
    print(f"Final Weighted F1 Score : {final_f1:.4f}")
    print("="*70)
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, test_preds, target_names=PROJECT_CLASSES, digits=4))


if __name__ == "__main__":
    fine_tune()
