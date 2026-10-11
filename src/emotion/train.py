"""
AVIS — Emotion Model Training Script
Trains the DualStreamEmotionModel on FER-2013 dataset.

Run:
    python -m src.emotion.train

Features:
    - Cosine LR scheduler with warmup
    - Label smoothing cross-entropy
    - Data augmentation via albumentations
    - MLflow experiment tracking
    - Best model checkpoint saving
    - Two-phase training: freeze backbone → unfreeze for fine-tuning
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

# Suppress noisy library warnings
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf")
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

import cv2
import mlflow
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

from src.detection.landmark_extractor import LandmarkExtractor
from src.emotion.model import build_model
from src.utils.config import config
from src.utils.logger import logger


# ── Dataset ───────────────────────────────────────────────────────────────────

class FER2013Dataset(Dataset):
    """
    FER-2013 dataset loader.
    Expects folder structure:
        data/fer2013/train/angry/  ... .jpg
        data/fer2013/train/happy/  ... .jpg
        ...
        data/fer2013/test/
    """

    AUGMENT = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.GaussNoise(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.4),
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
        root: Path,
        split: str = "train",
        class_names: list[str] | None = None,
        use_landmarks: bool = True,
    ):
        self.root = root / split
        self.split = split
        self.use_landmarks = use_landmarks
        self.class_names = class_names or config.emotion_classes
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.transform = self.AUGMENT if split == "train" else self.EVAL_TRANSFORM

        # Lazy init in __getitem__ to prevent multiprocessing pickle errors on Windows
        self.landmark_extractor = None

        self.samples: list[tuple[Path, int]] = []
        for cls_name in self.class_names:
            cls_dir = self.root / cls_name
            if not cls_dir.exists():
                logger.warning(f"Class folder not found: {cls_dir}")
                continue
            for img_path in cls_dir.glob("*.jpg"):
                self.samples.append((img_path, self.class_to_idx[cls_name]))
            for img_path in cls_dir.glob("*.png"):
                self.samples.append((img_path, self.class_to_idx[cls_name]))

        logger.info(f"FER-2013 [{split}]: {len(self.samples)} samples across {len(self.class_names)} classes.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        img_path, label = self.samples[idx]
        bgr = cv2.imread(str(img_path))
        bgr = cv2.resize(bgr, (self.IMG_SIZE, self.IMG_SIZE))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        augmented = self.transform(image=rgb)
        image_tensor = augmented["image"]   # (3, H, W)

        # Landmarks - instantiate lazily inside the worker process
        if self.use_landmarks:
            if self.landmark_extractor is None:
                from src.detection.landmark_extractor import LandmarkExtractor
                self.landmark_extractor = LandmarkExtractor()
            lm_vec = self.landmark_extractor.extract(bgr)
        else:
            lm_vec = np.zeros(936, dtype=np.float32)

        return {
            "image": image_tensor,
            "landmarks": torch.tensor(lm_vec, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ── Training loop ─────────────────────────────────────────────────────────────

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: torch.amp.GradScaler,
) -> dict[str, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for batch in pbar:
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

        total_loss += loss.item() * images.size(0)
        preds = out["logits"].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    pbar = tqdm(loader, desc="Evaluating", leave=False)
    for batch in pbar:
        images = batch["image"].to(device)
        landmarks = batch["landmarks"].to(device)
        labels = batch["label"].to(device)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            out = model(images, landmarks)
            loss = criterion(out["logits"], labels)

        total_loss += loss.item() * images.size(0)
        preds = out["logits"].argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return {"loss": total_loss / total, "accuracy": correct / total}


def train(data_root: str = "data/fer2013") -> None:
    """Main training entry point."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")

    # ── Datasets & loaders ────────────────────────────────────
    data_path = Path(data_root)
    train_ds = FER2013Dataset(data_path, "train")
    val_ds = FER2013Dataset(data_path, "test")

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size * 2,
        shuffle=False, num_workers=config.num_workers, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────
    model = build_model(freeze_backbone=True).to(device)
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # ── Calculate Class Weights ───────────────────────────────
    # FER-2013 is highly imbalanced (e.g., 'Disgust' is very rare).
    # We calculate inverse frequency weights: Total_Samples / (Num_Classes * Class_Count)
    labels = [sample[1] for sample in train_ds.samples]
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    num_classes = len(config.emotion_classes)
    
    weights = total_samples / (num_classes * class_counts)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    logger.info(f"Applied Class Weights: {weights.round(2)}")

    # ── Loss, optimizer, scheduler ────────────────────────────
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=config.label_smoothing)
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=config.num_epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── MLflow run ────────────────────────────────────────────
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    best_val_acc = 0.0
    unfreeze_done = False

    with mlflow.start_run():
        mlflow.log_params({
            "backbone": config.efficientnet_variant,
            "num_classes": config.num_emotion_classes,
            "batch_size": config.batch_size,
            "lr": config.learning_rate,
            "epochs": config.num_epochs,
            "label_smoothing": config.label_smoothing,
            "dual_stream": True,
        })

        for epoch in range(1, config.num_epochs + 1):
            # Phase-2: unfreeze backbone after 10 epochs
            if epoch == 11 and not unfreeze_done:
                model.unfreeze_backbone()
                for pg in optimizer.param_groups:
                    pg["lr"] = config.learning_rate * 0.1  # lower LR for fine-tune
                unfreeze_done = True
                logger.info("Backbone unfrozen for fine-tuning.")

            t0 = time.time()
            train_metrics = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
            val_metrics = evaluate(model, val_loader, criterion, device)
            scheduler.step()

            elapsed = time.time() - t0
            logger.info(
                f"Epoch {epoch:03d}/{config.num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} | "
                f"Time: {elapsed:.1f}s"
            )

            mlflow.log_metrics({
                "train_loss": train_metrics["loss"],
                "train_acc": train_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "val_acc": val_metrics["accuracy"],
                "lr": optimizer.param_groups[0]["lr"],
            }, step=epoch)

            # ── Save best checkpoint ───────────────────────────
            if val_metrics["accuracy"] > best_val_acc:
                best_val_acc = val_metrics["accuracy"]
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_acc": best_val_acc,
                        "config": {
                            "num_classes": config.num_emotion_classes,
                            "emotion_classes": config.emotion_classes,
                        },
                    },
                    config.emotion_model_path,
                )
                logger.info(f"  ✅ New best model saved (val_acc={best_val_acc:.4f})")
                
                # Register the model to MLflow Model Registry (commented out to avoid pt2 trace errors)
                # mlflow.pytorch.log_model(
                #     model,
                #     "model",
                #     registered_model_name="AVIS_Dual_Stream_Emotion"
                # )

        mlflow.log_metric("best_val_accuracy", best_val_acc)
        logger.info(f"Training complete. Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/fer2013", help="Path to FER-2013 dataset")
    args = parser.parse_args()
    train(args.data)
