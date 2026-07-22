"""
AVIS — Emotion Inference Module
Loads a trained DualStreamEmotionModel and runs inference on face crops.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from src.detection.landmark_extractor import LandmarkExtractor
from src.emotion.model import DualStreamEmotionModel, build_model
from src.utils.config import config
from src.utils.logger import logger

# Standard ImageNet normalization
_NORMALIZE = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
_IMG_SIZE = 224


@dataclass
class EmotionResult:
    label: str               # Predicted emotion class name
    confidence: float        # Probability of predicted class [0, 1]
    all_probs: dict[str, float]   # {emotion: probability} for all classes
    embedding: np.ndarray    # 128-dim emotion embedding for caption fusion


class EmotionInference:
    """
    Runs inference with the DualStreamEmotionModel.

    Usage:
        infer = EmotionInference()
        result = infer.predict(face_crop_bgr)
    """

    def __init__(self, model_path: Path | None = None):
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self.class_names = config.emotion_classes
        self.landmark_extractor = LandmarkExtractor()
        self.model = self._load_model(model_path or config.emotion_model_path)
        logger.info(f"EmotionInference ready on {self.device}.")

    def _load_model(self, path: Path) -> DualStreamEmotionModel:
        model = build_model(freeze_backbone=False)
        if path.exists():
            ckpt = torch.load(str(path), map_location=self.device)
            state = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state)
            logger.info(f"Loaded emotion model from {path}")
        else:
            logger.warning(
                f"Model weights not found at {path}. "
                "Using random weights — run training first!"
            )
        model.to(self.device).eval()
        return model

    def predict(self, face_bgr: np.ndarray) -> EmotionResult:
        """
        Predict emotion for a single face crop.

        Args:
            face_bgr: BGR numpy array (cropped face).

        Returns:
            EmotionResult with label, confidence, all_probs, embedding.
        """
        img_tensor = self._preprocess_image(face_bgr)
        lm_tensor = torch.tensor(
            self.landmark_extractor.extract(face_bgr),
            dtype=torch.float32,
        ).unsqueeze(0).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
            out = self.model(img_tensor, lm_tensor)

        probs = out["probs"][0].cpu().numpy()
        pred_idx = probs.argmax()

        return EmotionResult(
            label=self.class_names[pred_idx],
            confidence=float(probs[pred_idx]),
            all_probs={cls: float(p) for cls, p in zip(self.class_names, probs)},
            embedding=out["embeddings"][0].cpu().numpy(),
        )

    def predict_batch(self, face_crops: list[np.ndarray]) -> list[EmotionResult]:
        """Predict emotions for multiple face crops in one forward pass."""
        if not face_crops:
            return []

        images = torch.stack([self._preprocess_image(f)[0] for f in face_crops]).to(self.device)
        landmarks = torch.tensor(
            self.landmark_extractor.extract_batch(face_crops),
            dtype=torch.float32,
        ).to(self.device)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(self.device.type == "cuda")):
            out = self.model(images, landmarks)

        probs_np = out["probs"].cpu().numpy()
        embeddings_np = out["embeddings"].cpu().numpy()
        results = []
        for i, probs in enumerate(probs_np):
            pred_idx = probs.argmax()
            results.append(EmotionResult(
                label=self.class_names[pred_idx],
                confidence=float(probs[pred_idx]),
                all_probs={cls: float(p) for cls, p in zip(self.class_names, probs)},
                embedding=embeddings_np[i],
            ))
        return results

    def _preprocess_image(self, face_bgr: np.ndarray) -> torch.Tensor:
        """Resize, convert to RGB, normalize → (1, 3, 224, 224)."""
        resized = cv2.resize(face_bgr, (_IMG_SIZE, _IMG_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        tensor = _NORMALIZE(rgb).unsqueeze(0).to(self.device)
        return tensor
