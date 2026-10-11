"""
AVIS — Face Detector
Uses YOLOv8n-face (ultralytics) to detect and crop all faces in an image.
Falls back to a Haar cascade if YOLO weights not yet downloaded.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np
import torch
from PIL import Image

from src.utils.config import config
from src.utils.logger import logger


class FaceDetection(NamedTuple):
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2 (absolute pixels)
    confidence: float
    crop: np.ndarray  # BGR crop of the face


class FaceDetector:
    """
    YOLOv8n-face wrapper for multi-face detection.

    Usage:
        detector = FaceDetector()
        detections = detector.detect(image)   # PIL or np.ndarray (BGR)
    """

    # HuggingFace repo that has the ONNX/PT face model
    _HF_FACE_MODEL = "arnabdhar/YOLOv8-Face-Detection"
    _MODEL_FILENAME = "model.pt"

    def __init__(self, conf_threshold: float | None = None):
        self.conf = conf_threshold or config.face_detection_confidence
        self._model = None
        self._haar = None
        self._load_model()

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_model(self) -> None:
        """Load YOLOv8-face. Falls back to Haar cascade if unavailable."""
        try:
            from ultralytics import YOLO

            weights = config.yolo_model_path
            if not weights.exists():
                logger.info("Downloading YOLOv8-face weights from HuggingFace …")
                from huggingface_hub import hf_hub_download

                downloaded = hf_hub_download(
                    repo_id=self._HF_FACE_MODEL,
                    filename=self._MODEL_FILENAME,
                    local_dir=config.weights_dir,
                )
                weights = Path(downloaded)
                # rename to expected path
                weights.rename(config.yolo_model_path)
                weights = config.yolo_model_path

            self._model = YOLO(str(weights))
            logger.info("YOLOv8-face loaded successfully.")

        except Exception as exc:
            logger.warning(f"YOLOv8 unavailable ({exc}). Falling back to Haar cascade.")
            haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._haar = cv2.CascadeClassifier(haar_path)

    # ── Public API ────────────────────────────────────────────────────────────
    def detect(self, image: np.ndarray | Image.Image) -> list[FaceDetection]:
        """
        Detect all faces in an image.

        Args:
            image: PIL Image (RGB) or numpy array (BGR).

        Returns:
            List of FaceDetection named-tuples, sorted by confidence (desc).
        """
        bgr = self._to_bgr(image)

        if self._model is not None:
            return self._detect_yolo(bgr)
        return self._detect_haar(bgr)

    def detect_pil(self, image: Image.Image) -> list[FaceDetection]:
        """Convenience: detect from a PIL image."""
        return self.detect(image)

    # ── Internal detection methods ────────────────────────────────────────────
    def _detect_yolo(self, bgr: np.ndarray) -> list[FaceDetection]:
        results = self._model.predict(
            bgr,
            conf=self.conf,
            verbose=False,
            device=config.device if torch.cuda.is_available() else "cpu",
        )
        detections: list[FaceDetection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                conf = float(box.conf[0])
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                # Add 15% margin to preserve forehead, eyebrows, and jawline
                w, h = x2 - x1, y2 - y1
                pad_w, pad_h = int(w * 0.15), int(h * 0.15)
                crop_x1 = max(0, x1 - pad_w)
                crop_y1 = max(0, y1 - pad_h)
                crop_x2 = min(bgr.shape[1], x2 + pad_w)
                crop_y2 = min(bgr.shape[0], y2 + pad_h)
                crop = bgr[crop_y1:crop_y2, crop_x1:crop_x2]
                if crop.size == 0:
                    continue
                detections.append(FaceDetection((x1, y1, x2, y2), conf, crop))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        logger.debug(f"YOLOv8 detected {len(detections)} face(s).")
        return detections

    def _detect_haar(self, bgr: np.ndarray) -> list[FaceDetection]:
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        detections: list[FaceDetection] = []
        for x, y, w, h in faces:
            x2, y2 = x + w, y + h
            crop = bgr[y:y2, x:x2]
            detections.append(FaceDetection((x, y, x2, y2), 0.7, crop))
        logger.debug(f"Haar detected {len(detections)} face(s).")
        return detections

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _to_bgr(image: np.ndarray | Image.Image) -> np.ndarray:
        if isinstance(image, Image.Image):
            return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        return image  # assume already BGR
