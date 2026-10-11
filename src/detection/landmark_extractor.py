"""
AVIS — Facial Landmark Extractor (Stream 2 of the Dual-Stream Emotion Model)
Uses MediaPipe FaceMesh to extract 468 2D facial landmarks from a face crop.
The (x, y) coordinates are flattened into a 936-dim feature vector,
normalized to [0, 1] by MediaPipe.

Compatibility:
    - mediapipe == 0.10.13  → uses legacy mp.solutions.face_mesh API
    - mediapipe >= 0.10.14  → uses new MediaPipe Tasks API (FaceLandmarker)
    - mediapipe unavailable → returns zero vectors silently (graceful degradation)
"""

from __future__ import annotations

import os
from pathlib import Path

# Suppress TF Lite C++ INFO logs before mediapipe is imported
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np

from src.utils.logger import logger


class LandmarkExtractor:
    """
    MediaPipe FaceMesh landmark extractor.

    Returns a 936-dim vector (468 landmarks × 2 coords) normalized to [0, 1].
    Returns a zero vector if no face is detected or MediaPipe is unavailable.

    Usage:
        extractor = LandmarkExtractor()
        vec = extractor.extract(face_crop_bgr)  # np.ndarray (936,) or zeros
    """

    LANDMARK_DIM = 936  # 468 × 2

    # Remote model for new Tasks API (downloaded once and cached)
    _TASKS_MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    )
    _TASKS_MODEL_PATH = Path.home() / ".cache" / "avis" / "face_landmarker.task"

    def __init__(self, static_mode: bool = True):
        self._available = False
        self._use_tasks_api = False
        self._face_mesh = None
        self._landmarker = None

        self._init_mediapipe(static_mode)

    def _init_mediapipe(self, static_mode: bool) -> None:
        """Try legacy API first, fall back to Tasks API, then graceful zero mode."""
        try:
            import mediapipe as mp
        except ImportError:
            logger.warning("MediaPipe not installed. Landmark stream will return zeros.")
            return

        # ── Strategy 1: Legacy mp.solutions API (works on mediapipe <= 0.10.13) ──
        if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
            try:
                self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                    static_image_mode=static_mode,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.4,
                )
                self._available = True
                self._use_tasks_api = False
                logger.info(f"MediaPipe FaceMesh (legacy API, v{mp.__version__}) initialized.")
                return
            except Exception as exc:
                logger.warning(f"Legacy FaceMesh failed: {exc}. Trying Tasks API …")

        # ── Strategy 2: New MediaPipe Tasks API (mediapipe >= 0.10.14) ──────────
        try:
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision as mp_vision

            model_path = self._ensure_tasks_model()
            if model_path is None:
                raise RuntimeError("Could not download face_landmarker.task model.")

            base_options = mp_tasks.BaseOptions(model_asset_path=str(model_path))
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
                min_face_detection_confidence=0.4,
                running_mode=mp_vision.RunningMode.IMAGE,
            )
            self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self._available = True
            self._use_tasks_api = True
            logger.info(f"MediaPipe FaceLandmarker (Tasks API, v{mp.__version__}) initialized.")

        except Exception as exc:
            logger.warning(
                f"MediaPipe Tasks API unavailable ({exc}). "
                "Landmark stream will return zero vectors. "
                "This is safe — the visual stream still works."
            )

    @classmethod
    def _ensure_tasks_model(cls) -> Path | None:
        """Download the FaceLandmarker model if not cached."""
        if cls._TASKS_MODEL_PATH.exists():
            return cls._TASKS_MODEL_PATH
        try:
            import urllib.request

            cls._TASKS_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading face_landmarker.task model (~29MB) …")
            urllib.request.urlretrieve(cls._TASKS_MODEL_URL, cls._TASKS_MODEL_PATH)
            logger.info("FaceLandmarker model downloaded.")
            return cls._TASKS_MODEL_PATH
        except Exception as exc:
            logger.warning(f"Model download failed: {exc}")
            return None

    # ── Public API ────────────────────────────────────────────────────────────

    def extract(self, face_bgr: np.ndarray) -> np.ndarray:
        """
        Extract normalized 2D landmarks from a face crop.

        Args:
            face_bgr: BGR numpy array of a single cropped face.

        Returns:
            np.ndarray of shape (936,) with values in [0, 1].
            Returns zeros if extraction fails.
        """
        if not self._available or face_bgr is None or face_bgr.size == 0:
            return np.zeros(self.LANDMARK_DIM, dtype=np.float32)

        try:
            face_rgb = cv2.cvtColor(
                cv2.resize(face_bgr, (224, 224)),
                cv2.COLOR_BGR2RGB,
            )
            if self._use_tasks_api:
                return self._extract_tasks(face_rgb)
            return self._extract_legacy(face_rgb)
        except Exception as exc:
            logger.debug(f"Landmark extraction failed: {exc}")
            return np.zeros(self.LANDMARK_DIM, dtype=np.float32)

    def _extract_legacy(self, face_rgb: np.ndarray) -> np.ndarray:
        """Extract using mp.solutions.face_mesh."""
        results = self._face_mesh.process(face_rgb)
        if not results.multi_face_landmarks:
            return np.zeros(self.LANDMARK_DIM, dtype=np.float32)
        landmarks = results.multi_face_landmarks[0].landmark
        coords = np.array([[lm.x, lm.y] for lm in landmarks[:468]], dtype=np.float32)
        return coords.flatten()

    def _extract_tasks(self, face_rgb: np.ndarray) -> np.ndarray:
        """Extract using new MediaPipe Tasks FaceLandmarker."""
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=face_rgb)
        result = self._landmarker.detect(mp_image)
        if not result.face_landmarks:
            return np.zeros(self.LANDMARK_DIM, dtype=np.float32)
        landmarks = result.face_landmarks[0]
        coords = np.array([[lm.x, lm.y] for lm in landmarks[:468]], dtype=np.float32)
        return coords.flatten()

    def extract_batch(self, face_crops: list[np.ndarray]) -> np.ndarray:
        """
        Extract landmarks for a batch of face crops.

        Returns:
            np.ndarray of shape (N, 936).
        """
        return np.stack([self.extract(crop) for crop in face_crops])

    def close(self) -> None:
        """Release MediaPipe resources."""
        try:
            if self._face_mesh:
                self._face_mesh.close()
            if self._landmarker:
                self._landmarker.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
