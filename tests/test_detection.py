"""
AVIS — Unit Tests: Detection Module
"""

import numpy as np
import pytest
from PIL import Image

from src.detection.face_detector import FaceDetection, FaceDetector
from src.detection.landmark_extractor import LandmarkExtractor


@pytest.fixture(scope="module")
def detector():
    return FaceDetector()


@pytest.fixture(scope="module")
def landmark_extractor():
    """Always returns a LandmarkExtractor — may be in zero-vector mode if MediaPipe unavailable."""
    return LandmarkExtractor()


@pytest.fixture
def blank_image():
    """A plain white 224x224 BGR image (no face — tests zero-detection path)."""
    return np.ones((224, 224, 3), dtype=np.uint8) * 255


@pytest.fixture
def face_image_pil():
    """Simple synthetic PIL image."""
    return Image.new("RGB", (256, 256), color=(200, 180, 160))


class TestFaceDetector:
    def test_detect_returns_list(self, detector, blank_image):
        results = detector.detect(blank_image)
        assert isinstance(results, list)

    def test_detect_from_pil(self, detector, face_image_pil):
        results = detector.detect(face_image_pil)
        assert isinstance(results, list)

    def test_detection_structure(self, detector, blank_image):
        # Force Haar cascade path with a synthetic face-like image
        results = detector.detect(blank_image)
        for r in results:
            assert isinstance(r, FaceDetection)
            assert len(r.bbox) == 4
            assert 0.0 <= r.confidence <= 1.0
            assert r.crop.ndim == 3  # (H, W, 3)

    def test_empty_image_handled(self, detector):
        tiny = np.zeros((10, 10, 3), dtype=np.uint8)
        results = detector.detect(tiny)
        assert isinstance(results, list)


class TestLandmarkExtractor:
    def test_instantiates_without_error(self, landmark_extractor):
        """LandmarkExtractor must always construct — even in zero-vector mode."""
        assert landmark_extractor is not None

    def test_returns_correct_dim(self, landmark_extractor, blank_image):
        vec = landmark_extractor.extract(blank_image)
        assert vec.shape == (936,), f"Expected (936,), got {vec.shape}"

    def test_returns_zeros_on_empty(self, landmark_extractor):
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        vec = landmark_extractor.extract(empty)
        assert vec.shape == (936,)
        assert (vec == 0).all()

    def test_batch_returns_correct_shape(self, landmark_extractor, blank_image):
        batch = [blank_image] * 4
        result = landmark_extractor.extract_batch(batch)
        assert result.shape == (4, 936)

    def test_values_in_valid_range(self, landmark_extractor, blank_image):
        vec = landmark_extractor.extract(blank_image)
        # Values are either 0.0 (no face / zero-mode) or in [0, 1] (landmarks)
        assert vec.min() >= 0.0
        assert vec.max() <= 1.0

    def test_none_input_returns_zeros(self, landmark_extractor):
        vec = landmark_extractor.extract(None)
        assert vec.shape == (936,)
        assert (vec == 0).all()
