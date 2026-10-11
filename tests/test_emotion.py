"""
AVIS — Unit Tests: Emotion Model (architecture & forward pass)
Tests run on CPU with random weights — no trained model needed.
"""

import numpy as np
import pytest
import torch

from src.emotion.model import DualStreamEmotionModel, build_model
from src.utils.config import config


@pytest.fixture(scope="module")
def model():
    """Build model on CPU with random weights."""
    m = build_model(freeze_backbone=False)
    m.eval()
    return m


@pytest.fixture
def dummy_batch():
    B = 4
    images = torch.randn(B, 3, 224, 224)
    landmarks = torch.randn(B, 936)
    labels = torch.randint(0, config.num_emotion_classes, (B,))
    return images, landmarks, labels


class TestDualStreamModel:
    def test_forward_output_keys(self, model, dummy_batch):
        images, landmarks, _ = dummy_batch
        with torch.no_grad():
            out = model(images, landmarks)
        assert "logits" in out
        assert "probs" in out
        assert "embeddings" in out

    def test_logits_shape(self, model, dummy_batch):
        images, landmarks, _ = dummy_batch
        with torch.no_grad():
            out = model(images, landmarks)
        assert out["logits"].shape == (4, config.num_emotion_classes)

    def test_probs_sum_to_one(self, model, dummy_batch):
        images, landmarks, _ = dummy_batch
        with torch.no_grad():
            out = model(images, landmarks)
        sums = out["probs"].sum(dim=1)
        assert torch.allclose(sums, torch.ones(4), atol=1e-5)

    def test_embeddings_shape(self, model, dummy_batch):
        images, landmarks, _ = dummy_batch
        with torch.no_grad():
            out = model(images, landmarks)
        assert out["embeddings"].shape == (4, config.emotion_embed_dim)

    def test_single_sample(self, model):
        images = torch.randn(1, 3, 224, 224)
        landmarks = torch.randn(1, 936)
        with torch.no_grad():
            out = model(images, landmarks)
        assert out["logits"].shape == (1, config.num_emotion_classes)

    def test_unfreeze_backbone(self, model):
        model.unfreeze_backbone()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable > 0

    def test_parameter_count_reasonable(self, model):
        total = sum(p.numel() for p in model.parameters())
        # EfficientNet-B4 has ~19.3M params + landmark MLP + fusion head
        assert total > 15_000_000
        assert total < 30_000_000
