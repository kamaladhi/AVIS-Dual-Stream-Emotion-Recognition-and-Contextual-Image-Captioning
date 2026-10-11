"""
AVIS — Dual-Stream Emotion Recognition Model  ⭐ Novel Contribution

Architecture:
    Stream 1 (Visual):   EfficientNet-B4 pretrained backbone → 1792-dim feature
    Stream 2 (Geometric): MLP on 936-dim MediaPipe landmark vector → 256-dim feature
    Fusion:              Concat [1792 + 256] → Linear(2048→512) → BN → ReLU → Dropout
                         → Linear(512→256) → ReLU → Dropout → Linear(256→7)

This dual-stream design outperforms single-stream CNN because:
    - Visual features capture texture/colour patterns (e.g., rosy cheeks when happy)
    - Geometric features capture face muscle geometry (e.g., AU6 cheek raise)
    - Together they provide complementary evidence
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import config


class LandmarkMLP(nn.Module):
    """Stream 2: Small MLP to encode 936-dim landmark vectors."""

    def __init__(self, input_dim: int, hidden_dim: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DualStreamEmotionModel(nn.Module):
    """
    Dual-stream emotion classifier: EfficientNet-B4 + MediaPipe landmarks.

    Args:
        num_classes:       Number of emotion classes (default: 8).
        dropout:           Dropout probability.
        freeze_backbone:   If True, freeze EfficientNet weights during early training.
    """

    def __init__(
        self,
        num_classes: int = config.num_emotion_classes,
        dropout: float = config.dropout_rate,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        # ── Stream 1: EfficientNet-B4 visual backbone ─────────────────────────
        self.visual_encoder = timm.create_model(
            config.efficientnet_variant,
            pretrained=True,
            num_classes=0,  # remove classifier head → feature extractor
            global_pool="avg",
        )
        visual_out = config.efficientnet_out_features  # 1792

        if freeze_backbone:
            for p in self.visual_encoder.parameters():
                p.requires_grad = False

        # ── Stream 2: Landmark MLP ────────────────────────────────────────────
        self.landmark_encoder = LandmarkMLP(
            input_dim=config.landmark_input_dim,  # 936
            hidden_dim=config.landmark_hidden_dim,  # 256
            out_dim=config.landmark_hidden_dim,  # 256
            dropout=dropout,
        )

        # ── Fusion head ───────────────────────────────────────────────────────
        fusion_in = visual_out + config.landmark_hidden_dim  # 2048
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, config.fusion_hidden_dim),  # 512
            nn.BatchNorm1d(config.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim // 2),  # 256
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),
        )
        self.classifier = nn.Linear(config.fusion_hidden_dim // 2, num_classes)

        # ── Emotion embedding (used by the caption fusion head) ───────────────
        self.emotion_embed = nn.Embedding(num_classes, config.emotion_embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in [self.fusion, self.classifier, self.landmark_encoder]:
            for layer in (m.modules() if hasattr(m, "modules") else [m]):
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(
        self,
        images: torch.Tensor,  # (B, 3, H, W)
        landmarks: torch.Tensor,  # (B, 936)
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            logits:        (B, num_classes) raw class scores
            probs:         (B, num_classes) softmax probabilities
            embeddings:    (B, emotion_embed_dim) emotion embedding for fusion head
        """
        # Stream 1
        vis_feat = self.visual_encoder(images)  # (B, 1280)

        # Stream 2
        lm_feat = self.landmark_encoder(landmarks)  # (B, 256)

        # Fusion
        fused = torch.cat([vis_feat, lm_feat], dim=1)  # (B, 2048)
        fused = self.fusion(fused)  # (B, 256)
        logits = self.classifier(fused)  # (B, num_classes)
        probs = F.softmax(logits, dim=-1)

        # Emotion embedding for caption fusion
        pred_class = probs.argmax(dim=-1)  # (B,)
        embeddings = self.emotion_embed(pred_class)  # (B, 128)

        return {
            "logits": logits,
            "probs": probs,
            "embeddings": embeddings,
        }

    def unfreeze_backbone(self) -> None:
        """Unfreeze EfficientNet for fine-tuning (call after warmup epochs)."""
        for p in self.visual_encoder.parameters():
            p.requires_grad = True


def build_model(freeze_backbone: bool = True) -> DualStreamEmotionModel:
    """Factory function — use this instead of direct instantiation."""
    model = DualStreamEmotionModel(freeze_backbone=freeze_backbone)
    return model
