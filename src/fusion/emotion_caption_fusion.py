"""
AVIS — Emotion-Caption Fusion Head  ⭐ Novel Contribution

Takes:
    - base_caption (str)   — from BLIP-large
    - emotion_result       — from DualStreamEmotionModel

Produces:
    - emotion_caption (str) — emotionally conditioned, enriched caption

Two fusion strategies:
    1. PROMPT INJECTION (default, no extra VRAM):
       Constructs a rich prompt combining emotion context + base caption,
       then feeds it back to BLIP in conditional generation mode.

    2. EMBEDDING FUSION (advanced, requires fine-tuning):
       Projects emotion embedding into BLIP's token embedding space
       via a learned linear layer and prepends it to the input sequence.
       (Implemented as EmotionEmbeddingFusion — enable after training.)
"""

from __future__ import annotations

import re

import numpy as np
import torch
import torch.nn as nn

from src.utils.config import config
from src.utils.logger import logger

# ── Emotion → contextual descriptor mapping ───────────────────────────────────
EMOTION_DESCRIPTORS = {
    "happy": ("joyfully", "with a warm, happy expression"),
    "sad": ("somberly", "with a sorrowful, melancholic expression"),
    "angry": ("intensely", "with a tense, angry expression"),
    "fear": ("anxiously", "with a fearful, worried expression"),
    "surprise": ("in surprise", "with a startled, astonished expression"),
    "disgust": ("with distaste", "with a disgusted expression"),
    "neutral": ("calmly", "with a neutral, composed expression"),
    "contempt": ("dismissively", "with a contemptuous expression"),
}


class PromptFusionHead:
    """
    Strategy 1: Prompt-based emotion conditioning (zero extra VRAM).

    Constructs an emotion-aware conditional prompt and regenerates the caption
    using BLIP in conditional (text-guided) mode.
    """

    def __init__(self, captioner=None):
        self._captioner = captioner

    @property
    def captioner(self):
        return self._captioner

    @captioner.setter
    def captioner(self, value):
        self._captioner = value

    def fuse(
        self,
        base_caption: str,
        emotion: str,
        confidence: float,
        original_image,
    ) -> str:
        """
        Produce an emotion-conditioned caption.

        Args:
            base_caption:    Caption from BLIP (no emotion conditioning).
            emotion:         Predicted emotion label.
            confidence:      Prediction confidence [0, 1].
            original_image:  PIL or BGR numpy — full original image for re-captioning.

        Returns:
            Emotion-enriched caption string.
        """
        adverb, descriptor = EMOTION_DESCRIPTORS.get(emotion, ("", ""))

        # High-confidence: Use conditional re-generation from BLIP
        if confidence >= config.emotion_confidence_threshold and self._captioner is not None:
            prompt = f"a photo of a person {descriptor},"
            try:
                recaptioned = self._captioner.caption(original_image, prompt=prompt)
                if len(recaptioned) > 15:
                    return self._clean(recaptioned)
            except Exception as exc:
                logger.warning(f"Conditional captioning failed: {exc}. Falling back to injection.")

        # Fallback: simple linguistic injection
        return self._inject_emotion(base_caption, emotion, adverb, descriptor, confidence)

    @staticmethod
    def _inject_emotion(
        caption: str,
        emotion: str,
        adverb: str,
        descriptor: str,
        confidence: float,
    ) -> str:
        """Linguistically inject emotion into the base caption."""
        caption = caption.strip().rstrip(".")

        # Find if there's a person in the caption and enrich
        person_keywords = ["person", "man", "woman", "people", "child", "boy", "girl", "face"]
        has_person = any(kw in caption.lower() for kw in person_keywords)

        conf_qualifier = ""
        if confidence < 0.6:
            conf_qualifier = "possibly "

        if has_person:
            # Insert descriptor after the first person mention
            for kw in person_keywords:
                if kw in caption.lower():
                    pattern = re.compile(re.escape(kw), re.IGNORECASE)
                    enriched = pattern.sub(f"{kw} {conf_qualifier}{descriptor}", caption, count=1)
                    return enriched + "."

        # No person found — append emotion context
        return f"{caption}, {conf_qualifier}{descriptor}."

    @staticmethod
    def _clean(text: str) -> str:
        text = text.strip()
        if text and not text.endswith("."):
            text += "."
        return text


class EmotionEmbeddingFusion(nn.Module):
    """
    Strategy 2: Learned embedding fusion (advanced — requires fine-tuning).

    Projects the 128-dim emotion embedding into BLIP's 768-dim text embedding
    space and prepends it as a soft token before caption generation.
    Requires a BLIP model with accessible text encoder.
    """

    def __init__(
        self,
        emotion_embed_dim: int = config.emotion_embed_dim,
        blip_hidden_dim: int = 768,
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(emotion_embed_dim, blip_hidden_dim),
            nn.LayerNorm(blip_hidden_dim),
            nn.GELU(),
        )

    def forward(self, emotion_embedding: torch.Tensor) -> torch.Tensor:
        """
        Project emotion embedding to BLIP's hidden dimension.

        Args:
            emotion_embedding: (B, 128)

        Returns:
            Projected token: (B, 1, 768) — prepend to BLIP encoder input.
        """
        projected = self.projection(emotion_embedding)  # (B, 768)
        return projected.unsqueeze(1)  # (B, 1, 768)
