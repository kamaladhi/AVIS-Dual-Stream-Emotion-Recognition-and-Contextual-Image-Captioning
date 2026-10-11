"""
AVIS — BLIP-Large Image Captioning Wrapper
Runs Salesforce/blip-image-captioning-large in float16 to fit 4GB VRAM.

VRAM usage: ~1.9 GB (float16)
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from PIL import Image

from src.utils.config import config
from src.utils.logger import logger


class BLIPCaptioner:
    """
    BLIP-large image captioning wrapper.

    Usage:
        captioner = BLIPCaptioner()
        caption = captioner.caption(pil_image)
        captions = captioner.caption_batch([img1, img2])
    """

    def __init__(self):
        self._processor = None
        self._model = None
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        self._load()

    def _load(self) -> None:
        logger.info(f"Loading BLIP-large ({config.blip_model_id}) in {config.blip_precision} …")
        from transformers import BlipForConditionalGeneration, BlipProcessor

        self._processor = BlipProcessor.from_pretrained(
            config.blip_model_id,
            token=config.hf_token or None,
        )
        dtype = torch.float16 if config.blip_precision == "float16" else torch.float32
        self._model = BlipForConditionalGeneration.from_pretrained(
            config.blip_model_id,
            torch_dtype=dtype,
            token=config.hf_token or None,
        ).to(self.device)
        self._model.eval()
        logger.info(f"BLIP-large loaded on {self.device}.")

    # ── Public API ────────────────────────────────────────────────────────────

    def caption(
        self,
        image: Union[Image.Image, np.ndarray],
        prompt: str | None = None,
    ) -> str:
        """
        Generate a caption for a single image.

        Args:
            image:  PIL Image (RGB) or OpenCV BGR numpy array.
            prompt: Optional conditional prefix (e.g., "a photo of").

        Returns:
            Caption string.
        """
        pil = self._to_pil(image)
        return self.caption_batch([pil], [prompt] if prompt else [None])[0]

    def caption_batch(
        self,
        images: list[Union[Image.Image, np.ndarray]],
        prompts: list[str | None] | None = None,
    ) -> list[str]:
        """
        Generate captions for a list of images.

        Returns:
            List of caption strings.
        """
        pil_images = [self._to_pil(img) for img in images]
        if prompts is None:
            prompts = [None] * len(pil_images)

        inputs = self._processor(
            images=pil_images,
            text=[p or "" for p in prompts],
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Cast to model dtype
        if config.blip_precision == "float16":
            inputs["pixel_values"] = inputs["pixel_values"].half()

        with torch.no_grad():
            ids = self._model.generate(
                **inputs,
                max_new_tokens=config.max_caption_length,
                min_new_tokens=config.min_caption_length,
                num_beams=4,
                repetition_penalty=1.3,
            )

        captions = self._processor.batch_decode(ids, skip_special_tokens=True)
        return [c.strip() for c in captions]

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _to_pil(image: Union[Image.Image, np.ndarray]) -> Image.Image:
        if isinstance(image, Image.Image):
            return image.convert("RGB")
        # Assume BGR numpy array
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
