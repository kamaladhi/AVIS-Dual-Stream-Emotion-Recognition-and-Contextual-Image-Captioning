"""
AVIS — FastAPI Application Routes
"""
from __future__ import annotations

import io

import numpy as np
from fastapi import APIRouter, File, HTTPException, UploadFile
from PIL import Image

from src.api.schemas import HealthResponse, PredictResponse
from src.utils.config import config
from src.utils.logger import logger

router = APIRouter()

# Pipeline is injected from main.py (singleton)
_pipeline = None


def set_pipeline(pipeline) -> None:
    global _pipeline
    _pipeline = pipeline


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        model="AVIS v1.0",
        device=config.device,
    )


@router.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(file: UploadFile = File(..., description="Image file (JPG/PNG)")):
    """
    Run full AVIS inference on an uploaded image.

    Returns per-face emotion recognition + emotion-aware captions.
    """
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized.")

    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")

    try:
        contents = await file.read()
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}")

    try:
        result = _pipeline.run_image(pil_image, annotate=False)
    except Exception as exc:
        logger.error(f"Pipeline error: {exc}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}")

    return PredictResponse(**result.to_dict())
