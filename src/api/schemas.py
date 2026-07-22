"""
AVIS — Pydantic Schemas for FastAPI
"""
from __future__ import annotations
from pydantic import BaseModel, Field


class EmotionProbs(BaseModel):
    angry: float
    disgust: float
    fear: float
    happy: float
    neutral: float
    sad: float
    surprise: float


class FaceResult(BaseModel):
    bbox: list[int] = Field(..., description="[x1, y1, x2, y2] in pixels")
    detection_confidence: float
    emotion: str
    emotion_confidence: float
    emotion_probs: EmotionProbs
    base_caption: str
    emotion_caption: str


class PredictResponse(BaseModel):
    num_faces: int
    scene_caption: str
    agentic_caption: str
    processing_time_ms: float
    faces: list[FaceResult]


class HealthResponse(BaseModel):
    status: str = "ok"
    model: str = "AVIS v1.0"
    device: str
