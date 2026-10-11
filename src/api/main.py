"""
AVIS — FastAPI Application Entry Point

Run:
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import routes
from src.pipeline.avis_pipeline import AVISPipeline
from src.utils.config import config
from src.utils.logger import logger

# ── App lifecycle ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the AVIS pipeline on startup, release on shutdown."""
    logger.info("🚀 Starting AVIS API — loading models …")
    pipeline = AVISPipeline(lazy_load=False)
    routes.set_pipeline(pipeline)
    logger.info("✅ AVIS API ready.")
    yield
    logger.info("🛑 AVIS API shutting down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AVIS — Affective Visual Intelligence System",
    description=(
        "Production-grade emotion-aware image captioning API.\n\n"
        "**Pipeline:** YOLOv8-face → EfficientNet-B0 + MediaPipe (dual-stream) "
        "→ BLIP-large → Emotion-Caption Fusion"
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────
app.include_router(routes.router, prefix="/api/v1")


@app.get("/", tags=["Root"])
async def root():
    return {
        "service": "AVIS — Affective Visual Intelligence System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health",
    }
