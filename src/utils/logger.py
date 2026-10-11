"""
AVIS — Structured Logging via Loguru
Import `logger` from here everywhere in the project.
"""

import sys
from pathlib import Path

from loguru import logger

from src.utils.config import ROOT_DIR

# ── Log directory ─────────────────────────────────────────────────────────────
LOG_DIR = ROOT_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

# ── Remove default handler ────────────────────────────────────────────────────
logger.remove()

# ── Console handler (pretty, coloured) ───────────────────────────────────────
logger.add(
    sys.stderr,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> — "
        "<level>{message}</level>"
    ),
    level="INFO",
    colorize=True,
)

# ── File handler (rotating, keeps 7 days) ────────────────────────────────────
logger.add(
    LOG_DIR / "avis_{time:YYYY-MM-DD}.log",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} — {message}",
    level="DEBUG",
    rotation="00:00",  # rotate daily
    retention="7 days",
    compression="zip",
    enqueue=True,  # Required for safe multiprocessing on Windows
)

__all__ = ["logger"]
