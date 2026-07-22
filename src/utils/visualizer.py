"""
AVIS — Visualization Utilities
Draw bounding boxes, emotion labels, and captions on images/frames.
"""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.utils.config import config

# ── Emotion → colour mapping (BGR for OpenCV) ─────────────────────────────────
EMOTION_COLORS = {
    "angry":    (0,   0,   220),
    "disgust":  (0,   140, 0),
    "fear":     (130, 0,   130),
    "happy":    (0,   200, 50),
    "neutral":  (180, 180, 180),
    "sad":      (200, 100, 0),
    "surprise": (0,   180, 220),
    "contempt": (0,   100, 200),
}

EMOTION_EMOJIS = {
    "angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😄",
    "neutral": "😐", "sad": "😢", "surprise": "😲", "contempt": "😒",
}


def draw_face_results(
    frame: np.ndarray,
    faces: list[dict],
) -> np.ndarray:
    """
    Draw bounding boxes and emotion labels on an OpenCV frame.

    Args:
        frame:  BGR numpy array (H, W, 3)
        faces:  list of dicts with keys:
                  bbox (x1,y1,x2,y2), emotion, confidence, caption (optional)

    Returns:
        Annotated BGR frame.
    """
    overlay = frame.copy()

    for face in faces:
        x1, y1, x2, y2 = [int(v) for v in face["bbox"]]
        emotion: str = face.get("emotion", "neutral")
        conf: float = face.get("confidence", 0.0)
        color = EMOTION_COLORS.get(emotion, (200, 200, 200))

        # ── Bounding box ──────────────────────────────────────
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)

        # ── Semi-transparent label background ─────────────────
        label = f"{emotion.upper()}  {conf:.0%}"
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(overlay, (x1, y1 - lh - baseline - 6), (x1 + lw + 8, y1), color, -1)
        cv2.putText(
            overlay, label,
            (x1 + 4, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )

        # ── Caption below box (if present) ────────────────────
        caption = face.get("caption")
        if caption:
            _draw_wrapped_text(overlay, caption, (x1, y2 + 20), color, max_width=x2 - x1)

    # blend overlay for transparency effect
    return cv2.addWeighted(overlay, 0.9, frame, 0.1, 0)


def _draw_wrapped_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    max_width: int = 300,
    font_scale: float = 0.45,
    thickness: int = 1,
) -> None:
    """Draw word-wrapped text on an OpenCV frame in-place."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    words = text.split()
    lines, current = [], ""
    for word in words:
        test = f"{current} {word}".strip()
        (tw, _), _ = cv2.getTextSize(test, font, font_scale, thickness)
        if tw > max_width and current:
            lines.append(current)
            current = word
        else:
            current = test
    if current:
        lines.append(current)

    x, y = origin
    line_height = int(cv2.getTextSize("A", font, font_scale, thickness)[0][1] * 2.2)
    for i, line in enumerate(lines[:4]):   # max 4 lines
        cv2.putText(
            frame, line,
            (x, y + i * line_height),
            font, font_scale, color, thickness, cv2.LINE_AA,
        )


def frames_to_video(
    frames: list[np.ndarray],
    output_path: str,
    fps: float = 25.0,
) -> str:
    """Save a list of BGR frames as an mp4 video."""
    if not frames:
        raise ValueError("No frames provided.")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return output_path


def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert PIL (RGB) → OpenCV (BGR)."""
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(frame: np.ndarray) -> Image.Image:
    """Convert OpenCV (BGR) → PIL (RGB)."""
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
