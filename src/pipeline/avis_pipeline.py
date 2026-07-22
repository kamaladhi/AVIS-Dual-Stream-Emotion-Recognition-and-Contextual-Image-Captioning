"""
AVIS — Unified End-to-End Pipeline

Single entry point that chains:
    Face Detection → Emotion Recognition → BLIP Captioning → Emotion-Caption Fusion

Usage:
    pipeline = AVISPipeline()
    result = pipeline.run_image(pil_or_bgr_image)
    results = pipeline.run_video_frame(bgr_frame)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Union

import cv2
import numpy as np
from PIL import Image

from src.captioning.blip_wrapper import BLIPCaptioner
from src.detection.face_detector import FaceDetector, FaceDetection
from src.emotion.infer import EmotionInference, EmotionResult
from src.fusion.emotion_caption_fusion import PromptFusionHead
from src.utils.config import config
from src.utils.logger import logger
from src.utils.visualizer import draw_face_results, cv2_to_pil, pil_to_cv2


@dataclass
class FaceAnalysis:
    """Result for one detected face."""
    bbox: tuple[int, int, int, int]
    detection_confidence: float
    emotion: EmotionResult
    base_caption: str
    emotion_caption: str


@dataclass
class AVISResult:
    """Complete result for one image/frame."""
    faces: list[FaceAnalysis] = field(default_factory=list)
    scene_caption: str = ""           # whole-image caption (no face focus)
    agentic_caption: str = ""         # Synthesized context from LangGraph Agent
    annotated_frame: np.ndarray | None = None  # BGR frame with overlays
    processing_time_ms: float = 0.0

    @property
    def num_faces(self) -> int:
        return len(self.faces)

    def to_dict(self) -> dict:
        return {
            "num_faces": self.num_faces,
            "scene_caption": self.scene_caption,
            "agentic_caption": self.agentic_caption,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "faces": [
                {
                    "bbox": list(fa.bbox),
                    "detection_confidence": round(fa.detection_confidence, 3),
                    "emotion": fa.emotion.label,
                    "emotion_confidence": round(fa.emotion.confidence, 3),
                    "emotion_probs": {k: round(v, 3) for k, v in fa.emotion.all_probs.items()},
                    "base_caption": fa.base_caption,
                    "emotion_caption": fa.emotion_caption,
                }
                for fa in self.faces
            ],
        }


class AVISPipeline:
    """
    Affective Visual Intelligence System — main inference pipeline.

    Args:
        lazy_load: If True, models are loaded on first call (saves startup time).
    """

    def __init__(self, lazy_load: bool = False):
        self._face_detector: FaceDetector | None = None
        self._emotion_infer: EmotionInference | None = None
        self._captioner: BLIPCaptioner | None = None
        self._fusion: PromptFusionHead | None = None

        if not lazy_load:
            self._ensure_loaded()

    # ── Lazy loading ──────────────────────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        if self._face_detector is None:
            logger.info("Loading pipeline components …")
            self._face_detector = FaceDetector()
            self._emotion_infer = EmotionInference()
            self._captioner = BLIPCaptioner()
            self._fusion = PromptFusionHead(captioner=self._captioner)
            logger.info("All pipeline components loaded. ✅")

    # ── Public API ────────────────────────────────────────────────────────────

    def run_image(
        self,
        image: Union[Image.Image, np.ndarray],
        annotate: bool = True,
    ) -> AVISResult:
        """
        Full AVIS inference on a single image.

        Args:
            image:    PIL (RGB) or numpy BGR array.
            annotate: If True, draws bounding boxes & captions on the frame.

        Returns:
            AVISResult containing per-face analysis + scene caption.
        """
        self._ensure_loaded()
        t0 = time.perf_counter()

        # ── Convert to consistent formats ─────────────────────
        bgr = pil_to_cv2(image) if isinstance(image, Image.Image) else image
        pil = cv2_to_pil(bgr)

        # ── Stage 1: Face detection ───────────────────────────
        detections: list[FaceDetection] = self._face_detector.detect(bgr)
        logger.debug(f"Detected {len(detections)} face(s).")

        # ── Stage 2: Scene-level caption (whole image) ─────────
        scene_caption = self._captioner.caption(pil)

        # ── Stage 3: Per-face analysis ────────────────────────
        face_analyses: list[FaceAnalysis] = []
        faces_data = []  # For LangGraph agent

        if detections:
            crops = [d.crop for d in detections]
            emotion_results: list[EmotionResult] = self._emotion_infer.predict_batch(crops)

            for detection, emotion in zip(detections, emotion_results):
                # Stage 4: Base caption of face crop
                crop_pil = cv2_to_pil(detection.crop)
                base_cap = self._captioner.caption(crop_pil)

                # Stage 5: Emotion-conditioned caption
                emotion_cap = self._fusion.fuse(
                    base_caption=base_cap,
                    emotion=emotion.label,
                    confidence=emotion.confidence,
                    original_image=pil,
                )

                face_analyses.append(FaceAnalysis(
                    bbox=detection.bbox,
                    detection_confidence=detection.confidence,
                    emotion=emotion,
                    base_caption=base_cap,
                    emotion_caption=emotion_cap,
                ))
                
                faces_data.append({
                    "emotion": emotion.label,
                    "confidence": emotion.confidence,
                    "bbox": detection.bbox
                })

        # ── Stage 5.5: LangGraph Agentic Synthesis ────────────
        from src.agents import agent_app
        
        try:
            agent_input = {
                "image_path": "memory",
                "base_caption": scene_caption,
                "faces_data": faces_data,
                "final_caption": ""
            }
            agent_result = agent_app.invoke(agent_input)
            agentic_caption = agent_result["final_caption"]
        except Exception as e:
            logger.error(f"LangGraph Agent invocation failed: {e}")
            agentic_caption = scene_caption

        # ── Stage 6: Annotate frame ───────────────────────────
        annotated = None
        if annotate:
            face_dicts = [
                {
                    "bbox": fa.bbox,
                    "emotion": fa.emotion.label,
                    "confidence": fa.emotion.confidence,
                    "caption": fa.emotion_caption,
                }
                for fa in face_analyses
            ]
            annotated = draw_face_results(bgr.copy(), face_dicts)

        elapsed_ms = (time.perf_counter() - t0) * 1000
        
        if face_analyses:
            face_details = ", ".join([f"Face {i+1}: {fa.emotion.label.upper()} ({fa.emotion.confidence*100:.1f}%)" for i, fa in enumerate(face_analyses)])
            logger.info(f"Pipeline complete in {elapsed_ms:.1f}ms — Detected {len(face_analyses)} face(s) -> {face_details}")
        else:
            logger.info(f"Pipeline complete in {elapsed_ms:.1f}ms — 0 face(s)")

        return AVISResult(
            faces=face_analyses,
            scene_caption=scene_caption,
            agentic_caption=agentic_caption,
            annotated_frame=annotated,
            processing_time_ms=elapsed_ms,
        )

    def run_video_frame(self, bgr_frame: np.ndarray) -> AVISResult:
        """Alias for run_image — optimized for video stream usage."""
        return self.run_image(bgr_frame, annotate=True)

    def run_webcam(self, camera_id: int = 0, show_window: bool = True) -> None:
        """
        Real-time webcam inference loop.
        Press 'q' to quit.
        """
        self._ensure_loaded()
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")

        logger.info("Webcam stream started. Press 'q' to quit.")
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                # Run inference every 3rd frame to keep smooth FPS
                if frame_count % 3 == 0:
                    result = self.run_video_frame(frame)
                    display = result.annotated_frame if result.annotated_frame is not None else frame
                else:
                    display = frame

                if show_window:
                    cv2.imshow("AVIS — Affective Visual Intelligence", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        finally:
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Webcam stream stopped.")


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description="AVIS — Affective Visual Intelligence System")
    parser.add_argument("--image", help="Path to input image")
    parser.add_argument("--video", help="Path to input video file")
    parser.add_argument("--webcam", action="store_true", help="Run on webcam")
    parser.add_argument("--output", help="Path to save annotated output")
    args = parser.parse_args()

    pipeline = AVISPipeline()

    if args.webcam:
        pipeline.run_webcam()

    elif args.image:
        img = Image.open(args.image).convert("RGB")
        result = pipeline.run_image(img)
        print(json.dumps(result.to_dict(), indent=2))
        if args.output and result.annotated_frame is not None:
            cv2.imwrite(args.output, result.annotated_frame)
            print(f"Annotated image saved to {args.output}")

    elif args.video:
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_writer = None
        if args.output:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_writer = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if frame_idx % 3 == 0:  # process every 3rd frame
                result = pipeline.run_video_frame(frame)
                out_frame = result.annotated_frame or frame
            else:
                out_frame = frame
            if out_writer:
                out_writer.write(out_frame)

        cap.release()
        if out_writer:
            out_writer.release()
            print(f"Annotated video saved to {args.output}")
    else:
        parser.print_help()
