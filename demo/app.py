"""
AVIS — Gradio Demo Application

Three tabs:
    1. 📸 Image Upload    — Upload any image, see emotion + caption results
    2. 📹 Video Upload    — Process a video file, download annotated output
    3. 📷 Live Webcam     — Real-time webcam inference

Run:
    python demo/app.py
"""

from __future__ import annotations

import io
import json

# ── Lazy pipeline load ────────────────────────────────────────────────────────
import sys
import tempfile
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.avis_pipeline import AVISPipeline
from src.utils.config import config
from src.utils.visualizer import cv2_to_pil, pil_to_cv2

_pipeline: AVISPipeline | None = None


def get_pipeline() -> AVISPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = AVISPipeline(lazy_load=False)
    return _pipeline


# ── Emotion colour map for display ───────────────────────────────────────────
EMOTION_COLORS = {
    "angry": "#ef4444",
    "disgust": "#22c55e",
    "fear": "#a855f7",
    "happy": "#f59e0b",
    "neutral": "#94a3b8",
    "sad": "#3b82f6",
    "surprise": "#06b6d4",
    "contempt": "#f97316",
}


# ── Tab 1: Image inference ────────────────────────────────────────────────────
def process_image(image: np.ndarray) -> tuple:
    """Gradio callback for image tab."""
    if image is None:
        return None, "No image provided.", "{}"

    pil = Image.fromarray(image)
    result = get_pipeline().run_image(pil, annotate=True)

    annotated_rgb = (
        cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
        if result.annotated_frame is not None
        else image
    )

    # ── Build markdown summary ────────────────────────────────
    md = f"### 🌍 Scene Caption\n> {result.scene_caption}\n\n"
    md += f"**Faces detected:** {result.num_faces} | "
    md += f"**Processing time:** {result.processing_time_ms:.0f} ms\n\n"

    for i, face in enumerate(result.faces, 1):
        color = EMOTION_COLORS.get(face.emotion.label, "#94a3b8")
        md += f"---\n#### Face {i}\n"
        md += (
            f'<span style="background:{color};color:white;padding:4px 10px;'
            f'border-radius:12px;font-weight:bold">'
            f"{face.emotion.label.upper()} {face.emotion.confidence:.0%}</span>\n\n"
        )
        md += f"**Base caption:** {face.base_caption}\n\n"
        md += f"**Emotion caption:** _{face.emotion_caption}_\n\n"
        # Confidence bar
        md += "**Emotion probabilities:**\n"
        for emo, prob in sorted(face.emotion.all_probs.items(), key=lambda x: -x[1]):
            bar_len = int(prob * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            md += f"`{emo:<10}` {bar} {prob:.1%}\n"

    return annotated_rgb, md, json.dumps(result.to_dict(), indent=2)


# ── Tab 2: Video inference ────────────────────────────────────────────────────
def process_video(video_path: str) -> tuple:
    """Gradio callback for video tab."""
    if video_path is None:
        return None, "No video provided."

    pipeline = get_pipeline()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        out_path = tmp.name

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    frame_idx, processed = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % 3 == 0:
            result = pipeline.run_video_frame(frame)
            out_frame = result.annotated_frame if result.annotated_frame is not None else frame
            processed += 1
        else:
            out_frame = frame
        writer.write(out_frame)

    cap.release()
    writer.release()

    return out_path, f"Processed {processed} frames from {frame_idx} total."


# ── Tab 3: Webcam ─────────────────────────────────────────────────────────────
def process_webcam_frame(frame: np.ndarray) -> np.ndarray:
    """Gradio streaming callback for webcam tab."""
    if frame is None:
        return frame
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    result = get_pipeline().run_video_frame(bgr)
    if result.annotated_frame is not None:
        return cv2.cvtColor(result.annotated_frame, cv2.COLOR_BGR2RGB)
    return frame


# ── Build Gradio app ──────────────────────────────────────────────────────────


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="AVIS — Affective Visual Intelligence System",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
        ),
        css="""
        .title-bar { text-align: center; padding: 20px 0; }
        .title-bar h1 { font-size: 2.2rem; font-weight: 800; 
                         background: linear-gradient(135deg, #6366f1, #a855f7);
                         -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .subtitle { color: #64748b; font-size: 1rem; }
        .result-box { border-radius: 12px; padding: 16px; }
        """,
    ) as demo:

        # ── Header ────────────────────────────────────────────
        gr.HTML("""
        <div class="title-bar">
            <h1>🧠 AVIS — Affective Visual Intelligence System</h1>
            <p class="subtitle">
                Emotion-aware image captioning · YOLOv8 · EfficientNet-B4 + MediaPipe Dual-Stream · BLIP-large
            </p>
        </div>
        """)

        with gr.Tabs():
            # ── Tab 1: Image ───────────────────────────────────
            with gr.Tab("📸 Image Analysis"):
                with gr.Row():
                    with gr.Column(scale=1):
                        img_input = gr.Image(
                            label="Upload Image",
                            type="numpy",
                            elem_id="img_upload",
                        )
                        img_btn = gr.Button("🔍 Analyse", variant="primary", size="lg")

                    with gr.Column(scale=1):
                        img_output = gr.Image(label="Annotated Result", type="numpy")

                with gr.Row():
                    with gr.Column():
                        img_summary = gr.Markdown(
                            label="Analysis Summary", elem_classes=["result-box"]
                        )
                    with gr.Column():
                        img_json = gr.Code(language="json", label="Raw JSON Output", lines=12)

                gr.Examples(
                    examples=[],  # Add local example images here
                    inputs=img_input,
                )

                img_btn.click(
                    fn=process_image,
                    inputs=img_input,
                    outputs=[img_output, img_summary, img_json],
                )

            # ── Tab 2: Video ───────────────────────────────────
            with gr.Tab("🎬 Video Processing"):
                with gr.Row():
                    with gr.Column():
                        vid_input = gr.Video(label="Upload Video")
                        vid_btn = gr.Button("▶️ Process Video", variant="primary", size="lg")
                    with gr.Column():
                        vid_output = gr.Video(label="Annotated Video")
                        vid_status = gr.Textbox(label="Status", interactive=False)

                vid_btn.click(
                    fn=process_video,
                    inputs=vid_input,
                    outputs=[vid_output, vid_status],
                )

            # ── Tab 3: Webcam ──────────────────────────────────
            with gr.Tab("📷 Live Webcam"):
                gr.Markdown(
                    "**Real-time emotion recognition & captioning.** "
                    "Allow browser camera access, then click Start."
                )
                with gr.Row():
                    webcam_in = gr.Image(
                        sources=["webcam"],
                        streaming=True,
                        label="Webcam Feed",
                        type="numpy",
                    )
                    webcam_out = gr.Image(label="AVIS Output", type="numpy")

                webcam_in.stream(
                    fn=process_webcam_frame,
                    inputs=webcam_in,
                    outputs=webcam_out,
                    time_limit=300,
                )

        # ── Footer ────────────────────────────────────────────
        gr.Markdown(
            "---\n"
            "**AVIS v1.0** · Built as part of a production-grade CV portfolio project · "
            "[GitHub](https://github.com/kamaladhi/Image-captioning-with-facial-expression)"
        )

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
