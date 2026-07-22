"""
AVIS — Centralized Configuration
All settings are loaded from environment variables (.env file).
"""
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parent.parent.parent


class AVISConfig(BaseSettings):
    """Single source of truth for all AVIS configuration."""

    model_config = SettingsConfigDict(
        env_file=ROOT_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Device ────────────────────────────────────────────────
    device: Literal["cuda", "cpu"] = "cuda"
    blip_precision: Literal["float16", "float32"] = "float16"

    # ── HuggingFace ───────────────────────────────────────────
    hf_token: str = ""

    # ── Model Identifiers ─────────────────────────────────────
    blip_model_id: str = "Salesforce/blip-image-captioning-large"
    yolo_model_name: str = "yolov8n.pt"   # will be overridden by face model

    # ── Model Paths ───────────────────────────────────────────
    emotion_model_path: Path = ROOT_DIR / "weights" / "emotion_model_best.pth"
    yolo_model_path: Path = ROOT_DIR / "weights" / "yolov8n-face.pt"
    weights_dir: Path = ROOT_DIR / "weights"
    results_dir: Path = ROOT_DIR / "results"

    # ── Inference Thresholds ──────────────────────────────────
    emotion_confidence_threshold: float = 0.45
    face_detection_confidence: float = 0.5
    max_caption_length: int = 50
    min_caption_length: int = 10

    # ── Emotion Classes ───────────────────────────────────────
    emotion_classes: list[str] = [
        "angry", "disgust", "fear", "happy",
        "neutral", "sad", "surprise"
    ]
    num_emotion_classes: int = 7

    # ── Training ──────────────────────────────────────────────
    batch_size: int = 32
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    dropout_rate: float = 0.4
    label_smoothing: float = 0.1
    num_workers: int = 4

    # ── EfficientNet Dual-Stream ──────────────────────────────
    efficientnet_variant: str = "efficientnet_b4"
    efficientnet_out_features: int = 1792
    landmark_input_dim: int = 936       # 468 points × 2 (x, y)
    landmark_hidden_dim: int = 256
    fusion_hidden_dim: int = 512
    emotion_embed_dim: int = 128        # for caption fusion

    # ── API ───────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # ── MLflow ────────────────────────────────────────────────
    mlflow_tracking_uri: str = "sqlite:///mlflow.db"
    mlflow_experiment_name: str = "avis-emotion-recognition"

    # ── LangSmith ─────────────────────────────────────────────
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "avis-emotion-captioning"
    
    # ── LLM Configuration ─────────────────────────────────────
    groq_api_key: str = ""
    openrouter_api_key: str = ""
    llm_model_name: str = "llama-3.1-8b-instant"

    def ensure_dirs(self):
        """Create required directories if they don't exist."""
        for d in [self.weights_dir, self.results_dir]:
            d.mkdir(parents=True, exist_ok=True)


# Singleton instance — import this everywhere
config = AVISConfig()
config.ensure_dirs()
