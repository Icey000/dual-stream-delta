import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dual_qformer import DualVideo2CaptionLLM


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _required_path(name: str) -> Path:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    path = Path(value).expanduser()
    if not path.exists():
        raise RuntimeError(f"Configured path does not exist for {name}: {path}")
    return path


def _required_file(name: str) -> Path:
    path = _required_path(name)
    if not path.is_file():
        raise RuntimeError(f"Configured file is not a regular file for {name}: {path}")
    return path


def _extract_centered_clip(features: np.ndarray, frame_index: int, clip_length: int) -> np.ndarray:
    if features.ndim != 2:
        raise ValueError(f"Expected 2D features shaped [T, D], got shape {features.shape}")

    l_pad = clip_length // 2 + clip_length % 2
    r_pad = clip_length // 2
    original_length = int(features.shape[0])
    padded_length = original_length + l_pad + r_pad
    start = min(max(int(frame_index), 0), max(0, padded_length - clip_length))
    real_start = start - l_pad
    real_end = real_start + clip_length
    indices = np.clip(np.arange(real_start, real_end), 0, max(0, original_length - 1))
    return np.asarray(features[indices], dtype=np.float32)


class PredictRequest(BaseModel):
    match_id: str = Field(
        ...,
        description="Relative SoccerNet match path, e.g. 'england_epl/2016-2017/Arsenal_Chelsea'.",
    )
    half: int = Field(..., ge=1, le=2, description="Match half, either 1 or 2.")
    timestamp_seconds: int = Field(
        ...,
        ge=0,
        description="Timestamp in seconds within the selected half.",
    )
    max_new_tokens: int = Field(default=48, ge=1, le=128)
    num_beams: int | None = Field(default=None, ge=1, le=8)
    temperature: float | None = Field(default=None, gt=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)


class PredictResponse(BaseModel):
    match_id: str
    half: int
    timestamp_seconds: int
    caption: str
    device: str
    video_shape: list[int]
    audio_shape: list[int]


class ModelServer:
    def __init__(self) -> None:
        self.device = os.getenv(
            "MODEL_DEVICE",
            "cuda" if torch.cuda.is_available() else "cpu",
        )
        self.model: DualVideo2CaptionLLM | None = None

    def load(self) -> DualVideo2CaptionLLM:
        if self.model is not None:
            return self.model

        llm_model_path = os.getenv("LLM_MODEL_PATH", "Qwen/Qwen2.5-7B")
        checkpoint_path = _required_file("CAPTION_CHECKPOINT_PATH")
        video_input_dim = int(os.getenv("VIDEO_INPUT_DIM", "1024"))
        audio_input_dim = int(os.getenv("AUDIO_INPUT_DIM", "512"))
        video_tokens = int(os.getenv("VIDEO_TOKENS", "8"))
        audio_tokens = int(os.getenv("AUDIO_TOKENS", "8"))

        model = DualVideo2CaptionLLM(
            video_input_dim=video_input_dim,
            audio_input_dim=audio_input_dim,
            video_tokens=video_tokens,
            audio_tokens=audio_tokens,
            llm_model_path=llm_model_path,
            weights=str(checkpoint_path),
        )
        model.eval()
        model.to(self.device)
        self.model = model
        return model


class FeatureStore:
    def __init__(self) -> None:
        self.vision_root = _required_path("VISION_FEATURE_ROOT")
        self.audio_root = _required_path("AUDIO_FEATURE_ROOT")
        self.vision_feature_file = os.getenv("VISION_FEATURE_FILE", "baidu_soccer_embeddings.npy")
        self.framerate = int(os.getenv("FRAMERATE", "1"))
        self.window_size_seconds = int(os.getenv("WINDOW_SIZE_SECONDS", "30"))
        self.window_size_frames = self.window_size_seconds * self.framerate

    def load_event_features(self, match_id: str, half: int, timestamp_seconds: int) -> tuple[np.ndarray, np.ndarray]:
        frame_index = self.framerate * int(timestamp_seconds)
        vision_path = self.vision_root / match_id / f"{half}_{self.vision_feature_file}"
        audio_path = self.audio_root / match_id / f"{half}_audio_clap.npy"

        if not vision_path.is_file():
            raise FileNotFoundError(f"Missing vision features: {vision_path}")
        if not audio_path.is_file():
            raise FileNotFoundError(f"Missing audio features: {audio_path}")

        video_features = np.load(vision_path, mmap_mode="r")
        audio_features = np.load(audio_path, mmap_mode="r")

        if video_features.ndim > 2:
            video_features = video_features.reshape(-1, video_features.shape[-1])
        if audio_features.ndim > 2:
            audio_features = audio_features.reshape(-1, audio_features.shape[-1])

        video_clip = _extract_centered_clip(video_features, frame_index, self.window_size_frames)
        audio_clip = _extract_centered_clip(audio_features, frame_index, self.window_size_frames)
        return video_clip, audio_clip


def _build_generation_config(payload: PredictRequest) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if payload.num_beams is not None:
        config["num_beams"] = payload.num_beams
    if payload.temperature is not None:
        config["temperature"] = payload.temperature
    if payload.top_p is not None:
        config["top_p"] = payload.top_p
    if payload.temperature is not None or payload.top_p is not None:
        config["do_sample"] = True
    return config


server = ModelServer()
app = FastAPI(
    title="Dual-Stream Qwen Inference API",
    description="Minimal deployment wrapper for dense video caption generation from pre-extracted features.",
    version="0.2.0",
)


@app.on_event("startup")
def maybe_preload_model() -> None:
    if _env_flag("PRELOAD_MODEL", default=False):
        server.load()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": server.device,
        "model_loaded": server.model is not None,
        "uses_preextracted_features": True,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        feature_store = FeatureStore()
        video_clip, audio_clip = feature_store.load_event_features(
            match_id=payload.match_id,
            half=payload.half,
            timestamp_seconds=payload.timestamp_seconds,
        )
        model = server.load()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    video = torch.tensor(video_clip, dtype=torch.float32, device=server.device)
    audio = torch.tensor(audio_clip, dtype=torch.float32, device=server.device)
    generation_config = _build_generation_config(payload)

    with torch.inference_mode():
        caption = model.sample(
            video,
            audio,
            max_seq_length=payload.max_new_tokens,
            generation_config=generation_config or None,
        )

    return PredictResponse(
        match_id=payload.match_id,
        half=payload.half,
        timestamp_seconds=payload.timestamp_seconds,
        caption=caption,
        device=server.device,
        video_shape=list(video.shape),
        audio_shape=list(audio.shape),
    )
