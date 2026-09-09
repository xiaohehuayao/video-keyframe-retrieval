"""统一数据契约。image 为 RGB uint8 ndarray，时间单位为秒。"""
from dataclasses import dataclass
from typing import Any


@dataclass
class VideoMetadata:
    fps: float
    frame_count: int
    duration: float


@dataclass
class SampledFrame:
    frame_index: int
    timestamp: float
    image: Any


@dataclass
class ScoredFrame:
    frame_index: int
    timestamp: float
    score: float
    positive_score: float
    negative_score: float | None
    image: Any


@dataclass
class FrameScore:
    """首遍仅保留分数和定位信息，不持有 RGB 图片。"""
    frame_index: int
    timestamp: float
    score: float
    positive_score: float
    negative_score: float | None = None


@dataclass
class KeyframeResult:
    timestamp: float
    score: float
    jpg_bytes: bytes
    frame_index: int | None = None


@dataclass
class OperatorResult:
    keyframes: list[KeyframeResult]
    meta: dict[str, Any]


@dataclass
class RuntimeState:
    video_fps: float = 0.0
    video_duration: float = 0.0
    total_frame_count: int = 0
    decoded_frame_count: int = 0
    sampled_frame_count: int = 0
    image_batch_count: int = 0
    image_encoded_count: int = 0
    text_encoded_count: int = 0
    scored_frame_count: int = 0
    threshold_pass_count: int = 0
    candidate_count: int = 0
    effective_threshold: float | None = None
    reread_frame_count: int = 0
    reread_time_ms: float = 0.0
    jpeg_encode_time_ms: float = 0.0
    temporal_suppressed_count: int = 0
    topk_trimmed_count: int = 0
    returned_count: int = 0
    negative_prompt_enabled: bool = False
    decode_time_ms: float = 0.0
    image_encode_time_ms: float = 0.0
    text_encode_time_ms: float = 0.0
    scoring_time_ms: float = 0.0
    postprocess_time_ms: float = 0.0
    total_time_ms: float = 0.0
