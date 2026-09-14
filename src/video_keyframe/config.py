"""类型化配置及 YAML 加载；未知字段会被拒绝。"""
from dataclasses import dataclass, field
from pathlib import Path
from math import isfinite
from .exceptions import ConfigurationError


@dataclass
class ModelConfig:
    name: str = "google/siglip2-base-patch16-224"
    device: str = "cuda"
    dtype: str = "float16"
    batch_size: int = 32


@dataclass
class SamplingConfig:
    sample_fps: float | None = 1.0
    sample_stride: int | None = None
    max_video_duration: float | None = None
    decode_backend: str = "opencv"


@dataclass
class ScoringConfig:
    candidate_quantile: float = 0.90
    negative_weight: float = 0.0


@dataclass
class MultiAnchorConfig:
    enabled: bool = True
    min_anchor_gap_seconds: float = 5.0
    max_anchor_count: int = 5
    relative_ratio: float = 0.8
    max_peak_radius_seconds: float = 2.0
    max_frames_per_peak: int = 5


@dataclass
class VideoSegmentConfig:
    enabled: bool = False
    max_anchor_count: int = 2
    relative_ratio: float = 0.7
    max_peak_radius_seconds: float = 10.0
    merge_overlapping_segments: bool = True


@dataclass
class VideoOutputConfig:
    export_clips: bool = False
    ffmpeg_path: str = "ffmpeg"


@dataclass
class PostprocessConfig:
    p2_multi_anchor: MultiAnchorConfig = field(default_factory=MultiAnchorConfig)
    min_match_frame_gap: float = 2.0
    max_match_count: int = 10
    p3_video: VideoSegmentConfig = field(default_factory=VideoSegmentConfig)


@dataclass
class OutputConfig:
    jpeg_quality: int = 95
    include_score: bool = True
    include_frame_index: bool = True
    video: VideoOutputConfig = field(default_factory=VideoOutputConfig)


@dataclass
class OperatorConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    postprocess: PostprocessConfig = field(default_factory=PostprocessConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        def number(name, value, minimum, strict=False):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not isfinite(value) or (value <= minimum if strict else value < minimum)):
                raise ConfigurationError(f"{name} 数值无效: {value!r}")

        for name, value, minimum in [
            ("batch_size", self.model.batch_size, 1),
            ("max_match_count", self.postprocess.max_match_count, 0),
            ("jpeg_quality", self.output.jpeg_quality, 1),
        ]:
            if type(value) is not int or value < minimum:
                raise ConfigurationError(f"{name} 必须是 >= {minimum} 的整数")
        p2 = self.postprocess.p2_multi_anchor
        if not isinstance(p2, MultiAnchorConfig) or type(p2.enabled) is not bool:
            raise ConfigurationError("p2_multi_anchor must contain a boolean enabled")
        for name, minimum in (("max_anchor_count", 0), ("max_frames_per_peak", 1)):
            value = getattr(p2, name)
            if type(value) is not int or value < minimum:
                raise ConfigurationError(f"{name} must be an integer >= {minimum}")
        for name in ("min_anchor_gap_seconds", "relative_ratio", "max_peak_radius_seconds"):
            number(name, getattr(p2, name), 0)
        if p2.relative_ratio > 1:
            raise ConfigurationError("relative_ratio must be in [0, 1]")
        p3, video = self.postprocess.p3_video, self.output.video
        if not isinstance(p3, VideoSegmentConfig) or not isinstance(video, VideoOutputConfig):
            raise ConfigurationError("Invalid P3 configuration objects")
        for value in (p3.enabled, p3.merge_overlapping_segments, video.export_clips):
            if type(value) is not bool:
                raise ConfigurationError("P3 switches must be booleans")
        if type(p3.max_anchor_count) is not int or p3.max_anchor_count < 0:
            raise ConfigurationError("p3 max_anchor_count must be a nonnegative integer")
        number("p3 relative_ratio", p3.relative_ratio, 0)
        number("p3 max_peak_radius_seconds", p3.max_peak_radius_seconds, 0)
        if p3.relative_ratio > 1:
            raise ConfigurationError("p3 relative_ratio must be <= 1")
        if p3.enabled and not p2.enabled:
            raise ConfigurationError("P3 requires P2 enabled")
        if video.export_clips and not p3.enabled:
            raise ConfigurationError("Video export requires P3 enabled")
        if not isinstance(video.ffmpeg_path, str) or not video.ffmpeg_path.strip():
            raise ConfigurationError("ffmpeg_path must be a nonempty string")
        if self.output.jpeg_quality > 100:
            raise ConfigurationError("jpeg_quality 必须 <= 100")
        if not isinstance(self.model.name, str) or not self.model.name.strip():
            raise ConfigurationError("model.name 不能为空")
        if self.model.device not in ("cpu", "cuda", "auto"):
            raise ConfigurationError("device 必须为 cpu/cuda/auto")
        if self.model.dtype not in ("float32", "float16", "bfloat16", "fp32", "fp16", "bf16"):
            raise ConfigurationError("不支持的 dtype")
        if self.sampling.decode_backend != "opencv":
            raise ConfigurationError("P0 仅支持 opencv 解码")
        if self.sampling.sample_stride is not None:
            if type(self.sampling.sample_stride) is not int or self.sampling.sample_stride < 1:
                raise ConfigurationError("sample_stride 必须是正整数")
        elif self.sampling.sample_fps is None:
            raise ConfigurationError("必须设置 sample_fps 或 sample_stride")
        if self.sampling.sample_fps is not None:
            number("sample_fps", self.sampling.sample_fps, 0, True)
        if self.sampling.max_video_duration is not None:
            number("max_video_duration", self.sampling.max_video_duration, 0, True)
        number("candidate_quantile", self.scoring.candidate_quantile, 0)
        if self.scoring.candidate_quantile > 1:
            raise ConfigurationError("candidate_quantile 必须在 0..1 之间")
        number("negative_weight", self.scoring.negative_weight, 0)
        number("min_match_frame_gap", self.postprocess.min_match_frame_gap, 0)
        if type(self.output.include_score) is not bool or type(self.output.include_frame_index) is not bool:
            raise ConfigurationError("include_score/include_frame_index 必须是布尔值")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "OperatorConfig":
        import yaml
        try:
            with open(path, encoding="utf-8") as handle:
                data = yaml.safe_load(handle)
            if data is None:
                data = {}
            types = {"model": ModelConfig, "sampling": SamplingConfig,
                     "scoring": ScoringConfig, "postprocess": PostprocessConfig,
                     "output": OutputConfig}
            if "postprocess" in data and "p2_multi_anchor" in data["postprocess"]:
                data["postprocess"]["p2_multi_anchor"] = MultiAnchorConfig(
                    **data["postprocess"]["p2_multi_anchor"])
            if "postprocess" in data and "p3_video" in data["postprocess"]:
                data["postprocess"]["p3_video"] = VideoSegmentConfig(**data["postprocess"]["p3_video"])
            if "output" in data and "video" in data["output"]:
                data["output"]["video"] = VideoOutputConfig(**data["output"]["video"])
            config = cls(**{key: types[key](**value) for key, value in data.items()})
            config.validate()
            return config
        except (TypeError, KeyError, AttributeError, yaml.YAMLError) as exc:
            raise ConfigurationError(f"YAML 配置无效: {exc}") from exc
