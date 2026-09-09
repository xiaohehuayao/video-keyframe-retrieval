"""平台初始化入口：显式指定本地权重，加载一次后复用算子。"""
from pathlib import Path

from .config import OperatorConfig
from .exceptions import ConfigurationError
from .models.siglip2 import SigLIP2Model
from .operator import VideoKeyframeOperator


def create_operator(
    model_path: str | Path,
    *,
    device: str = "cpu",
    dtype: str = "float32",
    batch_size: int = 32,
    sample_fps: float = 1.0,
    candidate_quantile: float = 0.90,
    min_match_frame_gap: float = 2.0,
    max_match_count: int = 10,
    max_video_duration: float | None = None,
    jpeg_quality: int = 95,
) -> VideoKeyframeOperator:
    """加载本地 SigLIP2 并返回可复用算子；不依赖 YAML 或项目工作目录。

    model_path 在初始化时解析成绝对路径。此入口只配置正向检索。
    加载错误直接抛给调用方；同一实例的 run 不承诺并发安全。
    """
    path = Path(model_path).expanduser().resolve()
    if not path.is_dir():
        raise ConfigurationError(f"本地模型目录不存在：{path}")
    config = OperatorConfig()
    config.model.name = str(path)
    config.model.device = device
    config.model.dtype = dtype
    config.model.batch_size = batch_size
    config.sampling.sample_fps = sample_fps
    config.sampling.max_video_duration = max_video_duration
    config.scoring.candidate_quantile = candidate_quantile
    config.postprocess.min_match_frame_gap = min_match_frame_gap
    config.postprocess.max_match_count = max_match_count
    config.output.jpeg_quality = jpeg_quality
    config.validate()
    model = SigLIP2Model(
        model_name=config.model.name,
        device=device,
        dtype=dtype,
        batch_size=batch_size,
    )
    return VideoKeyframeOperator(model=model, config=config)
