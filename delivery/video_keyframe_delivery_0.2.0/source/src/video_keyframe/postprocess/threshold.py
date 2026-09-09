from typing import TypeVar
import numpy as np
from ..schemas import FrameScore, ScoredFrame

T = TypeVar("T", FrameScore, ScoredFrame)


def filter_by_threshold(scored_frames: list[T], threshold: float) -> list[T]:
    return [frame for frame in scored_frames if frame.score >= threshold]


def filter_by_quantile(scored_frames: list[T], quantile: float) -> tuple[list[T], float | None]:
    """对全局记录计算线性分位数；同分全部保留，空输入阈值为 None。"""
    if isinstance(quantile, bool) or not np.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError("quantile 必须为 0..1 的有限数值")
    if not scored_frames:
        return [], None
    scores = np.asarray([frame.score for frame in scored_frames], dtype=np.float64)
    if not np.isfinite(scores).all():
        raise ValueError("score 必须为有限数值")
    threshold = float(np.quantile(scores, quantile, method="linear"))
    return filter_by_threshold(scored_frames, threshold), threshold
