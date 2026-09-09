from bisect import bisect_left, insort
from math import isfinite
from typing import TypeVar
from ..schemas import FrameScore, ScoredFrame

T = TypeVar("T", FrameScore, ScoredFrame)


def suppress_by_temporal_gap(frames: list[T], min_gap_seconds: float) -> list[T]:
    """分数优先贪心；同分时较早帧优先；恰好等于 gap 时保留。"""
    if not isfinite(min_gap_seconds) or min_gap_seconds < 0:
        raise ValueError("min_gap_seconds 必须为非负有限值")
    selected, timestamps = [], []
    for frame in sorted(frames, key=lambda f: (-f.score, f.timestamp, f.frame_index)):
        index = bisect_left(timestamps, frame.timestamp)
        if index > 0 and frame.timestamp - timestamps[index - 1] < min_gap_seconds:
            continue
        if index < len(timestamps) and timestamps[index] - frame.timestamp < min_gap_seconds:
            continue
        selected.append(frame)
        insort(timestamps, frame.timestamp)
    return selected
