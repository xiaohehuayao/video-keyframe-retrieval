from typing import TypeVar
from ..schemas import FrameScore, ScoredFrame

T = TypeVar("T", FrameScore, ScoredFrame)


def select_topk(frames: list[T], k: int) -> list[T]:
    if type(k) is not int or k < 0:
        raise ValueError("k 必须是非负整数")
    return sorted(frames, key=lambda f: (-f.score, f.timestamp, f.frame_index))[:k]
