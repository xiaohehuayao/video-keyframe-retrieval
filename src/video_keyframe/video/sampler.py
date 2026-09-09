"""流式采样；stride 优先，FPS 采样使用时间网格避免取整累计漂移。"""
from collections.abc import Iterable, Iterator
from math import isfinite, floor
from ..schemas import SampledFrame


class FrameSampler:
    def sample(self, frames: Iterable[SampledFrame], fps: float,
               sample_stride: int | None = None,
               sample_fps: float | None = None) -> Iterator[SampledFrame]:
        if not isfinite(fps) or fps <= 0:
            raise ValueError("fps 必须为正数")
        if sample_stride is not None:
            if type(sample_stride) is not int or sample_stride < 1:
                raise ValueError("sample_stride 必须为正整数")
            for frame in frames:
                if frame.frame_index % sample_stride == 0:
                    yield frame
            return
        rate = 1.0 if sample_fps is None else sample_fps
        if not isfinite(rate) or rate <= 0:
            raise ValueError("sample_fps 必须为正数")
        rate = min(rate, fps)
        next_tick = 0
        for frame in frames:
            if frame.timestamp + 1e-9 >= next_tick / rate:
                yield frame
                next_tick = floor(frame.timestamp * rate + 1e-9) + 1
