from video_keyframe.schemas import SampledFrame
from video_keyframe.video.sampler import FrameSampler


def test_one_fps():
    frames = (SampledFrame(i, i / 30, None) for i in range(90))
    assert [f.frame_index for f in FrameSampler().sample(frames, 30, sample_fps=1)] == [0, 30, 60]


def test_fractional_fps_does_not_accumulate_drift():
    frames = (SampledFrame(i, i / 29.97, None) for i in range(3000))
    sampled = list(FrameSampler().sample(frames, 29.97, sample_fps=1))
    assert len(sampled) == 101
    assert all(0 <= f.timestamp - tick < 1 / 29.97 for tick, f in enumerate(sampled))


def test_stride_overrides_fps():
    frames = [SampledFrame(i, i / 30, None) for i in range(10)]
    assert [f.frame_index for f in FrameSampler().sample(frames, 30, 3, 1)] == [0, 3, 6, 9]
