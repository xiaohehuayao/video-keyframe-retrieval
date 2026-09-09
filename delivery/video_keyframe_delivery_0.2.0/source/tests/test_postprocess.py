"""使用手工构造的帧和分数验证后处理，不需要视频或模型。"""
from video_keyframe.schemas import ScoredFrame
from video_keyframe.postprocess.threshold import filter_by_threshold
from video_keyframe.postprocess.temporal_gap import suppress_by_temporal_gap
from video_keyframe.postprocess.topk import select_topk


def make_frame(timestamp: float, score: float) -> ScoredFrame:
    """假定原视频为 30 FPS；后处理不使用图像，因此 image=None。"""
    return ScoredFrame(
        frame_index=round(timestamp * 30),
        timestamp=timestamp,
        score=score,
        positive_score=score,
        negative_score=None,
        image=None,
    )


def test_threshold_includes_boundary():
    frames = [
        make_frame(0, 0.19),
        make_frame(1, 0.20),
        make_frame(2, 0.21),
    ]

    result = filter_by_threshold(frames, threshold=0.20)

    # 分数等于阈值时也应保留。
    assert [f.timestamp for f in result] == [1, 2]


def test_temporal_gap_prefers_score():
    frames = [
        make_frame(0, 0.60),
        make_frame(1, 0.90),
        make_frame(3, 0.70),
    ]

    result = suppress_by_temporal_gap(frames, min_gap_seconds=2.0)

    # 1 秒处得分更高，优先保留；0 秒处被抑制。
    # 3 秒处与 1 秒处恰好相隔 2 秒，应保留。
    assert [f.timestamp for f in result] == [1, 3]


def test_topk_sorts_before_selecting():
    frames = [
        make_frame(0, 0.30),
        make_frame(5, 0.90),
        make_frame(10, 0.60),
    ]

    result = select_topk(frames, k=2)

    assert [f.timestamp for f in result] == [5, 10]


def test_complete_postprocess():
    frames = [
        make_frame(0, 0.10),
        make_frame(1, 0.90),
        make_frame(2, 0.80),
        make_frame(3, 0.70),
        make_frame(6, 0.60),
    ]

    passed = filter_by_threshold(frames, threshold=0.20)
    assert [f.timestamp for f in passed] == [1, 2, 3, 6]

    kept = suppress_by_temporal_gap(passed, min_gap_seconds=2.0)
    assert [f.timestamp for f in kept] == [1, 3, 6]

    result = select_topk(kept, k=2)
    assert [f.timestamp for f in result] == [1, 3]
    
    #打印
    for frame in result:
      print(
        f"保留帧：帧号={frame.frame_index}, "
        f"时间戳={frame.timestamp}s, 分数={frame.score}"
    )


def test_empty_input():
    passed = filter_by_threshold([], threshold=0.20)
    kept = suppress_by_temporal_gap(passed, min_gap_seconds=2.0)

    assert select_topk(kept, k=10) == []
