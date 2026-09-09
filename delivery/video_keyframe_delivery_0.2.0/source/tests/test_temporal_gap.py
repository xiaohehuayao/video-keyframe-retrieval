from video_keyframe.schemas import ScoredFrame
from video_keyframe.postprocess.temporal_gap import suppress_by_temporal_gap
from video_keyframe.postprocess.topk import select_topk


def frame(t, score):
    return ScoredFrame(int(t * 30), t, score, score, None, None)


def test_best_score_wins_and_exact_gap_survives():
    frames = [frame(0, 0.4), frame(1, 0.8), frame(3, 0.7), frame(3.1, 0.6)]
    assert [f.timestamp for f in suppress_by_temporal_gap(frames, 2)] == [1, 3]


def test_tie_and_zero_topk():
    frames = [frame(2, 0.5), frame(1, 0.5)]
    assert suppress_by_temporal_gap(frames, 2)[0].timestamp == 1
    assert select_topk(frames, 0) == []
