import json
from dataclasses import replace

import pytest

from video_keyframe.config import OperatorConfig
from video_keyframe.exceptions import ConfigurationError
from video_keyframe.schemas import FrameScore
from video_keyframe.postprocess.peak_detection import detect_peaks
from video_keyframe.postprocess.anchor_selection import select_anchors
from video_keyframe.postprocess.peak_expansion import expand_peaks
from video_keyframe import VideoKeyframeOperator
from test_operator import video, FakeModel


def curve(scores, times=None):
    if times is None:
        times = range(len(scores))
    return [FrameScore(i, t, s, s) for i, (t, s) in enumerate(zip(times, scores))]


def test_wide_peak():
    frames = curve([.04, .10, .15, .18, .17, .14, .08])
    peaks = detect_peaks(frames)
    assert [p.timestamp for p in peaks] == [3]
    anchors = select_anchors(peaks, 5, 5)
    selected, meta = expand_peaks(frames, anchors, .8, 2, 5)
    assert [f.timestamp for f in selected] == [2, 3, 4]
    assert meta[0]["expanded_frame_count"] == 3


def test_far_events_and_frame_topk_bias():
    frames = curve([.20, .19, .18, .17, .16, .02, .15],
                   [20, 20.5, 21, 21.5, 22, 50, 100])
    assert [a.timestamp for a in select_anchors(detect_peaks(frames), 5, 2)] == [20, 100]


def test_close_peaks_and_exact_gap():
    peaks = curve([.20, .19, .16, .15], [21, 23, 62, 100])
    assert [a.timestamp for a in select_anchors(peaks, 5, 3)] == [21, 62, 100]
    assert len(select_anchors(curve([.2, .1], [0, 5]), 5, 2)) == 2


def test_peak_topk():
    frames = curve([v for i in range(10) for v in (0, (i + 1) / 10)] + [0])
    peaks = detect_peaks(frames)
    assert len(peaks) == 10
    assert [a.score for a in select_anchors(peaks, 0, 3)] == [1, .9, .8]
    assert select_anchors(peaks, 0, 0) == []


@pytest.mark.parametrize("scores,expected", [
    ([], []), ([1], [0]), ([1, 2, 3], [2]), ([3, 2, 1], [0]),
    ([1, 2, 2, 1], [1]), ([2, 2, 2], [1]),
    ([2, 1, 2, 1, 2], [0, 2, 4]), ([2, 1, 1, 2], [0, 3]),
])
def test_boundaries_plateaus(scores, expected):
    assert [p.frame_index for p in detect_peaks(curve(scores))] == expected


@pytest.mark.parametrize("scores,expected", [
    ([.161, .172, .171, .173, .162], [3]),
    ([.174, .176, .174], [1]),
    ([.173, .172, .171], [0]),
    ([.161, .173, .173, .171, .172, .161], [2]),
    ([.161, .173, .171, .173, .161], [1]),
    ([-.182, -.172, -.171, -.183], [2]),
])
def test_rounded_platform_representatives(scores, expected):
    frames = curve(scores)
    snapshot = [replace(f) for f in frames]
    peaks = detect_peaks(frames)
    assert [p.frame_index for p in peaks] == expected
    assert frames == snapshot
    assert all(p is frames[i] for p, i in zip(peaks, expected))


def test_original_precision_in_ranking_and_expansion():
    frames = curve([.10, .172, .171, .10, .173, .1382, .10])
    peaks = detect_peaks(frames)
    assert [p.frame_index for p in peaks] == [1, 4]
    anchors = select_anchors(peaks, 0, 1)
    assert anchors == [frames[4]]  # Rounded peak scores tie; original scores do not.
    selected, records = expand_peaks(frames, anchors, .8, 2, 5)
    assert selected == [frames[4]]  # .1382 < .173 * .8, but > .17 * .8.
    assert records[0]["peak_score"] == .173
    assert records[0]["local_threshold"] == pytest.approx(.1384)


def test_expansion_continuity_radius_limit_and_close_frames():
    frames = curve([.9, .1, .85, 1, .95, .9, .1, .9],
                   [19.5, 20, 20.5, 21, 21.5, 22, 22.5, 23])
    anchor = frames[3]
    selected, _ = expand_peaks(frames, [anchor], .8, 10, 10)
    assert [f.timestamp for f in selected] == [20.5, 21, 21.5, 22]
    selected, _ = expand_peaks(frames, [anchor], .8, .5, 5)
    assert [f.timestamp for f in selected] == [20.5, 21, 21.5]
    selected, _ = expand_peaks(frames, [anchor], .8, 10, 2)
    assert [f.timestamp for f in selected] == [21, 21.5]
    selected, _ = expand_peaks(frames, [anchor], .96, 10, 5)
    assert selected == [anchor]


def test_overlap_unique_and_negative_anchor_retained():
    frames = curve([1, .9, 1])
    selected, records = expand_peaks(frames, detect_peaks(frames), .8, 2, 5)
    assert len(selected) == 3 and len(records) == 2
    frames = curve([-.2, -.1, -.2])
    selected, _ = expand_peaks(frames, detect_peaks(frames), .8, 2, 5)
    assert selected == [frames[1]]


@pytest.mark.parametrize("name,value", [
    ("enabled", 1), ("max_anchor_count", -1), ("max_anchor_count", True),
    ("max_frames_per_peak", 0), ("relative_ratio", 1.1),
    ("relative_ratio", float("nan")), ("min_anchor_gap_seconds", -1),
    ("max_peak_radius_seconds", float("inf")),
])
def test_invalid_config(name, value):
    config = OperatorConfig()
    setattr(config.postprocess.p2_multi_anchor, name, value)
    with pytest.raises(ConfigurationError):
        config.validate()


def test_yaml():
    config = OperatorConfig.from_yaml("configs/default.yaml")
    assert config.postprocess.p2_multi_anchor.enabled
    assert config.postprocess.p2_multi_anchor.max_anchor_count == 5


def test_p2_pipeline_two_pass_meta_and_reset(video, monkeypatch):
    import video_keyframe.operator as module
    def forbidden(*args):
        pytest.fail("P2 must bypass legacy frame filtering")
    monkeypatch.setattr(module, "filter_by_quantile", forbidden)
    monkeypatch.setattr(module, "select_topk", forbidden)
    monkeypatch.setattr(module, "suppress_by_temporal_gap", forbidden)
    config = OperatorConfig()
    config.model.batch_size = 2
    operator = VideoKeyframeOperator(FakeModel(), config)
    for _ in range(2):
        result = operator.run(video.read_bytes(), "red")
        assert [f.timestamp for f in result.keyframes] == [0, 1, 2, 3, 4]
        assert all(f.jpg_bytes.startswith(bytes([255, 216])) for f in result.keyframes)
        assert result.meta["anchor_timestamps"] == [2]
        assert result.meta["detected_peak_count"] == 1
        assert result.meta["selected_anchor_count"] == 1
        assert result.meta["returned_keyframe_count"] == 5
        assert result.meta["reread_frame_count"] == 41
        assert result.meta["image_encoded_count"] == 5
        assert "min_match_frame_gap" not in result.meta
        json.dumps(result.meta)
    operator.config.postprocess.p2_multi_anchor.max_anchor_count = 0
    result = operator.run(video, "red")
    assert result.keyframes == []
    assert result.meta["reread_frame_count"] == 0
