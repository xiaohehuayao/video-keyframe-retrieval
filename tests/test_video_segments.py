from copy import deepcopy
import pytest
from video_keyframe.config import VideoSegmentConfig
from video_keyframe.schemas import VideoSegment
from video_keyframe.postprocess.video_segments import build_video_segments, merge_overlapping_segments
from test_multi_anchor import curve


def segment(i, start, end, score=.2):
    anchor = dict(frame_index=i, timestamp=(start + end)/2, score=score)
    return VideoSegment(i, start, end, end-start, [i], [anchor], anchor)


@pytest.mark.parametrize("bounds,expected", [
    ([(0, 3), (2, 5)], [(0, 5)]),
    ([(0, 5), (1, 2)], [(0, 5)]),
    ([(0, 3), (2, 5), (4, 8)], [(0, 8)]),
    ([(0, 2), (2, 4)], [(0, 2), (2, 4)]),
    ([(0, 2), (3, 4)], [(0, 2), (3, 4)]),
    ([(0, 2), (0, 2)], [(0, 2)]),
    ([], []),
])
def test_union(bounds, expected):
    inputs = [segment(i, *b, score=i/10) for i, b in enumerate(bounds)]
    snapshot = deepcopy(inputs)
    merged = merge_overlapping_segments(inputs[::-1])
    assert [(s.start, s.end) for s in merged] == expected
    assert inputs == snapshot
    assert sorted(i for s in merged for i in s.source_segment_ids) == list(range(len(inputs)))
    for s in merged:
        assert s.duration == s.end-s.start
        assert s.representative_anchor["score"] == max(a["score"] for a in s.anchors)


def test_boundaries_singleton_radius_and_invalid_duration():
    frames = curve([.1, .9, .1], [0, 1, 2])
    config = VideoSegmentConfig(enabled=True)
    raw, segments = build_video_segments(frames, [frames[1]], 3, config)
    assert (segments[0].start, segments[0].end) == (.5, 1.5)
    assert raw[0]["left_stop_reason"] == "below_threshold"
    config.max_peak_radius_seconds = .1
    raw, segments = build_video_segments(frames, [frames[1]], 3, config)
    assert (segments[0].start, segments[0].end) == (.9, 1.1)
    assert raw[0]["right_stop_reason"] == "radius_limit"
    config.max_peak_radius_seconds = 0
    raw, segments = build_video_segments(frames, [frames[1]], 3, config)
    assert not segments and raw[0]["skip_reason"] == "nonpositive_duration"
    config.max_peak_radius_seconds = 10
    raw, segments = build_video_segments(curve([1]), curve([1]), .1, config)
    assert (segments[0].start, segments[0].end) == (0, .1)
    assert build_video_segments([], [], 3, config) == ([], [])


def test_merge_switch_and_no_duration_cap():
    frames = curve([1]*40)
    config = VideoSegmentConfig(enabled=True, max_peak_radius_seconds=10)
    raw, segments = build_video_segments(frames, [frames[10], frames[25]], 40, config)
    assert len(raw) == 2 and len(segments) == 1
    assert segments[0].duration == 35
    config.merge_overlapping_segments = False
    assert len(build_video_segments(frames, [frames[10], frames[25]], 40, config)[1]) == 2
