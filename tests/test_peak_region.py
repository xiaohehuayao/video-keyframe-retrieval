from test_multi_anchor import curve
from video_keyframe.postprocess.peak_region import find_contiguous_region, select_region_keyframes


def test_region_continuity_before_jpg_limit():
    frames = curve([.9, .1, .85, 1, .95, .9, .1, .9])
    region = find_contiguous_region(frames, frames[3], .8, 10)
    assert (region.start_position, region.end_position) == (2, 5)
    assert region.left_stop_reason == region.right_stop_reason == "below_threshold"
    group = select_region_keyframes(frames, region, frames[3], 2)
    assert [f.frame_index for f in group] == [3, 4]
    assert region.end_position == 5


def test_radius_inclusive_and_negative_anchor():
    frames = curve([1]*5)
    region = find_contiguous_region(frames, frames[2], .8, 1)
    assert (region.start_position, region.end_position) == (1, 3)
    assert region.left_stop_reason == "radius_limit"
    frames = curve([-.2, -.1, -.2])
    region = find_contiguous_region(frames, frames[1], .8, 3)
    assert region.start_position == region.end_position == 1
