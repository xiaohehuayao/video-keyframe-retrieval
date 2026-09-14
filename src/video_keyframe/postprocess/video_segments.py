"""Sample midpoint boundaries and strict-overlap union; no merged duration cap."""
from copy import deepcopy
from ..schemas import VideoSegment
from .peak_region import find_contiguous_region


def build_video_segments(frames, anchors, video_duration, config):
    positions = {f.frame_index: i for i, f in enumerate(frames)}
    raw, segments = [], []
    for source_id, anchor in enumerate(anchors):
        #调用find_contiguous_region，将锚点扩张得到的采样帧下标区间，通过**采样中点算法换算成真实视频时间区间**
        region = find_contiguous_region(frames, anchor, config.relative_ratio,
                                        config.max_peak_radius_seconds, positions=positions)
        left, right = region.start_position, region.end_position
        start = (frames[left - 1].timestamp + frames[left].timestamp) / 2 if left else 0.0
        end = ((frames[right].timestamp + frames[right + 1].timestamp) / 2
               if right + 1 < len(frames) else video_duration)
        lower = max(0.0, anchor.timestamp - config.max_peak_radius_seconds)
        upper = min(video_duration, anchor.timestamp + config.max_peak_radius_seconds)
        start, end = max(start, lower), min(end, upper)
        left_reason = "radius_limit" if start == lower and lower > 0 else region.left_stop_reason
        right_reason = ("radius_limit" if end == upper and upper < video_duration
                        else region.right_stop_reason)
        anchor_info = dict(frame_index=anchor.frame_index, timestamp=anchor.timestamp,
                           score=anchor.score)
        record = dict(segment_id=source_id, anchor=anchor_info, start=start, end=end,
                      duration=max(0.0, end - start), first_sample_frame_index=frames[left].frame_index,
                      last_sample_frame_index=frames[right].frame_index,
                      region_sample_count=right - left + 1, local_threshold=region.local_threshold,
                      left_stop_reason=left_reason, right_stop_reason=right_reason,
                      boundary_method="sample_midpoint", valid=end > start,
                      skip_reason=None if end > start else "nonpositive_duration")
        raw.append(record)
        if end > start:
            segments.append(VideoSegment(source_id, start, end, end - start,
                                         [source_id], [anchor_info], anchor_info))
    if config.merge_overlapping_segments:
        segments = merge_overlapping_segments(segments)
    else:
        segments.sort(key=lambda s: (s.start, s.end, s.segment_id))
    for i, segment in enumerate(segments):
        segment.segment_id = i
    return raw, segments


def merge_overlapping_segments(segments):
    result = []
    for source in sorted(segments, key=lambda s: (s.start, s.end, s.segment_id)):
        segment = deepcopy(source)
        if not result or segment.start >= result[-1].end:
            result.append(segment)
            continue
        current = result[-1]
        current.end = max(current.end, segment.end)
        current.duration = current.end - current.start
        current.source_segment_ids.extend(segment.source_segment_ids)
        current.anchors.extend(segment.anchors)
        current.representative_anchor = min(
            current.anchors, key=lambda a: (-a["score"], a["timestamp"], a["frame_index"]))
    return result
