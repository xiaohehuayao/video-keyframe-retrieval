"""Continuous expansion around each anchor; no final-frame gap filter.连续向左右扩展，遇到低分帧或超出半径停止；保留 Anchor，限制每峰数量。"""


from .peak_region import find_contiguous_region, select_region_keyframes

#最终关键帧汇总及跨 Anchor 去重
def expand_peaks(frames, anchors, relative_ratio, max_peak_radius_seconds,
                 max_frames_per_peak):
    positions = {frame.frame_index: i for i, frame in enumerate(frames)}
    selected, records, seen = [], [], set()
    #以当前 anchor 为中心，向时间轴左右扩张，生成候选区间。
    for anchor in anchors:
        region = find_contiguous_region(frames, anchor, relative_ratio,
                                        max_peak_radius_seconds, positions=positions)
        threshold = region.local_threshold

        #调用select_region_keyframes，生成每个峰内的筛选的关键帧列表，最多 max_frames_per_peak 帧。
        group = select_region_keyframes(frames, region, anchor, max_frames_per_peak)
        records.append(dict(frame_index=anchor.frame_index, timestamp=anchor.timestamp,
                            peak_score=anchor.score, local_threshold=threshold,
                            expanded_frame_count=len(group),
                            expanded_timestamps=[f.timestamp for f in group]))

        #全局去重seen：同一个帧不能被多个anchor重复收集.seen是set()集合
        for frame in group:
            if frame.frame_index not in seen:
                seen.add(frame.frame_index)
                selected.append(frame)
    return selected, records
