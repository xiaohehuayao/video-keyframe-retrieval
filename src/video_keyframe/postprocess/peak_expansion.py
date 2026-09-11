"""Continuous expansion around each anchor; no final-frame gap filter.连续向左右扩展，遇到低分帧或超出半径停止；保留 Anchor，限制每峰数量。"""


def expand_peaks(frames, anchors, relative_ratio, max_peak_radius_seconds,
                 max_frames_per_peak):
    positions = {frame.frame_index: i for i, frame in enumerate(frames)}
    selected, records, seen = [], [], set()
    for anchor in anchors:
        center = positions[anchor.frame_index]
        threshold = anchor.score * relative_ratio
        neighbors = []
        for direction in (-1, 1):
            i = center + direction
            while 0 <= i < len(frames):
                frame = frames[i]
                if (frame.score < threshold or
                        abs(frame.timestamp - anchor.timestamp) > max_peak_radius_seconds):
                    break
                neighbors.append(frame)
                i += direction
        neighbors.sort(key=lambda f: (-f.score, abs(f.timestamp - anchor.timestamp),
                                      f.timestamp, f.frame_index))
        group = [anchor] + neighbors[:max_frames_per_peak - 1]
        group.sort(key=lambda f: (f.timestamp, f.frame_index))
        records.append(dict(frame_index=anchor.frame_index, timestamp=anchor.timestamp,
                            peak_score=anchor.score, local_threshold=threshold,
                            expanded_frame_count=len(group),
                            expanded_timestamps=[f.timestamp for f in group]))
        for frame in group:
            if frame.frame_index not in seen:
                seen.add(frame.frame_index)
                selected.append(frame)
    return selected, records
