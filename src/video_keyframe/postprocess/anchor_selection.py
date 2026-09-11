"""Score-ranked peak Top-K with anchor-only temporal diversity."""
#峰分数降序排名，满足 Anchor 时间间隔后保留，达到 K 即停止。

from math import isfinite


def select_anchors(peaks, min_anchor_gap_seconds, max_anchor_count):
    if not isfinite(min_anchor_gap_seconds) or min_anchor_gap_seconds < 0:
        raise ValueError("min_anchor_gap_seconds must be finite and nonnegative")
    if type(max_anchor_count) is not int or max_anchor_count < 0:
        raise ValueError("max_anchor_count must be a nonnegative integer")
    selected = []
    for peak in sorted(peaks, key=lambda f: (-f.score, f.timestamp, f.frame_index)):
        #`-f.score`：**降序**，分数高的 peak 排在前面（score 越大越好）.
        #如果两个 peak 分数一模一样，就比较时间戳，**时间更早的放前面**。
        #如果两个 peak 分数和时间戳都一样，就比较帧索引，**帧索引更小的放前面**。
        if len(selected) == max_anchor_count:
            break
        if all(abs(peak.timestamp - anchor.timestamp) >= min_anchor_gap_seconds
               for anchor in selected):
            selected.append(peak)
    return selected
