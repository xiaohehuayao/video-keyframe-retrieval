"""图片与视频共用的文本分数连续区间；不在此处限制输出帧数。"""
from ..schemas import PeakRegion

#筛选基于anchor的视频生成区间
def find_contiguous_region(frames, anchor, relative_ratio, max_peak_radius_seconds,
                           *, positions=None):
    if positions is None:
        positions = {f.frame_index: i for i, f in enumerate(frames)}
    center = positions[anchor.frame_index]
    threshold = anchor.score * relative_ratio
    edges, reasons = [], []
    for direction in (-1, 1):
        # -1 向左、1 向右；低于阈值或超出半径就停止，不跳过低分帧。
        edge = center
        reason = "video_boundary"
        i = center + direction
        while 0 <= i < len(frames):
            frame = frames[i]
            if abs(frame.timestamp - anchor.timestamp) > max_peak_radius_seconds:
                reason = "radius_limit"
                break
            if frame.score < threshold:
                reason = "below_threshold"
                break
            edge = i
            i += direction
        edges.append(edge)
        reasons.append(reason)
    return PeakRegion(edges[0], edges[1], threshold, reasons[0], reasons[1])
    #- `edges[0]`：向左扩展得到的起始下标- `edges[1]`：向右扩展得到的结束下标- `reasons[0]`：左侧停止原因- `reasons[1]`：右侧停止原因

#筛选关键帧
def select_region_keyframes(frames, region, anchor, max_frames_per_peak):
    neighbors = []
    frame_slice = frames[region.start_position : region.end_position + 1]
    for f in frame_slice:
    # 过滤掉锚点帧
        if f.frame_index != anchor.frame_index:
          neighbors.append(f)  

    # 优先原始分数，再按到 Anchor 的距离、时间戳、帧号排序。
    neighbors.sort(key=lambda f: (-f.score, abs(f.timestamp - anchor.timestamp),
                                  f.timestamp, f.frame_index))
    # Anchor 必须保留，组内最多 max_frames_per_peak 帧，输出按时间先后排列。
    group = [anchor] + neighbors[:max_frames_per_peak - 1]
    return sorted(group, key=lambda f: (f.timestamp, f.frame_index))
