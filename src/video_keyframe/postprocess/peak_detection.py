"""Local maxima on a time-ordered, finite similarity curve.检测局部峰，平台取中间帧，处理边界。"""
from math import isfinite


def detect_peaks(frames):
    """Collapse equal-score plateaus to their left-middle sample.

    Endpoints compare only with their existing neighbor. A constant curve
    (including a singleton) yields one representative peak.
    """
    for i, frame in enumerate(frames):
        if not isfinite(frame.score) or not isfinite(frame.timestamp):
            raise ValueError("Similarity curve must contain finite values")
        if i and frame.timestamp <= frames[i - 1].timestamp:
            raise ValueError("Similarity curve timestamps must increase strictly")
    peaks = []
    start = 0
    while start < len(frames):
        end = start
        while end + 1 < len(frames) and frames[end + 1].score == frames[start].score:
            end += 1
        score = frames[start].score
        if (start == 0 or score > frames[start - 1].score) and (
                end == len(frames) - 1 or score > frames[end + 1].score):
            peaks.append(frames[(start + end) // 2])
        start = end + 1
    return peaks
