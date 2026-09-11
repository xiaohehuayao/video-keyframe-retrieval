"""用两位小数检测局部平台峰，保留代表帧的原始分数。"""
from math import isfinite


def detect_peaks(frames):
    """平台判断及邻居比较均使用 round(score, 2)，不修改输入记录。

    平台代表取原始最高分；并列时取最靠近平台下标中心的帧，再取较早帧。
    边界只比较存在的邻居；单帧或舍入后全等分曲线产生一个代表峰。
    """
    for i, frame in enumerate(frames):
        if not isfinite(frame.score) or not isfinite(frame.timestamp):
            raise ValueError("Similarity curve must contain finite values")
        if i and frame.timestamp <= frames[i - 1].timestamp:
            raise ValueError("Similarity curve timestamps must increase strictly")
    detection_scores = [round(frame.score, 2) for frame in frames]
    peaks = []
    start = 0
    while start < len(frames):
        end = start
        while end + 1 < len(frames) and detection_scores[end + 1] == detection_scores[start]:
            end += 1
        score = detection_scores[start]
        if (start == 0 or score > detection_scores[start - 1]) and (
                end == len(frames) - 1 or score > detection_scores[end + 1]):
            representative = min(
                range(start, end + 1),
                key=lambda i: (-frames[i].score, abs(2 * i - start - end), i),
            )
            peaks.append(frames[representative])
        start = end + 1
    return peaks
