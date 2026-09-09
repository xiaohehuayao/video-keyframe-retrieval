import numpy as np


def combine_positive_negative_scores(positive_scores, negative_scores=None, negative_weight=0.0):
    """final = positive - weight * negative，不做概率转换或裁剪。"""
    if not np.isfinite(negative_weight) or negative_weight < 0:
        raise ValueError("negative_weight 必须为非负有限值")
    positive = np.asarray(positive_scores, dtype=np.float32)
    if negative_scores is None:
        return positive.copy()
    negative = np.asarray(negative_scores, dtype=np.float32)
    if negative.shape != positive.shape:
        raise ValueError("正负分数形状必须相同")
    return positive - negative_weight * negative
