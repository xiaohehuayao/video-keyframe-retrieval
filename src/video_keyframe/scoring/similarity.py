import numpy as np


def cosine_similarity_scores(image_embeddings, text_embeddings) -> np.ndarray:
    """[N,D] × [M,D] → [N,M]；单文本时返回 [N]。拒绝零向量。"""
    arrays = [np.asarray(x, dtype=np.float32) for x in (image_embeddings, text_embeddings)]
    for index, array in enumerate(arrays):
        if array.ndim != 2 or not np.isfinite(array).all():
            raise ValueError("embeddings 必须为有限值二维数组")
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        if (norms == 0).any():
            raise ValueError("embeddings 不能包含零向量")
        arrays[index] = array / norms
    if arrays[0].shape[1] != arrays[1].shape[1] or arrays[1].shape[0] == 0:
        raise ValueError("embedding 维度不匹配或文本为空")
    scores = np.clip(arrays[0] @ arrays[1].T, -1, 1)
    return scores[:, 0] if scores.shape[1] == 1 else scores
