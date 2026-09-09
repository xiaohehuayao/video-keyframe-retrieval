import numpy as np
import pytest
from video_keyframe.scoring.similarity import cosine_similarity_scores
from video_keyframe.scoring.negative_prompt import combine_positive_negative_scores


def test_cosine_normalizes_and_keeps_negative_values():
    np.testing.assert_allclose(cosine_similarity_scores([[2, 0], [0, 3], [-4, 0]], [[5, 0]]), [1, 0, -1])


def test_negative_weight():
    np.testing.assert_allclose(combine_positive_negative_scores([0.5], [0.4], 0.5), [0.3])


def test_reject_zero_embedding():
    with pytest.raises(ValueError):
        cosine_similarity_scores([[0, 0]], [[1, 0]])
