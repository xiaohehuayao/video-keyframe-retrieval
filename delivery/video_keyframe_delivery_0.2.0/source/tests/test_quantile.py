"""全局分位数筛选与配置迁移。"""
import math
import pytest
from video_keyframe.config import OperatorConfig
from video_keyframe.exceptions import ConfigurationError
from video_keyframe.schemas import FrameScore
from video_keyframe.postprocess.threshold import filter_by_quantile


def records(scores):
    return [FrameScore(i, float(i), score, score) for i, score in enumerate(scores)]


def test_quantile_and_input_order():
    frames = records(list(range(10)))
    kept, threshold = filter_by_quantile(frames, 0.9)
    assert threshold == pytest.approx(8.1)
    assert [f.frame_index for f in kept] == [9]
    reversed_kept, reversed_threshold = filter_by_quantile(list(reversed(frames)), 0.9)
    assert reversed_threshold == threshold
    assert reversed_kept == kept


def test_empty_single_ties_and_endpoints():
    assert filter_by_quantile([], 0.9) == ([], None)
    frames = records([0.5, 0.5, 0.5])
    assert filter_by_quantile(frames, 0.9) == (frames, 0.5)
    assert filter_by_quantile(frames[:1], 0.9) == (frames[:1], 0.5)
    frames = records([0.1, 0.2])
    assert filter_by_quantile(frames, 0)[0] == frames
    assert filter_by_quantile(frames, 1)[0] == frames[1:]


@pytest.mark.parametrize("value", [-0.1, 1.1, math.nan, math.inf, True])
def test_invalid_quantile(value):
    config = OperatorConfig()
    config.scoring.candidate_quantile = value
    with pytest.raises(ConfigurationError):
        config.validate()
    with pytest.raises(ValueError):
        filter_by_quantile([], value)


def test_invalid_score():
    with pytest.raises(ValueError):
        filter_by_quantile(records([math.nan]), 0.9)


def test_old_yaml_rejected_and_default_loads(tmp_path):
    path = tmp_path / "old.yaml"
    path.write_text("scoring:\n  text_sim_thresh: 0.2\n", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        OperatorConfig.from_yaml(path)
    from pathlib import Path
    config = OperatorConfig.from_yaml(Path(__file__).resolve().parents[1] / "configs/default.yaml")
    assert config.scoring.candidate_quantile == 0.9
