from unittest.mock import Mock
import pytest

from video_keyframe import create_operator
from video_keyframe.exceptions import ConfigurationError
import video_keyframe.factory as factory


def test_factory_loads_once_and_passes_config(tmp_path, monkeypatch):
    constructor = Mock()
    monkeypatch.setattr(factory, "SigLIP2Model", constructor)
    operator = create_operator(tmp_path, batch_size=7, candidate_quantile=0.8,
                               max_match_count=3, max_video_duration=60, jpeg_quality=80)
    constructor.assert_called_once_with(model_name=str(tmp_path.resolve()),
                                        device="cpu", dtype="float32", batch_size=7)
    assert operator.model is constructor.return_value
    assert operator.config.scoring.candidate_quantile == 0.8
    assert operator.config.scoring.negative_weight == 0
    assert operator.config.postprocess.max_match_count == 3
    assert operator.config.sampling.max_video_duration == 60
    assert operator.config.output.jpeg_quality == 80


def test_factory_rejects_invalid_input_before_loading(tmp_path, monkeypatch):
    constructor = Mock()
    monkeypatch.setattr(factory, "SigLIP2Model", constructor)
    with pytest.raises(ConfigurationError):
        create_operator(tmp_path / "missing")
    with pytest.raises(ConfigurationError):
        create_operator(tmp_path, candidate_quantile=2)
    constructor.assert_not_called()
