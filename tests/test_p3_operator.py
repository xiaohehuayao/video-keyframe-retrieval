from io import BytesIO
from pathlib import Path
import json
import numpy as np
import pytest
from video_keyframe import OperatorConfig, VideoKeyframeOperator
from video_keyframe.config import VideoSegmentConfig
from video_keyframe.exceptions import ConfigurationError
from test_operator import video, FakeModel


class CurveModel(FakeModel):
    def __init__(self):
        super().__init__()
        self.count = 0
    def encode_images(self, images):
        values = [.9, .1, .8, .1, .7][self.count:self.count + len(images)]
        self.count += len(images)
        return np.array([[v, np.sqrt(1-v*v)] for v in values], dtype=np.float32)


def test_shared_anchors_and_independent_limits(video):
    config = OperatorConfig()
    config.model.batch_size = 2
    config.postprocess.p2_multi_anchor.min_anchor_gap_seconds = 0
    config.postprocess.p2_multi_anchor.max_anchor_count = 3
    config.postprocess.p2_multi_anchor.max_frames_per_peak = 1
    baseline = VideoKeyframeOperator(CurveModel(), config).run(video, "query")
    config.postprocess.p3_video.enabled = True
    model = CurveModel()
    result = VideoKeyframeOperator(model, config).run(video, "query")
    assert result.meta["anchor_timestamps"] == [0, 2, 4]
    assert [s.representative_anchor["timestamp"] for s in result.segments] == [0, 2]
    assert result.keyframes == baseline.keyframes
    assert model.count == 5
    assert result.meta["raw_segment_count"] == 2
    assert result.meta["exported_clip_count"] == 0
    json.dumps(result.meta)
    config.postprocess.p2_multi_anchor.max_anchor_count = 0
    result = VideoKeyframeOperator(CurveModel(), config).run(video, "query")
    assert not result.keyframes and len(result.segments) == 2
    assert result.meta["reread_frame_count"] == 0


@pytest.mark.parametrize("kind", ["bytes", "url"])
def test_export_before_temporary_cleanup(video, tmp_path, monkeypatch, kind):
    import video_keyframe.operator as module
    import video_keyframe.video.source as source_module
    monkeypatch.setattr(module, "check_ffmpeg", lambda path: None)
    paths = []
    def export(source, segments, output_dir, executable):
        paths.append(Path(source))
        assert paths[-1].exists()
        for s in segments:
            s.export_status = "exported"
            s.filename = "clips/test.mp4"
    monkeypatch.setattr(module, "export_video_segments", export)
    monkeypatch.setattr(source_module, "urlopen", lambda *a, **k: BytesIO(video.read_bytes()))
    config = OperatorConfig()
    config.postprocess.p3_video.enabled = True
    config.output.video.export_clips = True
    source = video.read_bytes() if kind == "bytes" else "https://example.invalid/video.mp4"
    result = VideoKeyframeOperator(FakeModel(), config).run(source, "red", output_dir=tmp_path)
    assert len(paths) == 1 and not paths[0].exists()
    assert result.meta["exported_clip_count"] == 1


@pytest.mark.parametrize("field,value", [("relative_ratio", 1.1), ("relative_ratio", float("nan")),
    ("max_anchor_count", True), ("max_anchor_count", -1), ("max_peak_radius_seconds", -1),
    ("enabled", 1), ("merge_overlapping_segments", "yes")])
def test_invalid_p3(field, value):
    config = OperatorConfig()
    setattr(config.postprocess.p3_video, field, value)
    with pytest.raises(ConfigurationError):
        config.validate()


def test_export_configuration(video):
    config = OperatorConfig()
    config.output.video.export_clips = True
    with pytest.raises(ConfigurationError):
        config.validate()
    config.postprocess.p3_video.enabled = True
    with pytest.raises(ConfigurationError, match="output_dir"):
        VideoKeyframeOperator(FakeModel(), config).run(video, "q")
    config.postprocess.p2_multi_anchor.enabled = False
    with pytest.raises(ConfigurationError):
        config.validate()


def test_nested_yaml(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("postprocess:\n  p3_video:\n    enabled: true\n    max_anchor_count: 4\noutput:\n  video:\n    export_clips: false\n", encoding="utf-8")
    assert OperatorConfig.from_yaml(path).postprocess.p3_video.max_anchor_count == 4


def test_factory_p3_parameters(tmp_path, monkeypatch):
    import video_keyframe.factory as module
    monkeypatch.setattr(module, "SigLIP2Model", lambda **k: FakeModel())
    operator = module.create_operator(tmp_path, p3_enabled=True, video_max_anchor_count=4,
                                     video_relative_ratio=.6, video_max_peak_radius_seconds=8,
                                     merge_overlapping_segments=False)
    p3 = operator.config.postprocess.p3_video
    assert p3.enabled and p3.max_anchor_count == 4 and p3.relative_ratio == .6
    assert p3.max_peak_radius_seconds == 8 and not p3.merge_overlapping_segments


def test_search_parameters(monkeypatch):
    import video_keyframe.operator as module
    def run(self, source, prompt, negative_prompt, *, output_dir):
        assert self.config.postprocess.p3_video.enabled
        assert self.config.postprocess.p3_video.max_anchor_count == 4
        assert output_dir == "out"
        return "ok"
    monkeypatch.setattr(module.VideoKeyframeOperator, "run", run)
    assert module.search_keyframes("source", "q", p3_enabled=True,
                                   video_max_anchor_count=4, output_dir="out") == "ok"


@pytest.mark.parametrize("failed", [False, True])
def test_cli_outputs_meta_and_failure_exit(tmp_path, monkeypatch, failed):
    import sys
    import video_keyframe.cli as cli
    from video_keyframe.schemas import OperatorResult
    from test_video_segments import segment
    config = OperatorConfig()
    config.postprocess.p3_video.enabled = True
    monkeypatch.setattr(cli.OperatorConfig, "from_yaml", lambda path: config)
    clip = segment(0, 0, 1)
    clip.export_status = "failed" if failed else "not_requested"
    result = OperatorResult([], dict(p3_video_enabled=True, exported_clip_count=0,
                                    failed_clip_count=int(failed), keyframes=[],
                                    segments=[dict(start=0, end=1)]), [clip])
    def run(self, source, prompt, negative, *, output_dir):
        assert output_dir == tmp_path
        return result
    monkeypatch.setattr(cli.VideoKeyframeOperator, "run", run)
    monkeypatch.setattr(sys, "argv", ["video_keyframe", "source.mp4", "query", "--output-dir", str(tmp_path)])
    if failed:
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 1
    else:
        cli.main()
    assert json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))["segments"]
