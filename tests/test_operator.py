from io import BytesIO
import json
import numpy as np
import pytest
from PIL import Image
from video_keyframe import OperatorConfig, VideoKeyframeOperator
from video_keyframe.exceptions import ConfigurationError
from video_keyframe.video.source import VideoSourceResolver


def legacy_config():
    config = OperatorConfig()
    config.postprocess.p2_multi_anchor.enabled = False
    return config


class FakeModel:
    def __init__(self):
        self.texts = []

    def encode_text(self, texts):
        self.texts.extend(texts)
        return np.array([[1, 0] for _ in texts], dtype=np.float32)

    def encode_images(self, images):
        return np.array([[1, 0] for _ in images], dtype=np.float32)


@pytest.fixture
def video(tmp_path):
    import cv2
    path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    assert writer.isOpened(), "测试环境需要 MJPG 编码器"
    try:
        for _ in range(50):
            writer.write(np.full((32, 32, 3), (0, 0, 255), dtype=np.uint8))
    finally:
        writer.release()
    return path


def test_pipeline_and_meta(video):
    config = legacy_config()
    config.model.batch_size = 2
    model = FakeModel()
    result = VideoKeyframeOperator(model, config).run(video, "red", "blue")
    assert model.texts == ["red"]
    assert [f.timestamp for f in result.keyframes] == [0, 2, 4]
    assert result.meta["decoded_frame_count"] == 50
    assert result.meta["sampled_frame_count"] == 5
    assert result.meta["image_batch_count"] == 3
    assert result.meta["temporal_suppressed_count"] == 2
    assert result.meta["returned_count"] == 3
    assert result.meta["candidate_count"] == 5
    assert result.meta["effective_threshold"] == 1.0
    assert result.meta["reread_frame_count"] == 41
    assert "text_sim_thresh" not in result.meta
    image = Image.open(BytesIO(result.keyframes[0].jpg_bytes))
    r, g, b = image.getpixel((0, 0))
    assert r > 240 and b < 10 and g < 10
    json.dumps(result.meta)


def test_negative_scores_tie(video):
    config = legacy_config()
    config.scoring.negative_weight = 1
    model = FakeModel()
    result = VideoKeyframeOperator(model, config).run(video.read_bytes(), "red", "red")
    assert len(result.keyframes) == 3
    assert all(frame.score == 0 for frame in result.keyframes)
    assert result.meta["effective_threshold"] == 0
    assert result.meta["negative_prompt_enabled"] is True
    assert model.texts == ["red", "red"]


def test_source_cleanup(video):
    from pathlib import Path
    with VideoSourceResolver() as resolver:
        temporary = Path(resolver.resolve(video.read_bytes()))
        assert temporary.exists()
    assert not temporary.exists()
    assert video.exists()


def test_invalid_config():
    config = legacy_config()
    config.model.batch_size = 0
    with pytest.raises(ConfigurationError):
        VideoKeyframeOperator(FakeModel(), config)


def test_output_flags_and_state_reset(video):
    config = legacy_config()
    config.output.include_score = False
    config.output.include_frame_index = False
    config.postprocess.max_match_count = 1
    operator = VideoKeyframeOperator(FakeModel(), config)
    for _ in range(2):
        result = operator.run(video, "red")
        assert result.meta["sampled_frame_count"] == 5
        assert result.meta["topk_trimmed_count"] == 2
        assert result.meta["keyframes"] == [{"timestamp": 0.0}]
        assert result.keyframes[0].frame_index is None
        assert result.keyframes[0].score == 1.0


def test_empty_topk_skips_second_pass(video, monkeypatch):
    from video_keyframe.video.decoder import VideoDecoder
    def forbidden(*args):
        pytest.fail("Empty selection must skip second decoding")
    monkeypatch.setattr(VideoDecoder, "iter_selected_frames", forbidden)
    config = legacy_config()
    config.postprocess.max_match_count = 0
    result = VideoKeyframeOperator(FakeModel(), config).run(video, "red")
    assert result.keyframes == []
    assert result.meta["reread_frame_count"] == 0
    assert result.meta["reread_time_ms"] == 0


@pytest.mark.parametrize("source_kind", ["bytes", "url"])
def test_temporary_source_survives_second_pass(video, monkeypatch, source_kind):
    from pathlib import Path
    from video_keyframe.video.decoder import VideoDecoder
    import video_keyframe.video.source as source_module
    data = video.read_bytes()
    if source_kind == "url":
        # Mock HTTP response, exercising resolver download without public network.
        monkeypatch.setattr(source_module, "urlopen", lambda *a, **k: BytesIO(data))
        source = "https://example.invalid/video.mp4"
    else:
        source = data
    paths = []
    original = VideoDecoder.open
    def track_open(self, path):
        assert Path(path).is_file()
        paths.append(Path(path))
        return original(self, path)
    monkeypatch.setattr(VideoDecoder, "open", track_open)
    result = VideoKeyframeOperator(FakeModel(), legacy_config()).run(source, "red")
    assert len(paths) == 2 and paths[0] == paths[1]
    assert not paths[0].exists()
    assert result.meta["image_encoded_count"] == 5


def test_missing_selected_frame_and_cleanup(video, monkeypatch):
    from pathlib import Path
    from video_keyframe.video.decoder import VideoDecoder
    from video_keyframe.exceptions import VideoDecodeError
    paths, decoders = [], []
    original_open = VideoDecoder.open
    original_read = VideoDecoder.iter_selected_frames
    def track_open(self, path):
        paths.append(Path(path))
        decoders.append(self)
        return original_open(self, path)
    def missing(self, indices):
        yield from original_read(self, indices | {1000})
    monkeypatch.setattr(VideoDecoder, "open", track_open)
    monkeypatch.setattr(VideoDecoder, "iter_selected_frames", missing)
    with pytest.raises(VideoDecodeError, match="1000"):
        VideoKeyframeOperator(FakeModel(), legacy_config()).run(video.read_bytes(), "red")
    assert len(paths) == 2
    assert not paths[0].exists()
    assert all(decoder._capture is None for decoder in decoders)


def test_decoder_selected_frames_match_first_pass(video):
    from video_keyframe.video.decoder import VideoDecoder
    indices = {1, 20, 37}
    with VideoDecoder() as decoder:
        decoder.open(str(video))
        expected = {f.frame_index: f for f in decoder.iter_frames() if f.frame_index in indices}
    with VideoDecoder() as decoder:
        decoder.open(str(video))
        actual = list(decoder.iter_selected_frames(indices))
        assert decoder.read_frame_count == 38
    assert [f.frame_index for f in actual] == [1, 20, 37]
    for frame in actual:
        assert frame.timestamp == expected[frame.frame_index].timestamp
        np.testing.assert_array_equal(frame.image, expected[frame.frame_index].image)


@pytest.mark.parametrize("p2_enabled", [False, True])
def test_global_selection_batch_invariance_and_rgb_release(tmp_path, monkeypatch, p2_enabled):
    import cv2
    import weakref
    import video_keyframe.operator as operator_module
    from video_keyframe.video.decoder import VideoDecoder
    path = tmp_path / "colors.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (32, 32))
    assert writer.isOpened()
    try:
        for i in range(50):
            writer.write(np.full((32, 32, 3), i * 4, dtype=np.uint8))
    finally:
        writer.release()

    class ImageModel(FakeModel):
        def __init__(self):
            super().__init__()
            self.refs = []
            self.count = 0
        def encode_images(self, images):
            # Old RGB batches must not be retained across inference calls.
            assert not any(ref() is not None for ref in self.refs)
            self.refs = [weakref.ref(image) for image in images]
            self.count += len(images)
            return np.array([[float(image.mean()) / 255, 1] for image in images], dtype=np.float32)

    original_filter = operator_module.detect_peaks if p2_enabled else operator_module.filter_by_quantile
    models = []
    def inspect_records(records, *args):
        assert all(not hasattr(record, "image") for record in records)
        assert not any(ref() is not None for ref in models[-1].refs)
        return original_filter(records, *args)
    monkeypatch.setattr(operator_module, "detect_peaks" if p2_enabled else "filter_by_quantile", inspect_records)
    results = []
    for batch_size in (1, 2, 4):
        config = legacy_config()
        config.postprocess.p2_multi_anchor.enabled = p2_enabled
        config.model.batch_size = batch_size
        config.scoring.candidate_quantile = 0.5
        config.postprocess.min_match_frame_gap = 0
        config.postprocess.max_match_count = 2
        model = ImageModel()
        models.append(model)
        result = VideoKeyframeOperator(model, config).run(path, "bright")
        assert model.count == 5  # No model calls during the second pass.
        assert [f.frame_index for f in result.keyframes] == ([30, 40] if p2_enabled else [40, 30])
        assert result.meta["reread_frame_count"] == 41
        results.append(result)
        for frame in result.keyframes:
            with Image.open(BytesIO(frame.jpg_bytes)) as image:
                assert abs(float(np.asarray(image).mean()) - frame.frame_index * 4) < 5
    assert len({r.meta.get("effective_threshold") for r in results}) == 1
    assert len({tuple(f.score for f in r.keyframes) for r in results}) == 1
