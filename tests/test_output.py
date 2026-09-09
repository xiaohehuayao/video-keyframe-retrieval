"""真实视频采样帧配合模拟分数，验证并保存 JPEG 和 Meta。"""
from io import BytesIO
from itertools import islice
import json
from pathlib import Path

import pytest
from PIL import Image

from video_keyframe.config import OperatorConfig
from video_keyframe.schemas import ScoredFrame, RuntimeState
from video_keyframe.video.source import VideoSourceResolver
from video_keyframe.video.decoder import VideoDecoder
from video_keyframe.video.sampler import FrameSampler
from video_keyframe.postprocess.threshold import filter_by_quantile
from video_keyframe.postprocess.temporal_gap import suppress_by_temporal_gap
from video_keyframe.postprocess.topk import select_topk
from video_keyframe.output.jpeg import encode_jpeg
from video_keyframe.output.meta import build_meta


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VIDEO_PATH = PROJECT_ROOT.parent / "video_test" / "test1.mp4"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "output_module_test"


def test_output_from_real_video():
    if not VIDEO_PATH.is_file():
        pytest.skip(f"本地测试视频不存在：{VIDEO_PATH}")

    config = OperatorConfig()
    config.model.name = "mock-scores-no-model"
    config.postprocess.max_match_count = 2
    config.scoring.candidate_quantile = 0.25  # 此输出测试保留四帧候选

    state = RuntimeState()
    mock_scores = [0.10, 0.90, 0.80, 0.70, 0.60]

    # 1. 读取视频，并获取前五张 1 FPS 采样帧。
    with VideoSourceResolver() as resolver, VideoDecoder() as decoder:
        path = resolver.resolve(VIDEO_PATH)
        decoder.open(path)
        metadata = decoder.get_metadata()

        state.video_fps = metadata.fps
        state.video_duration = metadata.duration
        state.total_frame_count = metadata.frame_count

        def counted_frames():
            for frame in decoder.iter_frames():
                state.decoded_frame_count += 1
                yield frame

        sampled = list(islice(
            FrameSampler().sample(
                counted_frames(),
                fps=metadata.fps,
                sample_fps=config.sampling.sample_fps,
            ),
            5,
        ))

    assert len(sampled) == 5, "测试视频需要提供至少五张采样帧"
    state.sampled_frame_count = len(sampled)

    # 2. 使用真实图片，手动指定模拟分数。
    scored = [
        ScoredFrame(
            frame_index=frame.frame_index,
            timestamp=frame.timestamp,
            score=score,
            positive_score=score,
            negative_score=None,
            image=frame.image,
        )
        for frame, score in zip(sampled, mock_scores)
    ]
    state.scored_frame_count = len(scored)

    # 3. 后处理。
    passed, state.effective_threshold = filter_by_quantile(
        scored, config.scoring.candidate_quantile
    )
    kept = suppress_by_temporal_gap(
        passed, config.postprocess.min_match_frame_gap
    )
    selected = select_topk(
        kept, config.postprocess.max_match_count
    )

    assert [f.frame_index for f in selected] == [
        sampled[1].frame_index,
        sampled[3].frame_index,
    ]

    state.threshold_pass_count = len(passed)
    state.candidate_count = len(passed)
    state.temporal_suppressed_count = len(passed) - len(kept)
    state.topk_trimmed_count = len(kept) - len(selected)
    state.returned_count = len(selected)

    # 4. 编码 JPEG，验证能重新打开，再保存。
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    entries = []

    for rank, frame in enumerate(selected):
        jpg_bytes = encode_jpeg(
            frame.image, quality=config.output.jpeg_quality
        )

        with Image.open(BytesIO(jpg_bytes)) as image:
            assert image.format == "JPEG"
            image.load()
            assert image.size == (
                frame.image.shape[1],
                frame.image.shape[0],
            )

        filename = (
            f"{rank:02d}_frame_{frame.frame_index}"
            f"_{frame.timestamp:.3f}s.jpg"
        )
        image_path = OUTPUT_DIR / filename
        image_path.write_bytes(jpg_bytes)
        assert image_path.read_bytes() == jpg_bytes

        entries.append({
            "filename": filename,
            "frame_index": frame.frame_index,
            "timestamp": frame.timestamp,
            "score": frame.score,
            "positive_score": frame.positive_score,
            "negative_score": frame.negative_score,
        })

        print(
            f"\n保存：{image_path}\n"
            f"帧号={frame.frame_index}, "
            f"时间戳={frame.timestamp:.3f}s, "
            f"模拟分数={frame.score}"
        )

    # 5. 生成并保存 Meta；本测试不运行模型，也不测量耗时。
    meta = build_meta(state, config)
    meta["keyframes"] = entries
    meta["score_source"] = "manually_assigned"
    meta["timing_measured"] = False

    meta_path = OUTPUT_DIR / "meta.json"
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 验证 JSON 落盘后可读取，内容及统计正确。
    loaded = json.loads(meta_path.read_text(encoding="utf-8"))
    assert loaded == meta
    assert loaded["returned_count"] == 2
    assert len(loaded["keyframes"]) == 2
    assert loaded["threshold_pass_count"] == (
        loaded["temporal_suppressed_count"]
        + loaded["topk_trimmed_count"]
        + loaded["returned_count"]
    )

    print(f"\nMeta：{meta_path}")
