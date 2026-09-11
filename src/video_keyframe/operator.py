"""P2 主流程：采样评分 → 局部峰 → 多 Anchor 选择 → 邻域扩展 → 二次解码 → JPEG。

关闭 P2 时回退到 P0.5 的全局分位数过滤、帧间隔抑制和帧级 Top-K。
"""
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from .config import OperatorConfig
from .exceptions import ConfigurationError, VideoDecodeError, ModelError
from .schemas import RuntimeState, FrameScore, KeyframeResult, OperatorResult
from .video.source import VideoSourceResolver
from .video.decoder import VideoDecoder
from .video.sampler import FrameSampler
from .scoring.similarity import cosine_similarity_scores
from .scoring.negative_prompt import combine_positive_negative_scores
from .postprocess.threshold import filter_by_quantile
from .postprocess.temporal_gap import suppress_by_temporal_gap
from .postprocess.topk import select_topk
from .postprocess.peak_detection import detect_peaks
from .postprocess.anchor_selection import select_anchors
from .postprocess.peak_expansion import expand_peaks
from .output.jpeg import encode_jpeg
from .output.meta import build_meta
from .utils.timing import measure
from .models.base import VisionLanguageModel


class VideoKeyframeOperator:
    """复用实例可复用模型；每次 run 独立创建运行状态。"""

    def __init__(self, model: VisionLanguageModel | None = None,
                 config: OperatorConfig | None = None):
        self.config = deepcopy(config) if config is not None else OperatorConfig()
        self.config.validate()
        self.model = model

    def run(self, video_source: str | Path | bytes, prompt: str,
            negative_prompt: str | None = None) -> OperatorResult:
        """执行检索并返回 JPEG 关键帧及 Meta。

        P2 按 Anchor 分数降序分组，组内按时间升序，共享帧仅输出一次；
        P0.5 回退模式按帧分数降序返回。模型仅在第一遍解码时参与评分。
        """
        # 校验配置与提示词；P2 参数由 config.postprocess.p2_multi_anchor 提供。
        config = self.config
        config.validate()
        if not isinstance(prompt, str) or not prompt.strip():
            raise ConfigurationError("prompt 必须是非空字符串")
        if negative_prompt is not None and (not isinstance(negative_prompt, str) or not negative_prompt.strip()):
            raise ConfigurationError("negative_prompt 必须是非空字符串或 None")

        # 每次检索独立计数；汇总完整采样分数曲线，但不长期保留 RGB 图片。
        started = perf_counter()
        state = RuntimeState(negative_prompt_enabled=(negative_prompt is not None and config.scoring.negative_weight > 0))
        all_scores: list[FrameScore] = []  # 只累计定位信息和分数，不保存 RGB 图片

        # 解析为本地路径，临时视频文件在两遍解码完成后统一清理。
        with VideoSourceResolver() as resolver:
            path = resolver.resolve(video_source)
            with VideoDecoder() as decoder:
                decoder.open(path)
                metadata = decoder.get_metadata()
                state.video_fps = metadata.fps
                state.video_duration = metadata.duration
                state.total_frame_count = metadata.frame_count
                limit = config.sampling.max_video_duration
                if limit is not None and metadata.duration > limit:
                    raise VideoDecodeError(f"视频时长超过上限 {limit} 秒")
                if self.model is None:
                    from .models.siglip2 import SigLIP2Model
                    self.model = SigLIP2Model(config.model.name, config.model.device,
                                              config.model.dtype, config.model.batch_size)

                # 文本只编码一次，供所有图像批次复用；负向评分仍由配置控制。
                texts = [prompt] + ([negative_prompt] if state.negative_prompt_enabled else [])
                with measure(state, "text_encode_time_ms"):
                    text_features = self.model.encode_text(texts) # type: ignore
                if text_features.ndim != 2 or text_features.shape[0] != len(texts):
                    raise ModelError("文本特征数量与输入不一致")
                state.text_encoded_count = len(texts)

                # 第一遍惰性逐帧解码：记录解码耗时，并把帧交给采样器。
                def decoded_frames():
                    iterator = iter(decoder.iter_frames())
                    while True:
                        with measure(state, "decode_time_ms"):
                            frame = next(iterator, None)
                        if frame is None:
                            break
                        state.decoded_frame_count += 1
                        if limit is not None and frame.timestamp >= limit:
                            raise VideoDecodeError(f"解码视频超过时长上限 {limit} 秒")
                        yield frame

                # 批量编码并评分，只累积轻量 FrameScore；不在批内过滤或选 Top-K。
                # P2 必须保留全部采样分数，才能在完整时间曲线上找峰和扩展。
                def process_batch(batch):
                    with measure(state, "image_encode_time_ms"):
                        assert self.model is not None
                        features = self.model.encode_images([frame.image for frame in batch])
                    if features.ndim != 2 or features.shape[0] != len(batch):
                        raise ModelError("图像特征数量与输入不一致")
                    state.image_batch_count += 1
                    state.image_encoded_count += len(batch)
                    with measure(state, "scoring_time_ms"):
                        positive = cosine_similarity_scores(features, text_features[:1])
                        negative = (cosine_similarity_scores(features, text_features[1:2])
                                    if state.negative_prompt_enabled else None)
                        scores = combine_positive_negative_scores(positive, negative, config.scoring.negative_weight)
                        scored = [FrameScore(frame.frame_index, frame.timestamp, float(scores[i]),
                                              float(positive[i]), None if negative is None else float(negative[i])) for i, frame in enumerate(batch)]
                    state.scored_frame_count += len(scored)
                    all_scores.extend(scored)

                # 按配置采样，凑够 batch_size 推理一次，最后处理不足一批的帧。
                batch = []
                # 采样顺序即时间顺序，all_scores 跨批次保持时间递增。
                for frame in FrameSampler().sample(decoded_frames(), metadata.fps,
                                                    config.sampling.sample_stride, config.sampling.sample_fps):
                    state.sampled_frame_count += 1
                    batch.append(frame)
                    if len(batch) == config.model.batch_size:
                        process_batch(batch)
                        batch = []
                if batch:
                    process_batch(batch)
                # 释放最后一批及循环变量的图片引用，后处理仅使用轻量分数记录。
                batch.clear()
                if state.sampled_frame_count:
                    del frame # type: ignore

            # P2 后处理：全曲线找峰 → 峰级 Top-K 与时间多样性 → 连续邻域扩展。
            with measure(state, "postprocess_time_ms"):
                p2 = config.postprocess.p2_multi_anchor
                anchor_records = []
                peaks, anchors = [], []
                if p2.enabled:
                    # 检测局部峰；相同分数的平台合并为一个代表峰。
                    peaks = detect_peaks(all_scores)
                    # 按峰分数降序贪心选择，间隔仅约束 Anchor，数量上限为峰级 K。
                    anchors = select_anchors(peaks, p2.min_anchor_gap_seconds, p2.max_anchor_count)
                    # 左右连续扩展至局部阈值或半径边界，每峰限量且保留 Anchor。
                    # 扩展内部去除共享帧；之后不再施加最终帧间隔或帧级 Top-K。
                    selected, anchor_records = expand_peaks(
                        all_scores, anchors, p2.relative_ratio,
                        p2.max_peak_radius_seconds, p2.max_frames_per_peak)
                else:
                    # 仅关闭 P2 时执行 P0.5：全局分位数 → 帧间隔抑制 → 帧级 Top-K。
                    candidates, state.effective_threshold = filter_by_quantile(
                        all_scores, config.scoring.candidate_quantile)
                    state.candidate_count = state.threshold_pass_count = len(candidates)
                    kept = suppress_by_temporal_gap(candidates, config.postprocess.min_match_frame_gap)
                    selected = select_topk(kept, config.postprocess.max_match_count)
                    state.temporal_suppressed_count = len(candidates) - len(kept)
                    state.topk_trimmed_count = len(kept) - len(selected)

            # 第二遍重新打开同一本地视频，顺序读取到所需帧，仅对选中帧编码 JPEG。
            # 此处不再调用模型；空结果跳过第二遍，临时源文件此时仍然有效。
            jpeg_by_index = {}
            if selected:
                with VideoDecoder() as output_decoder:
                    with measure(state, "reread_time_ms"):
                        output_decoder.open(path)
                    iterator = output_decoder.iter_selected_frames({item.frame_index for item in selected})
                    try:
                        while True:
                            with measure(state, "reread_time_ms"):
                                output_frame = next(iterator, None)
                            if output_frame is None:
                                break
                            with measure(state, "jpeg_encode_time_ms"):
                                jpeg_by_index[output_frame.frame_index] = encode_jpeg(
                                    output_frame.image, config.output.jpeg_quality)
                            del output_frame
                    finally:
                        iterator.close()
                        state.reread_frame_count = output_decoder.read_frame_count

        # 按 selected 的顺序组装结果：P2 为 Anchor 排名顺序、组内时间顺序。
        keyframes, entries = [], []
        for frame in selected:
            keyframes.append(KeyframeResult(frame.timestamp, frame.score,
                             jpeg_by_index[frame.frame_index],
                             frame.frame_index if config.output.include_frame_index else None))
            entry = {"timestamp": frame.timestamp}
            if config.output.include_score:
                entry.update(score=frame.score, positive_score=frame.positive_score, negative_score=frame.negative_score) # type: ignore
            if config.output.include_frame_index:
                entry["frame_index"] = frame.frame_index
            entries.append(entry)
        state.returned_count = len(keyframes)
        state.total_time_ms = (perf_counter() - started) * 1000
        meta = build_meta(state, config)
        if p2.enabled:
            # P2 顶层 Meta 移除不适用的旧筛选参数与计数，完整配置快照仍保留。
            for key in ("candidate_quantile", "min_match_frame_gap", "max_match_count",
                        "threshold_pass_count", "candidate_count", "effective_threshold",
                        "temporal_suppressed_count", "topk_trimmed_count"):
                meta.pop(key, None)
            # 记录峰、Anchor、扩展明细与去重后的实际输出数量，便于检查重复事件。
            meta.update(detected_peak_count=len(peaks), selected_anchor_count=len(anchors),
                        anchor_timestamps=[f.timestamp for f in anchors],
                        anchor_scores=[f.score for f in anchors], anchors=anchor_records,
                        min_anchor_gap_seconds=p2.min_anchor_gap_seconds,
                        max_anchor_count=p2.max_anchor_count, relative_ratio=p2.relative_ratio,
                        max_peak_radius_seconds=p2.max_peak_radius_seconds,
                        max_frames_per_peak=p2.max_frames_per_peak,
                        returned_keyframe_count=len(keyframes))
        meta["selection_mode"] = "p2_multi_anchor" if p2.enabled else "p0.5_quantile"
        meta["keyframes"] = entries
        return OperatorResult(keyframes, meta)#返回 OperatorResult：包含 JPEG 二进制关键帧列表 + meta 元数据


def search_keyframes(video_source: str | Path | bytes, prompt: str, *,
                     negative_prompt: str | None = None, sample_fps: float = 1.0,
                     candidate_quantile: float = 0.90, negative_weight: float = 0.0,
                     min_match_frame_gap: float = 2.0, max_match_count: int = 10,
                     p2_enabled: bool = True, min_anchor_gap_seconds: float = 5.0,
                     max_anchor_count: int = 5, relative_ratio: float = 0.8,
                     max_peak_radius_seconds: float = 2.0,
                     max_frames_per_peak: int = 5) -> OperatorResult:
    """默认启用 P2 的便捷入口；模型复用使用 VideoKeyframeOperator。

    min_anchor_gap_seconds 和 max_anchor_count 控制 Anchor 选择；
    relative_ratio、max_peak_radius_seconds 和 max_frames_per_peak 控制邻域扩展。
    candidate_quantile、min_match_frame_gap、max_match_count 仅在关闭 P2 时生效。
    """
    config = OperatorConfig()
    config.sampling.sample_fps = sample_fps
    config.scoring.candidate_quantile = candidate_quantile
    config.scoring.negative_weight = negative_weight
    config.postprocess.min_match_frame_gap = min_match_frame_gap
    config.postprocess.max_match_count = max_match_count
    from .config import MultiAnchorConfig
    config.postprocess.p2_multi_anchor = MultiAnchorConfig(
        p2_enabled, min_anchor_gap_seconds, max_anchor_count, relative_ratio,
        max_peak_radius_seconds, max_frames_per_peak)
    return VideoKeyframeOperator(config=config).run(video_source, prompt, negative_prompt)
