"""对外主入口：视频 → 采样 → 特征 → 评分 → 后处理 → JPEG。"""
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
        """返回分数降序关键帧；negative_weight=0 时不编码负向文本。"""
        #参数校验
        config = self.config
        config.validate()
        if not isinstance(prompt, str) or not prompt.strip():
            raise ConfigurationError("prompt 必须是非空字符串")
        if negative_prompt is not None and (not isinstance(negative_prompt, str) or not negative_prompt.strip()):
            raise ConfigurationError("negative_prompt 必须是非空字符串或 None")

        #初始化运行状态
        started = perf_counter()
        state = RuntimeState(negative_prompt_enabled=(negative_prompt is not None and config.scoring.negative_weight > 0))
        all_scores: list[FrameScore] = []  # 只累计定位信息和分数，不保存 RGB 图片

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

                #文本编码
                texts = [prompt] + ([negative_prompt] if state.negative_prompt_enabled else [])
                with measure(state, "text_encode_time_ms"):#计时
                    text_features = self.model.encode_text(texts) # type: ignore
                if text_features.ndim != 2 or text_features.shape[0] != len(texts):
                    raise ModelError("文本特征数量与输入不一致")
                state.text_encoded_count = len(texts)

                #帧解码生成器：`yield` 生成器，**惰性逐帧解码**
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

                #批量处理一整个 batch 帧，把每帧的帧索引、时间戳、分数存入`all_scores`列表
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

                #循环攒帧，凑够 batch_size 就跑一次批量推理
                batch = []
                #FrameSampler 均匀采样 + 分批推理
                for frame in FrameSampler().sample(decoded_frames(), metadata.fps,
                                                    config.sampling.sample_stride, config.sampling.sample_fps):
                    state.sampled_frame_count += 1
                    batch.append(frame)
                    if len(batch) == config.model.batch_size:
                        process_batch(batch)
                        batch = []
                if batch:
                    process_batch(batch)
                batch.clear()
                if state.sampled_frame_count:
                    del frame # type: ignore

            #三步筛选关键帧
            with measure(state, "postprocess_time_ms"):
                candidates, state.effective_threshold = filter_by_quantile(
                    all_scores, config.scoring.candidate_quantile)
                state.candidate_count = state.threshold_pass_count = len(candidates)
                kept = suppress_by_temporal_gap(candidates, config.postprocess.min_match_frame_gap)
                selected = select_topk(kept, config.postprocess.max_match_count)
                state.temporal_suppressed_count = len(candidates) - len(kept)
                state.topk_trimmed_count = len(kept) - len(selected)

            # Keep the resolved local file alive throughout the second pass.
            #第二遍解码：重新打开视频，只提取选中的少量关键帧
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

        #结果组装
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
        meta["keyframes"] = entries
        return OperatorResult(keyframes, meta)#返回 OperatorResult：包含 JPEG 二进制关键帧列表 + meta 元数据


def search_keyframes(video_source: str | Path | bytes, prompt: str, *,
                     negative_prompt: str | None = None, sample_fps: float = 1.0,
                     candidate_quantile: float = 0.90, negative_weight: float = 0.0,
                     min_match_frame_gap: float = 2.0, max_match_count: int = 10) -> OperatorResult:
    """文档定义的便捷入口；高级配置及模型复用使用 VideoKeyframeOperator。"""
    config = OperatorConfig()
    config.sampling.sample_fps = sample_fps
    config.scoring.candidate_quantile = candidate_quantile
    config.scoring.negative_weight = negative_weight
    config.postprocess.min_match_frame_gap = min_match_frame_gap
    config.postprocess.max_match_count = max_match_count
    return VideoKeyframeOperator(config=config).run(video_source, prompt, negative_prompt)
