# 接口与参数契约

## 主入口

```python
def search_keyframes(
    video_source: str | pathlib.Path | bytes,
    prompt: str,
    *,
    negative_prompt: str | None = None,
    sample_fps: float = 1.0,
    candidate_quantile: float = 0.90,
    negative_weight: float = 0.0,
    min_match_frame_gap: float = 2.0,
    max_match_count: int = 10,
) -> OperatorResult: ...

class VideoKeyframeOperator:
    def __init__(self, model: VisionLanguageModel | None = None,
                 config: OperatorConfig | None = None): ...
    def run(self, video_source: str | pathlib.Path | bytes, prompt: str,
            negative_prompt: str | None = None) -> OperatorResult: ...
```

`video_source` 接受本地文件路径、HTTP(S) URL 或非空视频 bytes。`prompt` 是非空字符串。`negative_prompt` 为非空字符串或 None，只有提供文本且 `negative_weight > 0` 才执行负向编码。

便捷函数每次创建算子，使用默认 CUDA 配置。实例接口延迟加载模型并复用；传入 `model` 可注入测试模型或替换后端。注入模型时 `config.model.name` 应由调用者设为实际模型标识，避免 meta 误标。运行状态不跨调用累加；同一实例不承诺并发调用安全。

## 配置参数

配置类均位于 `video_keyframe.config`；根对象 `OperatorConfig` 包含 model、sampling、scoring、postprocess、output 五个配置节。`OperatorConfig.from_yaml(path)` 加载 YAML 并拒绝未知字段，缺省字段使用下表默认值。

| YAML 字段 | 类型 | 默认值 | 约束与含义 |
| --- | --- | --- | --- |
| model.name | str | google/siglip2-base-patch16-224 | 模型 ID 或本地目录，非空 |
| model.device | str | cuda | cpu / cuda / auto；auto 根据 CUDA 可用性选择设备 |
| model.dtype | str | float16 | float32/float16/bfloat16，支持 fp32/fp16/bf16 别名；CPU 要求 float32 |
| model.batch_size | int | 32 | >0，每批图像数 |
| sampling.sample_fps | float 或 null | 1.0 | >0；高于原 FPS 时最多采每个原始帧 |
| sampling.sample_stride | int 或 null | null | >0，从第 0 帧起每隔 N 帧采样；非空时优先于 sample_fps |
| sampling.max_video_duration | float 或 null | null | >0，秒；超长视频报错，不静默截断 |
| sampling.decode_backend | str | opencv | P0 仅接受 opencv |
| scoring.candidate_quantile | float | 0.90 | 0..1 有限值；全局线性分位数，0.90 保留前约 10%，同分全部保留 |
| scoring.negative_weight | float | 0.0 | 非负有限值；0 关闭负向编码 |
| postprocess.min_match_frame_gap | float | 2.0 | 非负有限值，单位秒；0 表示不抑制 |
| postprocess.max_match_count | int | 10 | >=0；0 返回空关键帧列表，仍计算运行统计 |
| output.jpeg_quality | int | 95 | 1..100 |
| output.include_score | bool | true | 控制 meta.keyframes 的分数字段；KeyframeResult.score 始终保留 |
| output.include_frame_index | bool | true | 控制 meta 中原始帧号；关闭时 KeyframeResult.frame_index=None |

`sample_fps` 和 `sample_stride` 不能同时为 null。数值参数拒绝 NaN/Infinity。`device=auto` 不自动更改 dtype，CPU 环境应同时设置 float32。

已移除固定阈值参数 `text_sim_thresh` 和筛选模式配置；旧 YAML 中的 `text_sim_thresh`、`candidate_filter_mode` 会被拒绝。Python 调用也应改用 `candidate_quantile`。分位数在全部批次完成后计算，不依赖批次划分；相对筛选通常会保留候选，不保证视频中存在语义目标。

## 数据结构

| 类型 | 字段 | 契约 |
| --- | --- | --- |
| VideoMetadata | fps, frame_count, duration | 原始 FPS、声明帧数、估计时长（秒） |
| SampledFrame | frame_index, timestamp, image | 从 0 开始的原始帧号、秒、RGB uint8 ndarray [H,W,3] |
| ScoredFrame | frame_index, timestamp, score, positive_score, negative_score, image | score 为最终分数；关闭负向时 negative_score=None |
| FrameScore | frame_index, timestamp, score, positive_score, negative_score | 两遍流程累计的轻量记录，没有 image；ScoredFrame 仅保留供图片相关示例和旧测试使用 |
| KeyframeResult | timestamp, score, jpg_bytes, frame_index | JPEG 二进制、秒级时间戳、最终分数、可选原始帧号 |
| OperatorResult | keyframes, meta | 按分数降序的结果列表、可 JSON 序列化的字典 |

空匹配正常返回 `OperatorResult(keyframes=[], meta=...)`。`jpg_bytes` 不直接放进 JSON；示例将其另存为文件。

## 内部模块接口

| 文件 | 接口 | 输入 → 输出 / 职责 |
| --- | --- | --- |
| video/source.py | VideoSourceResolver(timeout=30).resolve(source) -> str | 统一本地路径；使用 with 或显式 close 清理临时文件；timeout 为 URL 网络超时秒数 |
| video/decoder.py | VideoDecoder.open(path) -> self | 打开视频，使用 with 或 close 释放句柄 |
| video/decoder.py | get_metadata() -> VideoMetadata | 读取元数据 |
| video/decoder.py | iter_frames() -> Iterator[SampledFrame] | 顺序解码 RGB 图像 |
| video/decoder.py | iter_selected_frames(frame_indices: set[int]) | 刚 open 后从零顺序读取，只返回选中帧，到最大目标帧停止；缺失帧时报错；read_frame_count 统计本次打开后的成功读取数 |
| video/sampler.py | FrameSampler.sample(frames, fps, sample_stride=None, sample_fps=None) | 返回采样迭代器；两个采样参数都省略时按 1 FPS |
| models/base.py | VisionLanguageModel.encode_text(texts: list[str]) | 返回 L2 归一化 CPU float32 ndarray [M,D] |
| models/base.py | VisionLanguageModel.encode_images(images: list) | 返回 L2 归一化 CPU float32 ndarray [N,D] |
| models/siglip2.py | SigLIP2Model(model_name, device, dtype, batch_size) | 处理器、模型加载、批推理、L2 归一化 |
| scoring/similarity.py | cosine_similarity_scores(image_embeddings, text_embeddings) | [N,D] × [M,D] → [N,M]；M=1 时返回 [N]；拒绝零向量与非有限值 |
| scoring/negative_prompt.py | combine_positive_negative_scores(positive_scores, negative_scores=None, negative_weight=0.0) | 同形数组 → 最终分数数组 |
| postprocess/threshold.py | filter_by_threshold(scored_frames, threshold) | 保留 score >= threshold |
| postprocess/threshold.py | filter_by_quantile(scored_frames, quantile) | 返回 (候选列表, 实际阈值)；空输入返回 ([], None)；支持 FrameScore/ScoredFrame |
| postprocess/temporal_gap.py | suppress_by_temporal_gap(frames, min_gap_seconds) | 按分数贪心选择，检查与所有已选帧的间隔 |
| postprocess/topk.py | select_topk(frames, k) | 分数降序，最多 k 帧 |
| output/jpeg.py | encode_jpeg(image, quality=95) -> bytes | RGB ndarray 或 PIL 图像 → JPEG |
| output/meta.py | build_meta(state, config) -> dict | 状态、配置快照；operator 补充 keyframes 条目 |
| utils/device.py | resolve_device_dtype(device, dtype) | 返回 torch.device 和 torch.dtype |
| utils/timing.py | measure(state, field) | 上下文管理器，累加毫秒耗时 |

## Meta 与统计口径

`meta` 包含模型标识、关键配置的平铺字段、完整 `config` 快照、`keyframes` 条目以及 `RuntimeState` 的所有字段：

| 字段 | 口径 |
| --- | --- |
| video_fps / video_duration / total_frame_count | 解码器元数据，不代表实际解码计数 |
| decoded_frame_count | 首遍实际成功解码帧数 |
| reread_frame_count | 第二遍实际读取数，包含未选中的中间帧，到最大目标帧为止 |
| sampled_frame_count | 实际采样帧数 |
| image_batch_count | 算子调用 encode_images 的批次数（最后一批可不足 batch_size） |
| image_encoded_count / text_encoded_count | 编码图像数 / 文本数（1 或 2） |
| scored_frame_count | 完成评分的帧数 |
| candidate_count / threshold_pass_count | 全局分位数过滤后的候选数，两字段值相同 |
| effective_threshold | 本次实际分位数阈值，无评分记录时为 null |
| candidate_quantile | 本次配置的分位数 |
| temporal_suppressed_count | 阈值候选中被时间间隔抑制的数量 |
| topk_trimmed_count | 时间抑制后被 Top-K 截去的数量 |
| returned_count | 最终返回数量 |
| negative_prompt_enabled | 实际是否进行了负向编码 |
| decode_time_ms | 首遍拉取下一解码帧的累计时间 |
| reread_time_ms | 第二遍打开视频及拉取选中帧的累计时间，包含读取中间帧，不含 JPEG 编码 |
| jpeg_encode_time_ms | 选中帧 JPEG 编码的累计时间 |
| image_encode_time_ms / text_encode_time_ms | 特征编码及同步复制回 CPU 的时间 |
| scoring_time_ms / postprocess_time_ms | 评分 / 阈值、时间抑制、Top-K 的累计时间 |
| total_time_ms | run 总墙钟时间，包括下载、首次延迟加载模型、两遍读取、JPEG 和资源清理；各子计时之和不等于总时间 |

成功运行时满足 `threshold_pass_count = temporal_suppressed_count + topk_trimmed_count + returned_count`。

首遍不累计图片，只持有当前批次 RGB 和全部轻量评分记录。第二遍在同一个源路径上顺序读取，仅编码选中帧，再按第一遍选出的分数顺序组装返回值。`max_match_count=0` 时仍完成评分统计，但跳过第二遍读取。

`meta.keyframes` 示例（与返回结果顺序一致）：

```json
[{"timestamp": 12.0, "score": 0.34, "positive_score": 0.34,
  "negative_score": null, "frame_index": 360}]
```

## 错误与资源生命周期

- `ConfigurationError`：配置、空 prompt、设备或 dtype 无效。
- `VideoSourceError`：输入源类型错误、文件不存在、下载失败。
- `VideoDecodeError`：打不开视频、无可解码帧、无效 FPS、超时长限制、二次读取缺失选中帧。
- `ModelError`：模型加载/编码失败或输出数量不符。

以上继承 `VideoKeyframeError`。独立数学函数参数错误使用 `ValueError`；缺少依赖可能产生 `ImportError`，文件系统操作可能产生 `OSError`。VideoSourceResolver 上下文覆盖两遍读取，首遍句柄关闭后再次打开本地路径；成功及异常路径都会释放句柄和下载临时文件。模型仅首遍执行，输入视频在一次 run 期间应保持不变。
