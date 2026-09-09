# 视频关键帧检索

依据设计文档实现的 P0 Python 算子：输入视频和原始文本，使用 SigLIP2 找到匹配画面，返回 JPEG 字节、秒级时间戳、分数和运行统计。

```text
Video + Raw Prompt → Decode / Sampling → SigLIP2 → Cosine Similarity
                  → 全局分位数候选过滤 → Temporal Gap Suppression → Top-K
                  → 第二次顺序解码选中帧 → JPG + Timestamp + Meta
```

## 文件树

```text
video_keyframe_retrieval/
├── README.md
├── requirements.txt
├── pyproject.toml
├── configs/
│   └── default.yaml
├── docs/
│   └── api.md
├── src/video_keyframe/
│   ├── __init__.py
│   ├── __main__.py             # python -m video_keyframe 启动入口
│   ├── cli.py                  # 命令行参数、配置加载、结果保存
│   ├── operator.py             # 主流程及便捷入口
│   ├── config.py               # 类型化配置、校验、YAML 加载
│   ├── schemas.py              # 统一定义输入输出数据结构，及运行状态
│   ├── exceptions.py
│   ├── video/
│   │   ├── __init__.py
│   │   ├── source.py           # 负责处理输入，统一成可读取视频源
│   │   ├── decoder.py          # 视频读取，返回video_metadata
│   │   └── sampler.py          # FPS / stride 采样
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py             # 模型统一接口
│   │   └── siglip2.py          # SigLIP2模型，批推理及 L2 归一化
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── similarity.py #计算图文向量相似度
│   │   └── negative_prompt.py
│   ├── postprocess/
│   │   ├── __init__.py
│   │   ├── threshold.py #全局分位数阈值计算及候选过滤
│   │   ├── temporal_gap.py #去重，解决同一时间附近连续命中
│   │   └── topk.py #分数按top-k排序
│   ├── output/
│   │   ├── __init__.py
│   │   ├── jpeg.py
│   │   └── meta.py
│   └── utils/
│       ├── __init__.py
│       ├── device.py
│       └── timing.py
├── tests/
│   ├── test_sampler.py
│   ├── test_similarity.py
│   ├── test_temporal_gap.py
│   └── test_operator.py
└── examples/
    └── run_keyframe_search.py
```

## 安装与运行

Python >= 3.10；建议使用 Python 3.11/3.12 建立独立环境。

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m video_keyframe demo.mp4 "a dog running on grass"
# CPU 环境：命令行入口会同时设置 float32
python -m video_keyframe demo.mp4 "a dog running" --device cpu
```

默认配置为 CUDA + float16。首次真实推理会从 Hugging Face 下载模型，后续使用本地缓存；也可将 `model.name` 设置为本地模型目录。CUDA 运行需要支持对应硬件的 PyTorch 环境。

```python
from video_keyframe import search_keyframes

result = search_keyframes("demo.mp4", "a dog running", max_match_count=10)
for frame in result.keyframes:
    print(frame.timestamp, frame.score, len(frame.jpg_bytes))
print(result.meta)
```

高级参数、CPU 推理或多次调用时使用可复用实例：

```python
from video_keyframe import OperatorConfig, VideoKeyframeOperator

config = OperatorConfig.from_yaml("configs/default.yaml")
config.model.device = "cpu"
config.model.dtype = "float32"
operator = VideoKeyframeOperator(config=config)
result = operator.run("demo.mp4", "a dog running")
```

接口签名、参数范围、字段和异常见 [接口文档](docs/api.md)。命令行默认输出 `outputs/*.jpg` 和 `outputs/meta.json`，可用 `--output-dir` 指定目录；Python 算子接口本身不写输出文件。

正式入口为 `python -m video_keyframe`：`__main__.py` 调用 `cli.main()`，再由 CLI 调用 `VideoKeyframeOperator.run()`。原来的 `python examples/run_keyframe_search.py ...` 仍可使用，它转调用同一个 CLI。

在源码或可编辑安装环境中，默认配置定位到本项目的 `configs/default.yaml`；可以通过 `--config` 指定其他 YAML。独立安装包不附带该目录，需显式指定配置。命令行覆盖参数只影响本次运行，不写回 YAML；配置内的相对模型路径仍相对于当前工作目录，因此建议从项目根目录运行。

## P0 语义与限制

- 默认 1 FPS、batch_size=32、负向关闭、candidate_quantile=0.90、间隔 2 秒、最多 10 帧、JPEG 质量 95。
- 分数为余弦相似度，不是概率；有负向提示时为 `positive - negative_weight * negative`。
- 全部批次评分结束后，使用 NumPy 线性分位数计算全局阈值，保留 score >= 阈值的帧，再按分数贪心时间抑制和 Top-K。0.90 表示前约 10%，同分全部保留；相同分数优先较早帧，间隔恰好 2 秒可同时保留。Top-K 不补足候选数量；相对排名不能确认目标一定存在。
- P0 采用 OpenCV 和 `frame_index / fps` 时间戳，适用于恒定帧率视频；可变帧率精确 PTS 与 PyAV 留待后续实现。损坏视频的中途读取失败可能被 OpenCV 当作结束。
- 首遍仅累计 `FrameScore`（帧号、时间戳及分数），不持有 RGB 图片；图片保留在当前推理批次内。筛选结束后重新打开本地视频，从零顺序读取到最大选中帧号，仅将选中帧转成 RGB 并编码 JPEG。模型不重复推理，结果仍按分数降序返回。内存主要为当前批次图片、O(N) 轻量记录及最多 K 张 JPEG；代价是增加第二遍读取耗时。
- URL/bytes 临时视频在两遍读取完成后清理。空选集跳过第二遍；选中帧读取失败抛出 `VideoDecodeError`，不静默少返回。
- URL 输入会完整下载到临时目录，退出时自动清理。当前面向可信输入的本地算子；没有 HTTP 服务端接口。
- 模型采用 Transformers 4.x API（依赖限制 `<5`），处理器采用 64 token 定长填充和截断。参考 [Hugging Face SigLIP2 官方文档](https://huggingface.co/docs/transformers/v4.53.0/en/model_doc/siglip2)。原始 prompt 不额外加模板。

## 验证

```powershell
python -m pip install -e ".[dev]"
python -m pytest tests/test_quantile.py tests/test_operator.py tests/test_sampler.py tests/test_similarity.py tests/test_temporal_gap.py tests/test_postprocess.py
```

上述基础测试使用假模型和生成的短视频，验证采样、全局分位数、跨批次一致性、RGB 释放、二次读取、JPEG 颜色和异常清理，无需下载权重。`tests/test_siglips2.py` 是真实模型测试，`tests/test_output.py` 使用本地 test1.mp4 并保存示例图片，需单独按需执行。

## 从固定阈值迁移

删除 YAML 中的 `text_sim_thresh`、`candidate_filter_mode`，使用 `scoring.candidate_quantile: 0.90`。不再提供绝对阈值或模式切换；旧 YAML 字段会被配置加载器拒绝。Python 便捷接口改为 `search_keyframes(..., candidate_quantile=0.90)`。

Meta 新增 `effective_threshold`、`candidate_count`、`reread_frame_count`、`reread_time_ms`、`jpeg_encode_time_ms`。`threshold_pass_count` 保留为候选数的兼容字段；原有解码计数和耗时只统计首遍，总耗时包含两遍。

本次修改前的源码、配置、测试和文档备份位于 `backup/p0/`，按原目录结构保存；`manifest.json` 记录 SHA-256。视频、权重、虚拟环境和输出图片不在备份内。
