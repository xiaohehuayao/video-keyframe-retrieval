# 平台接入指南（0.2.0）

## 安装

解压交付 ZIP，进入交付目录。建议平台创建独立虚拟环境。CPU 通用安装：

```sh
python -m pip install "./dist/video_keyframe_retrieval-0.2.0-py3-none-any.whl[model]"
```

wheel 只包含算子代码，不捆绑第三方依赖或权重，pip 安装需要包索引访问能力。离线部署应在联网且与目标 OS、Python、设备匹配的环境单独准备依赖 wheelhouse。GPU 部署先根据目标驱动、硬件安装适配的 PyTorch，再安装本 wheel；不要直接复制开发机的 .venv。具体已验证版本见交付根目录 `validation.md` 和 `environment.json`。

## 准备权重

模型为 `google/siglip2-base-patch16-224`。代码包不含权重，可单独下载：

```sh
python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='google/siglip2-base-patch16-224', local_dir='models/siglip2-base-patch16-224')"
```

下载完成后，平台提供模型目录的绝对路径。交付根目录 `model_files.sha256.json` 记录开发机实际验证的本地模型文件摘要（不含文件内容）；下载命令默认使用仓库当前版本，若要求逐字节复现，应核对摘要或另行传输同一份权重。代码包不声明模型授权，使用模型时遵循模型仓库附带的条款。

## 初始化一次，按请求调用

```python
from video_keyframe import create_operator

# 服务启动时：model_path 由部署配置注入，必须为已存在的本地目录。
service = create_operator(
    model_path="/absolute/path/to/siglip2-base-patch16-224",
    device="cpu",                 # GPU 改为 cuda，同时 dtype=float16
    dtype="float32",
    batch_size=32,
    sample_fps=1.0,
    candidate_quantile=0.90,
    min_match_frame_gap=2.0,
    max_match_count=10,
    max_video_duration=None,
    jpeg_quality=95,
)

# 请求处理中：该实例串行调用；模型不会再次加载。
result = service.run(video_source="/data/video.mp4", prompt="娃娃机")
for frame in result.keyframes:
    print(frame.timestamp, frame.score, frame.frame_index)
    jpeg_bytes = frame.jpg_bytes   # 平台负责上传对象存储或写文件
meta = result.meta
```

`create_operator` 的默认值即上例，除 `model_path` 外均为可选关键字参数。模型加载错误在初始化时抛出；batch_size、分位数、间隔、数量等由配置校验。该入口固定负向权重为 0，P0 仅使用正向提示词。

本地目录在初始化时转换为绝对路径，无需源码目录或 YAML。不要在每个请求中重新 create_operator；不要使用会重新创建算子的 search_keyframes 作为高频请求入口。同一实例不承诺并发安全，平台串行调度或管理独立实例池；多个实例会增加 RAM/显存占用。

## 输入输出与异常

- video_source：本地路径（str/Path）、完整视频 bytes、HTTP(S) 视频直链。播放网页和 m3u8 平台解析不属于本算子能力。输入必须在本次 run 期间保持不变。
- prompt：非空字符串，不额外套模板；模型处理器按 64 token 填充或截断。
- 返回 OperatorResult，其 keyframes 是分数降序的 KeyframeResult 列表。字段为 timestamp（秒）、score（余弦分数，不是概率）、jpg_bytes（JPEG 二进制）、frame_index（0 起始）。
- meta 是可 JSON 序列化的字典；jpg_bytes 不在 meta 中。平台需先存储图片，再自行加入 URL/文件名；不要直接 JSON 序列化整个 OperatorResult。
- 算子不会写输出目录。返回空列表不是异常；模型前 10% 的相对排名也不代表目标确实存在。
- ConfigurationError、VideoSourceError、VideoDecodeError、ModelError 继承 VideoKeyframeError，可统一捕获并由平台映射业务错误码。数值函数可抛 ValueError，缺依赖可抛 ImportError，文件系统可抛 OSError。

```python
from video_keyframe.exceptions import VideoKeyframeError

try:
    result = service.run(video_source=video_path, prompt=prompt)
except VideoKeyframeError as exc:
    # 由平台记录日志并返回自己的错误结构。
    raise
```

## 算法与资源口径

首遍顺序解码并采样，模型每个采样帧编码一次，仅累计 FrameScore。全部分数计算全局线性分位数，保留 >= 阈值的记录，同分全部保留；0.90 对应前约 10%。再执行 2 秒时间抑制和 Top-K，不补足 K。

第二遍重新打开同一本地文件，从零读到最大选中帧号，仅对选中帧转换 RGB 并编码 JPEG，结果重排为分数降序。输入 URL/bytes 使用的临时文件直到第二遍结束才删除，异常路径也清理。磁盘需要容纳一份下载视频；网络源面向可信输入，服务端对 URL 访问范围、上传大小、超时和并发限制由平台负责。

RAM 主要为模型、当前批次图片、O(N) 评分记录和最多 K 张 JPEG。第二遍增加读取时间；本次未进行长视频峰值内存或平台吞吐压测。P0 时间戳为 frame_index/fps，尚不支持 VFR 精确 PTS；OpenCV 中途解码错误可能被视为 EOF。

Meta：effective_threshold 为本次阈值；candidate_count 等于 threshold_pass_count；decoded_frame_count/decode_time_ms 只计首遍；reread_frame_count 包含第二遍的中间帧；reread_time_ms 不含 JPEG 编码；jpeg_encode_time_ms 单列。总时间包括模型延迟加载（若使用原始算子接口）、解码、输出和清理；使用 create_operator 时加载在 run 之前。

## 命令行和源码

可运行交付示例（任意工作目录，显式路径）：

```sh
python examples/platform_usage.py --model-path /models/siglip2-base-patch16-224 --video /data/video.mp4 --prompt "娃娃机" --device cpu --output-dir ./result
```

也可执行 `python -m video_keyframe VIDEO PROMPT --config /path/to/config.yaml --device cpu --output-dir ./result`。安装后的 wheel 不携带项目根目录配置，应显式传 --config，并将 YAML 的 model.name 改为实际权重绝对路径。源码及可编辑安装仍可使用 source/configs/default.yaml。重复使用输出目录不会清理旧 JPG。
