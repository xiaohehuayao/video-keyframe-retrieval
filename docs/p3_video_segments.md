# P3：图片与连续视频片段检索

P3 复用 P2.5 的 Frame-Text 分数与连续区间扩展，不新增模型、图像特征缓存或图像—图像相似度。模型仅推理一次。Anchor 排名和时间间隔筛选共用，选择数量为两个分支 K 的较大值，再分别取前 K 个。

## 配置和运行

现有图片配置及 sample_fps 保持用户当前设置。视频功能默认开启；在 `configs/default.yaml` 设置：

```yaml
postprocess:
  p3_video:
    enabled: true
    max_anchor_count: 2
    relative_ratio: 0.7
    max_peak_radius_seconds: 10.0
    merge_overlapping_segments: true
output:
  video:
    export_clips: false
    ffmpeg_path: ffmpeg
```

只计算区间时，不需安装新依赖。导出 MP4 时将 `export_clips` 改为 `true`，并提供带 `libx264`、`aac` 编码器的 FFmpeg；可把 `ffmpeg_path` 设为本机可执行文件绝对路径。没有新增 Python 包或升级模型依赖。

先检查可执行文件：

```powershell
ffmpeg -version
ffmpeg -encoders
```

运行方式保持不变：

```powershell
.\.venv\Scripts\python.exe -m video_keyframe "C:\Users\18202\Desktop\CECLOUD\suanzi\video_test\test3.mp4" "一个红帽子的男人倒水泥" --device cpu --output-dir outputs/model_test/p3/test3
```

输出为目录下的原有 JPG、`meta.json`，以及开启导出后的 `clips/000_<start>-<end>.mp4`。相同文件名在成功导出时替换；不同名称的历史结果不自动清理，建议每次比较使用新目录，并以本次 Meta 为准。

## 共用区间、独立参数

`peak_region.find_contiguous_region()` 保留 Anchor，沿时间左右扫描，低于 `anchor.score * relative_ratio` 或超过半径立即停止。使用原始分数，不使用找峰阶段的两位小数。负分 Anchor 仍保留，其他帧按原公式判断。

图片调用 `select_region_keyframes()` 限制每峰 JPG 数量。视频用自身比例与半径重新扩展，不受图片的 `max_frames_per_peak` 限制。

视频起止时间取边缘通过帧与外侧采样帧的中点，裁剪到视频范围和 Anchor 半径内，区间采用 `[start, end)`。原始帧号字段表示边缘通过的采样帧，不等于裁剪视频的首尾帧。只有一个采样帧时仍可形成非零区间；零半径等导致非正时长时记录跳过原因，不导出。

## 重叠合并

按 start 排序，严格重叠时取并集，支持包含、相同区间和链式重叠。仅边界相接时不合并，不跨越空隙。保留全部来源区间 ID 和 Anchor，代表 Anchor 取原始分数最高者，同分取较早时间、较小帧号。

合并后按时间排序并重新编号，没有额外时长上限。`max_anchor_count` 限制候选 Anchor 数，不保证最终视频数量；合并可能减少数量。`max_peak_radius_seconds` 只限制合并前区间。

## Python 接口

`create_operator()` 和 `search_keyframes()` 新增：`p3_enabled`、`video_max_anchor_count`、`video_relative_ratio`、`video_max_peak_radius_seconds`、`merge_overlapping_segments`、`export_clips`、`ffmpeg_path`。

`operator.run(source, prompt, negative_prompt=None, *, output_dir=None)` 保留原位置参数。实际导出时传入 output_dir；`search_keyframes()` 也支持该参数。`OperatorResult.segments` 为 `VideoSegment` 列表，默认空列表，保留旧的 keyframes、meta 接口。

FFmpeg 提前检查后再加载模型；视频导出在临时源清理之前完成。采用 H.264、可选第一条音轨 AAC，奇数画面尺寸补齐到偶数。导出器使用参数列表调用 subprocess，不经过 shell。每个片段先写临时文件，成功后替换目标；失败删除临时文件并记录错误，继续其他片段。CLI 保存 Meta 后，如有片段失败以退出码 1 结束。API 调用方应检查 export_status。

## Meta

顶层新增 `p3_video_enabled`、`shared_anchor_count`、`video_selected_anchor_count`、`raw_segment_count`、`merged_segment_count`、`segment_count`、`exported_clip_count`、`failed_clip_count`、`video_boundary_time_ms`、`video_export_time_ms`。

`raw_segments` 保存每个 Anchor 的原始区间：segment_id、anchor、start/end/duration、first/last_sample_frame_index、region_sample_count、local_threshold、left/right_stop_reason、boundary_method、valid、skip_reason。

`segments` 保存最终区间：segment_id、start/end/duration、source_segment_ids、anchors、representative_anchor、boundary_method、export_status、filename、error。根据 source_segment_ids 可追溯所有原始区间。完整参数位于 config 快照。

停止原因为 below_threshold、radius_limit、video_boundary；无正时长时 skip_reason 为 nonpositive_duration。导出状态为 not_requested、exported、failed。区间时间是目标裁剪范围，不宣称媒体容器时长和采样边界毫无误差。

## 测试与限制

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_peak_region.py tests/test_video_segments.py tests/test_video_export.py tests/test_p3_operator.py tests/test_multi_anchor.py tests/test_factory.py tests/test_quantile.py tests/test_operator.py tests/test_sampler.py tests/test_similarity.py tests/test_temporal_gap.py tests/test_postprocess.py -q
```

真实导出测试自动生成临时视频，分别验证有音频、无音频、时长和可解码性；未安装 FFmpeg 时跳过这两项，其他测试使用模拟导出器。测试不加载 SigLIP2 权重。

视频内部保存原始连续帧，但采样分数不能保证整个区间语义持续相关。当前不包含 No-Match 判断、语义事件识别、合并后时长上限或 Intra-modal Boundary。

FFmpeg 参数依据：[官方命令行文档](https://ffmpeg.org/ffmpeg.html)。
