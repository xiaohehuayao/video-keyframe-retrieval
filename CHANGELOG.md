# 更新记录

按 P3 → P2.5 → P2 → P0.5 倒序记录已实现的变化。P 编号表示开发阶段；Python 包版本目前仍为 `0.2.0`，不代表这些阶段都已发布为独立发行包。

## P3 — 关键帧与连续视频片段输出

对应提交：`46ac91b`；Anchor 图片输出调整：`f306b1b`。

- 将图片扩展拆为两个职责：`peak_region.py` 查找连续候选区间、从区间限量选帧；`peak_expansion.py` 组织图片分支、记录扩展信息并对跨 Anchor 的重复帧去重。
- 图片和视频共用候选峰排名及 Anchor 时间间隔约束，按两个分支中较大的 K 选择 Anchor，再各取前 K 个；两分支独立设置数量、相对阈值比例和扩展半径。
- 新增 `video_segments.py`：使用完整 Frame-Text 分数区间估计 `[start, end)`，通过边缘采样帧与外侧邻帧的时间中点确定边界，再裁剪到视频范围和搜索半径。视频区间不受 JPG 数量限制。
- 新增重叠区间合并：支持部分重叠、完全包含和链式重叠；仅边界相接时不合并，不跨空隙拼接。保留来源区间及全部 Anchor，选最高分 Anchor 作为代表；合并后不另设时长上限。
- 新增 `output/video.py`：通过 FFmpeg 将原视频连续区间导出为 H.264 MP4，有音频时编码为 AAC。采用临时文件完成后替换目标文件；失败记录状态与原因。URL/bytes 临时源在导出完成后才清理。
- 新增 `postprocess.p3_video` 配置：`enabled`、`max_anchor_count`、`relative_ratio`、`max_peak_radius_seconds`、`merge_overlapping_segments`；新增 `output.video.export_clips`、`ffmpeg_path`。
- 扩展 `create_operator()`、`search_keyframes()` 和 `run(..., output_dir=...)`，新增 `VideoSegment` 及 `OperatorResult.segments`，保持 CLI 基本调用方式。
- Meta 新增原始区间 `raw_segments`、最终区间 `segments`、共用及视频 Anchor 数量、合并前后数量、导出成功/失败数量、边界计算与视频导出耗时；可通过来源 ID 追溯合并过程。
- 图片配置调整为 `max_anchor_count: 3`、`max_frames_per_peak: 1`，只输出 Anchor 本身，最多 3 张 JPG；不影响视频分支。当前 YAML 开启 P3 与导出，视频最多 2 个 Anchor、比例 0.7、半径 10 秒；类型配置的 P3/导出默认值仍为关闭。
- 区间计算不新增 Python 依赖或模型推理；实际 MP4 导出依赖外部 FFmpeg（需 `libx264`、`aac` 编码器）。没有引入图像—图像相似度或 Intra-modal Adaptive Boundary。
- 新增区间、合并、导出及算子集成测试和 [P3 使用说明](docs/p3_video_segments.md)。实现验证为 96 项通过、2 项因缺少 FFmpeg 跳过；安装 FFmpeg 后，导出测试单独运行 4 项全部通过，包括有音频及无音频的实际 MP4。

## P2.5 — 两位小数平台峰检测

对应提交：`5c3815d`。

- 在 `peak_detection.py` 中固定使用 `round(score, 2)` 识别连续平台和比较左右邻居，减少微小分数波动形成的多个局部峰；未增加精度配置参数。
- 平台代表从固定中间帧改为原始分数最高帧；最高分并列时选择最靠近平台下标中心的位置，再以较早位置作为最终决胜条件。
- 不修改 `FrameScore` 的原始分数，Anchor 排名、邻域扩展阈值和 Meta 继续使用原始精度。
- 更新算子注释与说明，补充舍入边界、平台代表、负分及原始精度保持测试；当时验证 67 项通过。

## P2 — 多 Anchor 关键帧选择

对应提交：`d4179f8`；注释补充：`c669f5e`（标签 `p2-baseline`）。

- 从“全局分位数过滤 → 帧级 Top-K”升级为“全体采样分数 → 局部峰 → 峰分数排名 → Anchor 时间约束 → 峰级 Top-K → 邻域扩展”。P2 路径不预先筛掉低于全局分位数的帧。
- 新增 `peak_detection.py`、`anchor_selection.py`、`peak_expansion.py`。最初对完全等分平台取中间代表，后由 P2.5 更新检测精度及代表规则。
- 新增 `postprocess.p2_multi_anchor`：`enabled`、`min_anchor_gap_seconds`、`max_anchor_count`、`relative_ratio`、`max_peak_radius_seconds`、`max_frames_per_peak`。
- `min_anchor_gap_seconds` 仅约束 Anchor 之间的时间距离，替代 P2 路径中的旧最终帧间隔约束；同一 Anchor 周围允许输出时间很近的多张图片。
- 邻域扩展保持连续，低于相对阈值或超出半径立即停止；Anchor 必须保留，其余候选按分数和距离限量选择。组内时间升序、组间按 Anchor 分数排列，同一原始帧仅输出一次。
- 关闭 P2 时保留 P0.5 回退路径；旧 `candidate_quantile`、`min_match_frame_gap`、`max_match_count` 不作用于 P2/P3 选择。
- 新增峰数量、Anchor 时间/分数、扩展记录及返回数量等 Meta，扩展工厂与便捷调用接口，继续沿用二次解码及批次图片释放。
- 新增多 Anchor 测试和 [P2 使用说明](docs/p2_multi_anchor.md)，覆盖宽峰、多时间事件、近峰抑制、平台边界、扩展、去重及批次无关性；当时验证 60 项通过。

## P0.5 — 0.2.0 基线

对应提交：`2fdb42f`（标签 `v0.2.0`）；采样配置调整：`deb384c`。

- 全局分位数筛选替代固定阈值；`text_sim_thresh` 改为 `candidate_quantile`（默认 0.90）。
- 首遍仅保留轻量分数记录，第二遍顺序读取选中帧，模型仅推理一遍。
- 新增实际阈值、候选数、二次读取和 JPEG 编码统计。
- 新增 `create_operator()` 平台初始化接口，显式传入本地权重并复用模型。
- 正式命令行入口为 `python -m video_keyframe`，保留旧示例兼容入口。
- 提供 wheel、源码包和平台接入说明；模型权重与运行依赖不随代码包分发。
- 模型依赖下限调整为 Transformers 4.57，匹配当前 `dtype` 加载参数；本次验证的具体版本见 validation.md。
- 后续将 `configs/default.yaml` 的采样率由 1 FPS 调整为 2 FPS；Python 接口自身的默认值不随 YAML 自动改变。

## 当前能力边界

- P2/P2.5 减少局部重复选帧，P3 输出与文本分数相关的连续区间；局部峰或时间重叠不等同于已识别同一个真实事件。
- 未实现 No-Match 拒绝：提示词与视频无关时，仍可能产生 Anchor 和区间。P1 Qwen 改写不属于当前这条实现链路。
- 上述测试数是各阶段验证记录，不是每次修改后重新执行完整测试的声明。

# 0.2.0

- 全局分位数筛选替代固定阈值；`text_sim_thresh` 改为 `candidate_quantile`（默认 0.90）。
- 首遍仅保留轻量分数记录，第二遍顺序读取选中帧，模型仅推理一遍。
- 新增实际阈值、候选数、二次读取和 JPEG 编码统计。
- 新增 `create_operator()` 平台初始化接口，显式传入本地权重并复用模型。
- 正式命令行入口为 `python -m video_keyframe`，保留旧示例兼容入口。
- 提供 wheel、源码包和平台接入说明；模型权重与运行依赖不随代码包分发。
- 模型依赖下限调整为 Transformers 4.57，匹配当前 `dtype` 加载参数；本次验证的具体版本见 validation.md。
