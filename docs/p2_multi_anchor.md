# P2：Multi-Anchor Selection

全体采样帧评分 → 局部峰 → 峰分数排名 → Anchor 时间间隔约束和 Top-K → 连续邻域扩展 → 二次解码 → JPG。

## 配置与调用

`configs/default.yaml` 默认开启 P2，保留原有 `sample_fps: 2.0`。

```yaml
postprocess:
  p2_multi_anchor:
    enabled: true
    min_anchor_gap_seconds: 5.0
    max_anchor_count: 5
    relative_ratio: 0.8
    max_peak_radius_seconds: 2.0
    max_frames_per_peak: 5
```

在项目根目录运行：

```powershell
.\.venv\Scripts\python.exe -m video_keyframe "C:\Users\18202\Desktop\CECLOUD\suanzi\video_test\test3.mp4" "一个红帽子的男人倒水泥" --device cpu --output-dir outputs/model_test/p2/test3
```

输出目录包含 JPG 和 `meta.json`。Python 的 `create_operator()` 和 `search_keyframes()` 同样支持上述参数，其中开启开关名为 `p2_enabled`。

## 算法约定

- `peak_detection.detect_peaks()` 使用全部轻量分数记录，不预先做分位数过滤。连续等分平台合为一个峰，代表为中间采样帧，偶数长度取左侧中间帧。平台必须高于两侧邻居；边界仅比较存在的一侧。单帧和全等分曲线得到一个代表峰，空输入得到零个峰。时间戳必须严格递增，分数与时间戳必须有限。
- `anchor_selection.select_anchors()` 按分数降序，平分时按时间、帧号升序。与已选所有 Anchor 的时间距离均大于等于 `min_anchor_gap_seconds` 才保留，达到 `max_anchor_count` 即停止。
- `peak_expansion.expand_peaks()` 从每个 Anchor 两侧连续扫描。遇到首个低于 `peak_score * relative_ratio` 的帧或超出半径时，该方向停止。阈值和半径边界均包含。
- Anchor 本身始终保留，包括负分峰。负分乘比例可能使局部阈值高于峰值，此时通常仅保留 Anchor；不擅自改变用户指定的阈值公式。
- 超过每峰数量上限时，Anchor 之外按分数优先、距离次优先、较早时间再次优先截取。截取后的输出不承诺每个相邻采样点都保留，但扫描过程不跨越低分谷。
- Anchor 按峰分数排列，每组输出按时间升序。同一原始帧被多个 Anchor 选中时只输出一次，归入先处理的组；各 Anchor 的 Meta 仍记录自身完整扩展结果。因此各组扩展数量之和可能大于实际输出数量。

## 与 P0.5 的关系

P2 路径中，`min_anchor_gap_seconds` 替代旧的最终帧间隔约束，仅约束 Anchor。`max_anchor_count` 限制峰数；最终帧数量最多为它乘以 `max_frames_per_peak`，不会再做 Frame-level Top-K 或最终帧 Gap Suppression。

为了保留回退能力，旧配置字段和算法仍存在，但只在 `enabled: false` 时使用。旧的 `candidate_quantile`、`min_match_frame_gap`、`max_match_count` 对 P2 结果不生效。完整配置快照可能仍包含这些旧默认字段，顶层 P2 Meta 不输出旧筛选计数。

## Meta

新增 `selection_mode`、`detected_peak_count`、`selected_anchor_count`、`anchor_timestamps`、`anchor_scores`、五个算法参数、`returned_keyframe_count`。`anchors` 每项包含 `frame_index`、`timestamp`、`peak_score`、`local_threshold`、`expanded_frame_count`、`expanded_timestamps`。

P2 不提供无匹配拒绝能力；无关视频仍可能产生局部峰。最多返回 K 个 Anchor，也不保证所有重复事件一定被召回。

## 验证

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_multi_anchor.py tests/test_factory.py tests/test_quantile.py tests/test_operator.py tests/test_sampler.py tests/test_similarity.py tests/test_temporal_gap.py tests/test_postprocess.py -q
```

使用模拟分数、模拟编码器和临时生成的视频，覆盖宽峰、远距离事件、近峰抑制、Frame-level Top-K 偏置、Peak Top-K、连续扩展、近距离输出、边界平台、负分、重叠去重、配置校验、Meta、二次解码、批次无关性及图片释放。此命令不加载 SigLIP2 权重。
