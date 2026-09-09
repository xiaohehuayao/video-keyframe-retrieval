# 0.2.0

- 全局分位数筛选替代固定阈值；`text_sim_thresh` 改为 `candidate_quantile`（默认 0.90）。
- 首遍仅保留轻量分数记录，第二遍顺序读取选中帧，模型仅推理一遍。
- 新增实际阈值、候选数、二次读取和 JPEG 编码统计。
- 新增 `create_operator()` 平台初始化接口，显式传入本地权重并复用模型。
- 正式命令行入口为 `python -m video_keyframe`，保留旧示例兼容入口。
- 提供 wheel、源码包和平台接入说明；模型权重与运行依赖不随代码包分发。
- 模型依赖下限调整为 Transformers 4.57，匹配当前 `dtype` 加载参数；本次验证的具体版本见 validation.md。
