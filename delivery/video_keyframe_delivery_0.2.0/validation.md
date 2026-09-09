# 交付验证记录

日期：2026-09-09；版本：0.2.0。

## 已完成

| 项目 | 结果 |
| --- | --- |
| 源码基础测试 | 35 passed |
| wheel 构建 | 成功；由源码发行包构建 wheel |
| 归档内容检查 | wheel / sdist 不含 backup、outputs、视频、权重、虚拟环境及缓存 |
| 独立虚拟环境安装 | 从 wheel 安装 model/dev 依赖成功，未使用 editable 安装 |
| 依赖一致性 | uv pip check 通过 |
| wheel 基础测试 | 项目目录之外执行，禁用源码 pythonpath 配置，35 passed |
| 导入隔离 | 确认 video_keyframe 来自独立环境的 site-packages |
| 实际 SigLIP2 CPU 推理 | 本地权重离线加载成功；临时合成短视频通过路径及 bytes 两种输入完成全流程 |
| 实例复用 | 两次调用复用同一个模型实例 |
| 输出 | JPEG 可解码且尺寸正确；Meta 可序列化、计数关系成立、二次读取生效 |

基础测试覆盖全局分位数及批次划分一致性、RGB 引用释放、二次读取对应关系、空输出、选中帧缺失、URL/bytes 临时文件生命周期与异常清理。URL 测试使用模拟 HTTP 响应，不代表公网连接验证。

实际模型验证使用临时生成的 2 秒、5 FPS、32×32 彩色视频，采样 2 帧，提示词 `a red image`，CPU float32，batch_size=2。模型仅做首遍编码，第二遍生成 JPEG。具体本次统计见 `smoke_result.json`；设备、Python 和包版本见 `environment.json`。设置 HF_HUB_OFFLINE 和 TRANSFORMERS_OFFLINE，未下载新权重。

加载处理器时出现 `use_fast` 未显式指定的提示，测试使用模型保存的常规处理器；该提示未导致失败。

## 范围与限制

- 没有测试目标平台 Linux/GPU/CUDA 环境；这些需要平台方验证。
- 没有进行长视频峰值内存、并发压力或吞吐压测，不能据此承诺性能。
- 合成视频测试验证链路，不代表语义检索准确率或业务阈值已标定。
- 未重新运行开发者的 test1/test2 整段视频，未覆盖已有输出。
- 代码包不是完整离线环境；不含依赖 wheelhouse 和模型权重。

## 复现基础测试

在 source 目录安装 dev 依赖后执行：

```sh
python -m pytest tests/test_factory.py tests/test_quantile.py tests/test_operator.py tests/test_sampler.py tests/test_similarity.py tests/test_temporal_gap.py tests/test_postprocess.py -q
```

平台安装 wheel 后，优先用交付示例指定本地权重和实际视频完成目标环境验收。模型文件摘要记录在 model_files.sha256.json；环境版本仅是本次验证记录，不是跨平台依赖锁定文件。
