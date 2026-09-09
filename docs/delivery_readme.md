# 视频关键帧检索交付包 0.2.0

交付目标：平台安装算子，启动时加载 SigLIP2，按请求输入视频和提示词，得到 JPEG bytes、时间戳、分数和 Meta。HTTP 服务、存储和调度由平台适配。

## 目录

```text
README.md                    本说明
CHANGELOG.md                 版本变更
dist/                        wheel 和源码发行包
source/                      可编辑完整源码、配置示例、测试
examples/platform_usage.py   平台调用示例
docs/api.md                  数据与接口细节
docs/integration.md          安装及平台集成步骤
validation.md               本次验证结果与范围
environment.json            验证环境版本（记录，不是跨平台锁文件）
model_files.sha256.json      本地验证权重的文件摘要，不包含权重
SHA256SUMS.json              交付文件校验清单
```

## 安装与使用

```sh
python -m pip install "./dist/video_keyframe_retrieval-0.2.0-py3-none-any.whl[model]"
```

CUDA 环境先安装匹配目标硬件的 PyTorch。代码包不包含运行依赖和权重，不能作为完整离线环境直接使用。

```python
from video_keyframe import create_operator

service = create_operator(
    model_path="/absolute/path/to/siglip2-base-patch16-224",
    device="cpu",
    dtype="float32",
)
result = service.run("/data/test.mp4", "娃娃机")
for frame in result.keyframes:
    print(frame.timestamp, frame.score)
    # 平台保存 frame.jpg_bytes 或上传对象存储。
```

初始化只做一次；同一实例串行复用。完整参数、错误处理和部署约束见 [集成指南](docs/integration.md)。

## 验证与重新构建

基础测试不依赖真实权重，源代码目录下执行：

```sh
python -m pip install -e "./source[dev]"
python -m pytest source/tests/test_factory.py source/tests/test_quantile.py source/tests/test_operator.py source/tests/test_sampler.py source/tests/test_similarity.py source/tests/test_temporal_gap.py source/tests/test_postprocess.py
```

`source/tests/test_siglips2.py` 需要本地模型；`source/tests/test_output.py` 需要另行提供测试视频并会保存图片，不属于默认基础验证。不要直接运行所有测试后将缺资源失败视为安装失败。

从源码重新构建：

```sh
python -m pip install build
python -m build source
```

没有打包开发环境、backup、outputs、个人视频或模型权重。模型下载方法及版本复现方式见集成指南。本次具体测试结果以 validation.md 为准。
