"""算子异常；调用方可统一捕获 VideoKeyframeError。"""


class VideoKeyframeError(Exception):
    """算子基础异常。"""


class ConfigurationError(VideoKeyframeError, ValueError):
    """配置或调用参数无效。"""


class VideoSourceError(VideoKeyframeError):
    """视频源无法读取或下载。"""


class VideoDecodeError(VideoKeyframeError):
    """无法打开视频或元数据无效。"""


class ModelError(VideoKeyframeError):
    """模型加载或推理失败。"""
