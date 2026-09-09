from ..exceptions import ConfigurationError


def resolve_device_dtype(device: str, dtype: str):
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in ("cpu", "cuda"):
        raise ConfigurationError("device 必须为 cpu/cuda/auto")
    if device == "cuda" and not torch.cuda.is_available():
        raise ConfigurationError("CUDA 不可用，请设置 device=cpu、dtype=float32")
    aliases = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}
    dtype = aliases.get(dtype, dtype)
    if dtype not in ("float32", "float16", "bfloat16"):
        raise ConfigurationError("dtype 无效")
    if device == "cpu" and dtype != "float32":
        raise ConfigurationError("P0 CPU 推理请使用 float32")
    return torch.device(device), getattr(torch, dtype)
