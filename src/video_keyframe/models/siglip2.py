"""Hugging Face Transformers 4.x SigLIP2 适配器。"""
from .base import VisionLanguageModel
from ..exceptions import ModelError
from ..utils.device import resolve_device_dtype


class SigLIP2Model(VisionLanguageModel):
    def __init__(self, model_name="google/siglip2-base-patch16-224",
                 device="cuda", dtype="float16", batch_size=32):
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size 必须为正整数")
        self.model_name = model_name
        self.batch_size = batch_size
        self.device, self.dtype = resolve_device_dtype(device, dtype)
        try:#初始化
            from transformers import AutoModel, AutoProcessor
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, dtype=self.dtype).to(self.device).eval()
        except Exception as exc:
            raise ModelError(f"SigLIP2 加载失败: {exc}") from exc

    def _encode(self, items, kind):
        import numpy as np
        import torch
        if not items:
            raise ValueError("模型输入不能为空")
        outputs = []
        try:
            with torch.inference_mode():
                for start in range(0, len(items), self.batch_size):
                    batch = items[start:start + self.batch_size]
                    if kind == "text":
                        inputs = self.processor(text=batch, padding="max_length", max_length=64,
                                                truncation=True, return_tensors="pt")
                        forward = self.model.get_text_features
                    else:
                        inputs = self.processor(images=batch, return_tensors="pt")
                        forward = self.model.get_image_features
                    inputs = {key: value.to(device=self.device, dtype=self.dtype)
                              if value.is_floating_point() else value.to(self.device)
                              for key, value in inputs.items()}
                    features = forward(**inputs).float()
                    features = torch.nn.functional.normalize(features, dim=-1)
                    outputs.append(features.cpu().numpy())
            return np.concatenate(outputs, axis=0)
        except Exception as exc:
            raise ModelError(f"SigLIP2 {kind} 编码失败: {exc}") from exc

    def encode_text(self, texts):
        return self._encode(texts, "text")

    def encode_images(self, images):
        return self._encode(images, "images")
