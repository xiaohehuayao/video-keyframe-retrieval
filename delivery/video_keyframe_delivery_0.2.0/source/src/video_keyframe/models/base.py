from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class VisionLanguageModel(ABC):
    """返回 CPU float32 ndarray [N, D]，每行是 L2 归一化特征。"""

    @abstractmethod
    def encode_text(self, texts: list[str]) -> np.ndarray:
        """编码文本，输入不可为空。"""

    @abstractmethod
    def encode_images(self, images: list[Any]) -> np.ndarray:
        """编码 RGB ndarray 或 PIL 图像，输入不可为空。"""
