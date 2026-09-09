"""使用本地权重验证 SigLIP2 编码接口，不验证语义检索效果。"""
from pathlib import Path

import numpy as np
from PIL import Image

from video_keyframe.models.siglip2 import SigLIP2Model
from video_keyframe.scoring.similarity import cosine_similarity_scores


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "siglip2-base-patch16-224"


def test_siglip2_local_encoding():
    # 明确使用本地绝对路径；缺少权重时失败，不将下载缺失视为通过。
    assert MODEL_PATH.is_dir(), f"请先下载模型到：{MODEL_PATH}"

    model = SigLIP2Model(
        model_name=str(MODEL_PATH),
        device="cpu",
        dtype="float32",
        batch_size=2,
    )

    texts = ["a red image"]
    images = [
        Image.new("RGB", (224, 224), "red"),
        Image.new("RGB", (224, 224), "blue"),
    ]

    text_features = model.encode_text(texts)
    image_features = model.encode_images(images)

    assert isinstance(text_features, np.ndarray)
    assert isinstance(image_features, np.ndarray)
    assert text_features.ndim == 2
    assert image_features.ndim == 2
    assert text_features.shape[0] == 1
    assert image_features.shape[0] == 2
    assert text_features.shape[1] == image_features.shape[1]
    assert text_features.shape[1] > 0

    for features in (text_features, image_features):
        assert features.dtype == np.float32
        assert np.isfinite(features).all()
        np.testing.assert_allclose(
            np.linalg.norm(features, axis=1),
            1.0,
            atol=1e-4,
        )

    scores = cosine_similarity_scores(image_features, text_features)

    assert scores.shape == (2,)
    assert np.isfinite(scores).all()
    assert ((scores >= -1) & (scores <= 1)).all()

    # 仅检查编码与评分契约，不强制要求红图比蓝图得分高。
    print("\n模型目录：", MODEL_PATH)
    print("文本特征形状：", text_features.shape)
    print("图像特征形状：", image_features.shape)
    print("红图、蓝图与文本的相似度：", scores)
