from io import BytesIO
from PIL import Image


def encode_jpeg(image, quality: int = 95) -> bytes:
    """输入 RGB ndarray 或 PIL.Image。"""
    if type(quality) is not int or not 1 <= quality <= 100:
        raise ValueError("quality 必须为 1..100 的整数")
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality)
    return buffer.getvalue()
