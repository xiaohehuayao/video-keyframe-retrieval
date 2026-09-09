"""OpenCV 顺序解码；P0 时间戳采用 frame_index / fps。"""
from math import isfinite
from ..schemas import SampledFrame, VideoMetadata
from ..exceptions import VideoDecodeError


class VideoDecoder:
    def __init__(self):
        self._capture = None
        self.read_frame_count = 0

    def open(self, path: str) -> "VideoDecoder":
        import cv2
        self.close()
        self.read_frame_count = 0
        self._capture = cv2.VideoCapture(path)
        if not self._capture.isOpened():
            self.close()
            raise VideoDecodeError(f"无法打开视频: {path}")
        return self

    def get_metadata(self) -> VideoMetadata:
        import cv2
        if self._capture is None:
            raise VideoDecodeError("请先 open 视频")
        fps = self._capture.get(cv2.CAP_PROP_FPS)
        count = self._capture.get(cv2.CAP_PROP_FRAME_COUNT)
        if not isfinite(fps) or fps <= 0 or not isfinite(count) or count < 0:
            raise VideoDecodeError("视频 FPS 或帧数无效")
        return VideoMetadata(fps, int(count), count / fps)

    def iter_frames(self):
        import cv2
        metadata = self.get_metadata()
        index = 0
        while True:
            assert self._capture is not None
            ok, image = self._capture.read()
            if not ok:
                break
            self.read_frame_count += 1
            yield SampledFrame(index, index / metadata.fps, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            index += 1
        if index == 0:
            raise VideoDecodeError("视频没有可解码帧")

    def iter_selected_frames(self, frame_indices: set[int]):
        """刚 open 后从第零帧顺序读取，最大目标帧处停止；只转换选中帧为 RGB。"""
        import cv2
        pending = set(frame_indices)
        if any(type(index) is not int or index < 0 for index in pending):
            raise ValueError("frame_indices 必须包含非负整数")
        metadata = self.get_metadata()
        if self.read_frame_count:
            raise VideoDecodeError("选中帧读取前请重新 open 视频")
        index = 0
        while pending:
            assert self._capture is not None
            ok, image = self._capture.read()
            if not ok:
                raise VideoDecodeError(f"无法读取选中帧: {sorted(pending)}")
            self.read_frame_count += 1
            if index in pending:
                pending.remove(index)
                yield SampledFrame(index, index / metadata.fps, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            index += 1

    def close(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
