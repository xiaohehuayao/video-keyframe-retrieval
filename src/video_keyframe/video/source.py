"""本地路径、bytes、HTTP(S) URL 统一为本地路径。"""
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit
from urllib.request import urlopen
from shutil import copyfileobj
from ..exceptions import VideoSourceError


class VideoSourceResolver:
    """应使用 with；退出时删除 bytes/URL 产生的临时文件。"""

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._temp = None

    def __enter__(self):
        return self

    def resolve(self, source: str | Path | bytes) -> str:
        try:
            if isinstance(source, (str, Path)) and urlsplit(str(source)).scheme not in ("http", "https"):
                path = Path(source).expanduser().resolve()
                if not path.is_file():
                    raise VideoSourceError(f"视频文件不存在: {path}")
                return str(path)
            if not isinstance(source, (str, bytes)):
                raise VideoSourceError("video_source 必须是路径、HTTP(S) URL 或 bytes")
            if self._temp is None:
                self._temp = TemporaryDirectory(prefix="video-keyframe-")
            path = Path(self._temp.name) / "source.video"
            with path.open("wb") as handle:
                if isinstance(source, bytes):
                    if not source:
                        raise VideoSourceError("视频 bytes 不能为空")
                    handle.write(source)
                else:
                    with urlopen(source, timeout=self.timeout) as response:
                        copyfileobj(response, handle)
            return str(path)
        except VideoSourceError:
            raise
        except Exception as exc:
            raise VideoSourceError(f"读取视频源失败: {exc}") from exc

    def close(self):
        if self._temp is not None:
            self._temp.cleanup()
            self._temp = None

    def __exit__(self, *args):
        self.close()
