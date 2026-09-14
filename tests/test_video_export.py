from pathlib import Path
from types import SimpleNamespace
import shutil
import subprocess
import pytest
from video_keyframe.output.video import check_ffmpeg, export_video_segments
from video_keyframe.exceptions import ConfigurationError
from test_video_segments import segment


def test_preflight_missing(monkeypatch):
    def missing(*a, **k):
        raise FileNotFoundError("missing")
    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(ConfigurationError):
        check_ffmpeg("missing")


def test_export_success_and_failure(tmp_path, monkeypatch):
    segments = [segment(0, 1, 3), segment(1, 4, 5)]
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert not kwargs.get("shell", False)
        assert "0:a:0?" in command and "libx264" in command
        if len(calls) == 1:
            Path(command[-1]).write_bytes(b"mp4")
            return SimpleNamespace(returncode=0, stderr="")
        return SimpleNamespace(returncode=1, stderr="encoder failed")
    monkeypatch.setattr(subprocess, "run", run)
    export_video_segments(tmp_path / "source file.mp4", segments, tmp_path, "ffmpeg")
    assert segments[0].export_status == "exported"
    assert (tmp_path / segments[0].filename).exists()
    assert segments[1].export_status == "failed" and segments[1].filename is None
    assert "encoder failed" in segments[1].error
    assert len(list((tmp_path / "clips").iterdir())) == 1


@pytest.mark.parametrize("with_audio", [False, True])
def test_real_ffmpeg_export(tmp_path, with_audio):
    executable = shutil.which("ffmpeg")
    if executable is None:
        pytest.skip("FFmpeg not installed; real export test requires FFmpeg")
    import cv2
    check_ffmpeg(executable)
    source = tmp_path / "source.mp4"
    command = [executable, "-y", "-f", "lavfi", "-i", "testsrc2=size=64x64:rate=10:duration=4"]
    if with_audio:
        command += ["-f", "lavfi", "-i", "sine=frequency=440:duration=4"]
    command += ["-c:v", "libx264", "-c:a", "aac", str(source)]
    subprocess.run(command, check=True, capture_output=True)
    segments = [segment(0, .5, 2.5)]
    export_video_segments(source, segments, tmp_path, executable)
    assert segments[0].export_status == "exported", segments[0].error
    output = tmp_path / segments[0].filename
    capture = cv2.VideoCapture(str(output))
    try:
        assert capture.isOpened()
        fps = capture.get(cv2.CAP_PROP_FPS)
        count = 0
        while capture.read()[0]:
            count += 1
        assert abs(count / fps - 2) <= .11
    finally:
        capture.release()
    audio = subprocess.run([executable, "-i", str(output), "-map", "0:a:0", "-f", "null", "-"], capture_output=True)
    assert (audio.returncode == 0) == with_audio
