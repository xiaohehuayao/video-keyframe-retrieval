"""Export continuous source intervals through FFmpeg, without a shell."""
import os
from pathlib import Path
import subprocess
from tempfile import NamedTemporaryFile
from ..exceptions import ConfigurationError


def check_ffmpeg(executable):
    try:
        result = subprocess.run([executable, "-hide_banner", "-encoders"],
                                capture_output=True, text=True, encoding="utf-8",
                                errors="replace", timeout=15,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError) as exc:
        raise ConfigurationError(f"FFmpeg is unavailable: {executable}: {exc}") from exc
    encoders = {line.split()[1] for line in result.stdout.splitlines() if len(line.split()) >= 2}
    if result.returncode or not {"libx264", "aac"}.issubset(encoders):
        raise ConfigurationError("FFmpeg must provide libx264 and aac encoders")

#导出裁剪片段
def export_video_segments(source, segments, output_dir, executable):
    clips = Path(output_dir) / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    for segment in segments:
        filename = f"{segment.segment_id:03d}_{segment.start:.3f}-{segment.end:.3f}.mp4"
        target = clips / filename
        temporary = None
        try:
            with NamedTemporaryFile(dir=clips, suffix=".mp4", delete=False) as handle:
                temporary = Path(handle.name)
            command = [executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
                       "-ss", str(segment.start), "-i", str(Path(source).resolve()),
                       "-t", str(segment.duration), "-map", "0:v:0", "-map", "0:a:0?",
                       "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                       "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-movflags", "+faststart", str(temporary.resolve())]
            result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                                    errors="replace",
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            if result.returncode or not temporary.stat().st_size:
                raise RuntimeError(result.stderr[-2000:] or "FFmpeg produced no video")
            os.replace(temporary, target)
            segment.export_status = "exported"
            segment.filename = "clips/" + filename
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            segment.export_status = "failed"
            segment.filename = None
            segment.error = str(exc)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
