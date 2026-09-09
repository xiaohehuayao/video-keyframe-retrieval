"""独立验证 source → decoder → sampler，不加载模型。"""
import argparse
import csv
from dataclasses import asdict
import json
from math import ceil, isfinite
from pathlib import Path
from tempfile import mkdtemp

from video_keyframe.output.jpeg import encode_jpeg
from video_keyframe.video.decoder import VideoDecoder
from video_keyframe.video.sampler import FrameSampler
from video_keyframe.video.source import VideoSourceResolver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="本地视频路径或 HTTP(S) 视频直链")
    parser.add_argument("--sample-fps", type=float, default=1.0)
    parser.add_argument("--sample-stride", type=int, default=None, help="设置时优先于 FPS")
    parser.add_argument("--save-images", type=int, default=10, help="保存前 N 张采样图，0 为不保存")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/video_inspection"))
    args = parser.parse_args()
    if not isfinite(args.sample_fps) or args.sample_fps <= 0:
        parser.error("sample-fps 必须为正有限值")
    if args.sample_stride is not None and args.sample_stride < 1:
        parser.error("sample-stride 必须为正整数")
    if args.save_images < 0:
        parser.error("save-images 必须为非负整数")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = Path(mkdtemp(prefix="run_", dir=args.output_dir)).resolve()
    decoded_count = 0
    sampled_count = 0
    sampling_valid = True
    with VideoSourceResolver() as resolver, VideoDecoder() as decoder:
        path = resolver.resolve(args.video)
        print(f"Resolved source: {path}")
        decoder.open(path)
        metadata = decoder.get_metadata()
        print("Metadata:", json.dumps(asdict(metadata)))

        def frames():
            nonlocal decoded_count
            for frame in decoder.iter_frames():
                decoded_count += 1
                yield frame

        rate = min(args.sample_fps, metadata.fps)
        with (output / "samples.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["sample_index", "frame_index", "timestamp_seconds", "image_file"])
            writer.writeheader()
            for frame in FrameSampler().sample(frames(), metadata.fps, args.sample_stride, args.sample_fps):
                if args.sample_stride is not None:
                    valid = frame.frame_index == sampled_count * args.sample_stride
                else:
                    # 采样应位于目标时刻或其后的第一张原始帧。
                    error = frame.timestamp - sampled_count / rate
                    valid = -1e-9 <= error < 1 / metadata.fps + 1e-9
                sampling_valid = sampling_valid and valid
                filename = ""
                if sampled_count < args.save_images:
                    filename = f"sample_{sampled_count:05d}_frame_{frame.frame_index:08d}_{frame.timestamp:.3f}s.jpg"
                    (output / filename).write_bytes(encode_jpeg(frame.image))
                writer.writerow(dict(sample_index=sampled_count, frame_index=frame.frame_index,
                                     timestamp_seconds=f"{frame.timestamp:.9f}", image_file=filename))
                if sampled_count < 10:
                    print(f"sample={sampled_count:4d} frame={frame.frame_index:8d} timestamp={frame.timestamp:.6f}s")
                sampled_count += 1
    expected = (ceil(decoded_count / args.sample_stride) if args.sample_stride is not None
                else int(((decoded_count - 1) / metadata.fps + 1e-9) * rate) + 1)
    report = {
        "source": path, "metadata": asdict(metadata),
        "sample_fps": args.sample_fps, "sample_stride": args.sample_stride,
        "decoded_frame_count": decoded_count, "sampled_frame_count": sampled_count,
        "expected_sample_count": expected,
        "metadata_frame_count_matches": decoded_count == metadata.frame_count,
        "sampling_grid_valid": sampling_valid,
        "sample_count_matches": sampled_count == expected,
        "saved_image_count": min(sampled_count, args.save_images),
        "timestamp_basis": "frame_index / fps (CFR assumption; not original PTS)",
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Output: {output}")
    if not all(report[key] for key in ("metadata_frame_count_matches", "sampling_grid_valid", "sample_count_matches")):
        raise SystemExit("检查存在不一致，详情见 report.json")


if __name__ == "__main__":
    main()
