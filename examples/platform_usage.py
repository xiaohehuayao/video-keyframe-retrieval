"""安装 wheel 后可直接运行：仅依赖显式参数，不依赖源码目录。"""
import argparse
from pathlib import Path
import json

from video_keyframe import create_operator


def main():
    parser = argparse.ArgumentParser(description="平台接入示例：初始化一次，再调用 run")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path("platform_output"))
    args = parser.parse_args()

    # 平台应在服务启动时执行一次，并管理该实例的串行请求。
    service = create_operator(
        model_path=args.model_path,
        device=args.device,
        dtype="float32" if args.device == "cpu" else "float16",
    )
    result = service.run(video_source=args.video, prompt=args.prompt)

    # 存储属于调用方职责；平台也可将 bytes 上传到对象存储。
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(result.keyframes):
        filename = f"{index:03d}_{frame.timestamp:.3f}s.jpg"
        (args.output_dir / filename).write_bytes(frame.jpg_bytes)
        result.meta["keyframes"][index]["filename"] = filename
    (args.output_dir / "meta.json").write_text(
        json.dumps(result.meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Returned {len(result.keyframes)} frames to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
