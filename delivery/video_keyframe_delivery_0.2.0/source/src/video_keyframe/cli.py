"""正式命令行入口：参数解析、配置加载、调用算子及结果保存。"""
import argparse
import json
from pathlib import Path
from .config import OperatorConfig
from .operator import VideoKeyframeOperator


def main():
    parser = argparse.ArgumentParser(description="使用 SigLIP2 检索视频关键帧")
    parser.add_argument("video_source", help="本地视频路径或 HTTP(S) URL")
    parser.add_argument("prompt", help="原始检索文本")
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[2] / "configs/default.yaml",
                        help="配置文件路径，默认使用源码项目的 configs/default.yaml")
    parser.add_argument("--negative-prompt", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()
    config = OperatorConfig.from_yaml(args.config)
    if args.device:
        config.model.device = args.device
        if args.device == "cpu":
            config.model.dtype = "float32"
    result = VideoKeyframeOperator(config=config).run(args.video_source, args.prompt, args.negative_prompt)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, frame in enumerate(result.keyframes):
        filename = f"{index:03d}_{frame.timestamp:.3f}s.jpg"
        (args.output_dir / filename).write_bytes(frame.jpg_bytes)
        result.meta["keyframes"][index]["filename"] = filename
    (args.output_dir / "meta.json").write_text(json.dumps(result.meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"已输出 {len(result.keyframes)} 张关键帧至 {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
