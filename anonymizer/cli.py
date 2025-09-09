from __future__ import annotations
import argparse
from pathlib import Path
from .config import Config
from .pipeline import AnonymizePipeline

def parse_args():
    p = argparse.ArgumentParser(description="Video anonymizer MVP (eyes/elbows)")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--input", type=str, help="input video path", default=None)
    p.add_argument("--output", type=str, help="output video path", default=None)
    p.add_argument("--parts", type=str, help="comma-separated parts e.g. eyes,elbows", default=None)
    p.add_argument("--style", type=str, help="mosaic|gaussian|boxblur|pixelate", default=None)
    p.add_argument("--safety", type=int, help="safety margin px", default=None)
    p.add_argument("--ttl", type=int, help="ttl frames", default=None)
    p.add_argument("--gpu-optimized", action="store_true", help="Use GPU optimized settings (larger model, FP16)")
    return p.parse_args()

def main():
    args = parse_args()
    overrides = {}
    if args.input: overrides["input"] = args.input
    if args.output: overrides["output"] = args.output
    if args.parts: overrides["parts"] = [s.strip() for s in args.parts.split(",") if s.strip()]
    if args.style: overrides["style"] = args.style
    if args.safety is not None: overrides["safety_margin_px"] = args.safety
    if args.ttl is not None: overrides["ttl_frames"] = args.ttl

    # GPU 최적화 설정
    if args.gpu_optimized:
        try:
            import torch
            if torch.cuda.is_available():
                print(f"[GPU] Device: {torch.cuda.get_device_name(0)}")
                print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                overrides.update({
                    "pose_model": "yolov8s-pose.pt",  # 더 큰 모델로 변경
                    "confidence": 0.3,              # confidence 낮춤
                    "device": "0",                  # GPU 디바이스 명시
                    "half_precision": True          # FP16 사용
                })
                print("[GPU] GPU optimized settings applied: yolov8s-pose, FP16, conf=0.3")
            else:
                print("[Warning] CUDA not available, GPU optimization disabled")
        except ImportError:
            print("[Warning] PyTorch not found, GPU optimization disabled")

    cfg = Config.from_yaml(args.config, overrides)

    # ensure paths
    in_path = Path(cfg.input)
    out_path = Path(cfg.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pipe = AnonymizePipeline(cfg)
    pipe.run(str(in_path), str(out_path))

if __name__ == "__main__":
    main()
