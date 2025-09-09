#!/usr/bin/env python3
import argparse
from pathlib import Path
from .config import Config
from .pipeline_optimized import OptimizedAnonymizePipeline

def main():
    parser = argparse.ArgumentParser(description="GPU 최적화된 Video anonymizer MVP (eyes/elbows)")
    parser.add_argument("--config", type=str, default="configs/gpu_optimized.yaml")
    parser.add_argument("--input", type=str, help="input video path")
    parser.add_argument("--output", type=str, help="output video path")
    parser.add_argument("--parts", type=str, help="comma-separated parts e.g. eyes,elbows")
    parser.add_argument("--style", type=str, help="mosaic|gaussian|boxblur|pixelate")
    parser.add_argument("--safety", type=int, help="safety margin px")
    parser.add_argument("--ttl", type=int, help="ttl frames")
    parser.add_argument("--batch-size", type=int, help="GPU batch size for processing")
    parser.add_argument("--device", type=str, help="CUDA device (0, 1, etc.)")
    parser.add_argument("--half-precision", action="store_true", help="Use FP16 for faster inference")
    parser.add_argument("--confidence", type=float, help="YOLO confidence threshold")
    
    args = parser.parse_args()
    
    # 설정파일 확인
    config_path = args.config
    if not Path(config_path).exists():
        print(f"Warning: Config file {config_path} not found, using default config")
        config_path = None
    
    overrides = {}
    if args.input: overrides["input"] = args.input
    if args.output: overrides["output"] = args.output
    if args.parts: overrides["parts"] = args.parts.split(",")
    if args.style: overrides["style"] = args.style
    if args.safety: overrides["safety_margin_px"] = args.safety
    if args.ttl: overrides["ttl_frames"] = args.ttl
    if args.batch_size: overrides["batch_size"] = args.batch_size
    if args.device: overrides["device"] = args.device
    if args.half_precision: overrides["half_precision"] = True
    if args.confidence: overrides["confidence"] = args.confidence
    
    cfg = Config.from_yaml(config_path, overrides)
    
    # GPU 정보 출력
    import torch
    if torch.cuda.is_available():
        print(f"[GPU] Device: {torch.cuda.get_device_name(0)}")
        print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"[Config] Batch size: {cfg.batch_size}, Half precision: {cfg.half_precision}")
    else:
        print("[Warning] CUDA not available, falling back to CPU")
    
    pipe = OptimizedAnonymizePipeline(cfg)
    pipe.run(cfg.input, cfg.output)

if __name__ == "__main__":
    main()