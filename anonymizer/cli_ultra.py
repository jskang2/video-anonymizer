#!/usr/bin/env python3
import argparse
from pathlib import Path
from .config import Config
from .pipeline_batch_optimized import BatchOptimizedPipeline
from .pipeline_multithreaded import MultithreadedPipeline
from .pipeline_ultra_optimized import UltraOptimizedPipeline
from .pipeline_cpu_gpu_optimized import CPUGPUOptimizedPipeline
from .pipeline_speed_optimized import SpeedOptimizedPipeline

def main():
    parser = argparse.ArgumentParser(description="Ultra Optimized Video Anonymizer")
    parser.add_argument("--config", type=str, default="configs/gpu_optimized.yaml")
    parser.add_argument("--input", type=str, required=True, help="input video path")
    parser.add_argument("--output", type=str, required=True, help="output video path")
    parser.add_argument("--pipeline", type=str, choices=['batch', 'multithread', 'ultra', 'cpu-gpu', 'speed'], 
                       default='cpu-gpu', help="optimization pipeline type")
    parser.add_argument("--batch-size", type=int, help="GPU batch size")
    parser.add_argument("--device", type=str, help="CUDA device")
    parser.add_argument("--confidence", type=float, help="Object detection confidence threshold")
    parser.add_argument("--safety-margin", type=int, help="Safety margin in pixels for ROI")
    
    args = parser.parse_args()
    
    # 설정 생성
    overrides = {
        "input": args.input,
        "output": args.output,
        "pose_model": "models/yolov8s-pose.pt",
        "half_precision": False
    }

    if args.device is not None:
        overrides["device"] = int(args.device)
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.confidence is not None:
        overrides["confidence"] = args.confidence
    if args.safety_margin is not None:
        overrides["safety_margin_px"] = args.safety_margin

    config_path = args.config if Path(args.config).exists() else None
    cfg = Config.from_yaml(config_path, overrides)
    
    # GPU 정보 출력
    import torch
    if torch.cuda.is_available():
        print(f"[GPU] Device: {torch.cuda.get_device_name(cfg.device)}")
        print(f"[GPU] Memory: {torch.cuda.get_device_properties(cfg.device).total_memory / 1024**3:.1f} GB")
    
    print(f"[Config] Pipeline: {args.pipeline}")
    print(f"[Config] Batch size: {cfg.batch_size}")
    print(f"[Config] Model: {cfg.pose_model}")
    print(f"[Config] Confidence: {cfg.confidence}")
    print(f"[Config] Safety Margin: {cfg.safety_margin_px}px")

    # 파이프라인 선택 및 실행
    if args.pipeline == 'batch':
        pipe = BatchOptimizedPipeline(cfg)
    elif args.pipeline == 'multithread':
        pipe = MultithreadedPipeline(cfg)
    elif args.pipeline == 'ultra':
        pipe = UltraOptimizedPipeline(cfg)
    elif args.pipeline == 'cpu-gpu':
        pipe = CPUGPUOptimizedPipeline(cfg)
    elif args.pipeline == 'speed':
        pipe = SpeedOptimizedPipeline(cfg)
    else:
        raise ValueError(f"Unknown pipeline: {args.pipeline}")
    
    pipe.run(cfg.input, cfg.output)

if __name__ == "__main__":
    main()