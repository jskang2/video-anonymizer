#!/usr/bin/env python3
"""
Ultra 고성능 파이프라인 + 자동 최적화 CLI
기존 검증된 고성능 파이프라인에 하드웨어 자동 감지 및 최적화 기능 추가
"""
import argparse
import sys
from pathlib import Path
from .config import Config
from .pipeline_ultra_optimized import UltraOptimizedPipeline
from .pipeline_speed_optimized import SpeedOptimizedPipeline
from .pipeline_auto_optimized import AutoOptimizedPipeline
from .auto_optimizer import AutoConfig, HardwareProfiler

def select_optimal_pipeline(hw_info, user_preference=None):
    """하드웨어 정보를 바탕으로 최적 파이프라인 자동 선택"""
    
    if user_preference and user_preference != 'auto':
        return user_preference
    
    # GPU 메모리 기반 파이프라인 자동 선택
    if not hw_info.gpu_available:
        print("[Auto] GPU 없음 → Auto 파이프라인 선택 (CPU 최적화)")
        return 'auto'
    elif hw_info.gpu_memory_gb >= 8.0:
        print(f"[Auto] 고사양 GPU ({hw_info.gpu_memory_gb:.1f}GB) → Ultra 파이프라인 선택 (최고 품질)")
        return 'ultra'
    elif hw_info.gpu_memory_gb >= 4.0:
        print(f"[Auto] 중사양 GPU ({hw_info.gpu_memory_gb:.1f}GB) → Speed 파이프라인 선택 (속도 우선)")
        return 'speed'
    else:
        print(f"[Auto] 저사양 GPU ({hw_info.gpu_memory_gb:.1f}GB 미만) → Auto 파이프라인 선택 (안정성)")
        return 'auto'

def main():
    parser = argparse.ArgumentParser(
        description="🚀 Ultra 고성능 비디오 익명화 + 자동 최적화",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 완전 자동 최적화 (권장)
  python -m anonymizer.cli_ultra_auto --input video.mp4 --output result.mp4 --auto
  
  # 특정 파이프라인 지정 + 자동 설정
  python -m anonymizer.cli_ultra_auto --input video.mp4 --output result.mp4 --pipeline ultra --auto
  
  # 수동 설정 (기존 방식)
  python -m anonymizer.cli_ultra_auto --input video.mp4 --output result.mp4 --pipeline ultra --batch-size 16
        """
    )
    
    # 필수 인자
    parser.add_argument("--input", type=str, help="입력 비디오 파일 경로")
    parser.add_argument("--output", type=str, help="출력 비디오 파일 경로")
    
    # 자동 최적화 옵션
    parser.add_argument("--auto", action="store_true", 
                       help="🤖 하드웨어 자동 감지 및 최적 설정 적용 (권장)")
    parser.add_argument("--hardware-info", action="store_true",
                       help="하드웨어 정보만 출력하고 종료")
    
    # 파이프라인 선택
    parser.add_argument("--pipeline", type=str, 
                       choices=['auto', 'ultra', 'speed'], 
                       default='auto',
                       help="파이프라인 유형 (auto: 자동 선택, ultra: 고품질, speed: 고속도)")
    
    # 설정 파일
    parser.add_argument("--config", type=str, default="configs/gpu_optimized.yaml",
                       help="기본 설정 파일")
    
    # 고급 옵션 (자동 설정 오버라이드용)
    parser.add_argument("--batch-size", type=int, help="GPU 배치 크기 (자동 설정 무시)")
    parser.add_argument("--device", type=str, help="CUDA 디바이스")
    parser.add_argument("--confidence", type=float, help="객체 검출 신뢰도 임계값")
    parser.add_argument("--safety-margin", type=int, help="ROI 안전 여백 (픽셀)")
    parser.add_argument("--force-cpu", action="store_true", help="GPU 사용하지 않고 CPU만 사용")
    
    # 성능 옵션
    parser.add_argument("--max-performance", action="store_true",
                       help="🚀 최대 성능 모드 (시스템 자원 최대 활용)")
    parser.add_argument("--balanced", action="store_true",
                       help="⚖️ 균형 모드 (안정성과 성능 균형)")
    
    args = parser.parse_args()
    
    # 하드웨어 정보가 아닌 경우 input/output 필수 확인
    if not args.hardware_info and (not args.input or not args.output):
        parser.error("--input and --output are required unless using --hardware-info")
    
    # 하드웨어 정보만 출력하는 경우
    if args.hardware_info:
        print("🔍 하드웨어 정보 감지 중...")
        hw_info = HardwareProfiler.detect_hardware()
        
        print("\n" + "="*60)
        print("💻 감지된 하드웨어 정보")
        print("="*60)
        print(f"🖥️  CPU: {hw_info.cpu_cores}코어 / {hw_info.cpu_threads}스레드")
        print(f"🧠 RAM: {hw_info.total_ram_gb:.1f}GB (사용가능: {hw_info.available_ram_gb:.1f}GB)")
        
        if hw_info.gpu_available:
            print(f"🎮 GPU: {hw_info.gpu_name}")
            print(f"   └─ 메모리: {hw_info.gpu_memory_gb:.1f}GB")
            print(f"   └─ Compute Capability: {hw_info.gpu_compute_capability[0]}.{hw_info.gpu_compute_capability[1]}")
        else:
            print("🎮 GPU: 감지되지 않음")
        
        # 권장 파이프라인 출력
        optimal_pipeline = select_optimal_pipeline(hw_info)
        print(f"\n💡 권장 파이프라인: {optimal_pipeline}")
        return 0
    
    # 자동 최적화 모드
    if args.auto or args.pipeline == 'auto':
        print("🤖 자동 최적화 모드 활성화")
        print("🔍 하드웨어 분석 및 최적 설정 생성 중...")
        
        # 자동 설정 생성
        auto_config = AutoConfig()
        user_overrides = {}
        
        # 성능 모드에 따른 오버라이드
        if args.max_performance:
            print("🚀 최대 성능 모드 활성화")
            user_overrides.update({
                'confidence': 0.4,  # 높은 임계값으로 빠른 처리
                'batch_size_multiplier': 1.5,  # 배치 크기 50% 증가
                'cpu_workers_multiplier': 1.2,  # CPU 워커 20% 증가
            })
        elif args.balanced:
            print("⚖️ 균형 모드 활성화")
            user_overrides.update({
                'confidence': 0.3,  # 중간 임계값
                'batch_size_multiplier': 1.0,  # 기본 배치 크기
                'cpu_workers_multiplier': 1.0,  # 기본 CPU 워커
            })
        
        # 강제 CPU 모드
        if args.force_cpu:
            user_overrides['gpu_available'] = False
            print("🖥️ 강제 CPU 모드 활성화")
        
        # 사용자 수동 오버라이드 적용
        if args.batch_size:
            user_overrides['batch_size'] = args.batch_size
        if args.confidence:
            user_overrides['confidence'] = args.confidence
        if args.safety_margin:
            user_overrides['safety_margin_px'] = args.safety_margin
        if args.device:
            user_overrides['device'] = args.device
        
        optimal_settings = auto_config.generate_optimal_config(user_overrides)
        hw_info = auto_config.hardware_info
        
        # 최적 파이프라인 자동 선택
        if args.pipeline == 'auto':
            selected_pipeline = select_optimal_pipeline(hw_info)
        else:
            selected_pipeline = args.pipeline
            print(f"[Auto] 사용자 지정 파이프라인: {selected_pipeline}")
        
        # Config 객체 생성 (자동 최적화 설정 적용)
        config_overrides = {
            "input": args.input,
            "output": args.output,
            "device": "cuda:0",  # GPU 환경에서 실행하므로 고정
            "batch_size": optimal_settings.batch_size,
            "confidence": optimal_settings.confidence,
            "pose_model": optimal_settings.pose_model,
            "half_precision": optimal_settings.half_precision,
            "safety_margin_px": args.safety_margin or 12,
        }
        
    else:
        print("🔧 수동 설정 모드")
        # 기존 수동 설정 방식
        config_overrides = {
            "input": args.input,
            "output": args.output,
            "pose_model": "yolov8s-pose.pt",
            "half_precision": False
        }
        
        # 사용자 지정 값들 적용
        if args.device is not None:
            # 디바이스 문자열 정규화 (수정된 detectors.py와 호환)
            if args.device.isdigit():
                config_overrides["device"] = f"cuda:{args.device}"
            else:
                config_overrides["device"] = args.device
        if args.batch_size is not None:
            config_overrides["batch_size"] = args.batch_size
        if args.confidence is not None:
            config_overrides["confidence"] = args.confidence
        if args.safety_margin is not None:
            config_overrides["safety_margin_px"] = args.safety_margin
        
        selected_pipeline = args.pipeline if args.pipeline != 'auto' else 'auto'
        hw_info = HardwareProfiler.detect_hardware()
    
    # Config 객체 생성
    config_path = args.config if Path(args.config).exists() else None
    cfg = Config.from_yaml(config_path, config_overrides)
    
    # GPU 정보 출력
    try:
        import torch
        gpu_available = torch.cuda.is_available()
    except ImportError:
        gpu_available = False
    
    if gpu_available and not args.force_cpu:
        device_id = 0
        if hasattr(cfg, 'device') and cfg.device.startswith('cuda:'):
            device_id = int(cfg.device.split(':')[1])
        print(f"[GPU] Device: {torch.cuda.get_device_name(device_id)}")
        print(f"[GPU] Memory: {torch.cuda.get_device_properties(device_id).total_memory / 1024**3:.1f} GB")
    else:
        print("[CPU] Using CPU-only mode")
    
    print(f"\n🚀 실행 설정:")
    print(f"   📁 입력: {cfg.input}")
    print(f"   📁 출력: {cfg.output}")
    print(f"   🔧 파이프라인: {selected_pipeline}")
    print(f"   🎛️  배치 크기: {cfg.batch_size}")
    print(f"   🤖 모델: {cfg.pose_model}")
    print(f"   🎯 신뢰도: {cfg.confidence}")
    print(f"   🛡️ 안전 여백: {cfg.safety_margin_px}px")
    if hasattr(cfg, 'half_precision'):
        print(f"   ⚡ Half precision: {cfg.half_precision}")
    print("="*60)

    # 파이프라인 선택 및 실행
    try:
        if selected_pipeline == 'ultra':
            pipe = UltraOptimizedPipeline(cfg)
        elif selected_pipeline == 'speed':
            pipe = SpeedOptimizedPipeline(cfg)
        elif selected_pipeline == 'auto':
            pipe = AutoOptimizedPipeline(cfg)
        else:
            raise ValueError(f"알 수 없는 파이프라인: {selected_pipeline}")
        
        print(f"🎬 비디오 처리 시작...")
        pipe.run(cfg.input, cfg.output)
        
        print(f"\n✅ 처리 완료!")
        print(f"📁 결과 파일: {cfg.output}")
        
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        return 1

    return 0

if __name__ == "__main__":
    sys.exit(main())