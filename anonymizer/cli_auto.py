#!/usr/bin/env python3
"""
자동 최적화 CLI - 하드웨어에 맞는 최적 설정을 자동으로 찾아 실행
"""
import argparse
import sys
from pathlib import Path
from .config import Config
from .pipeline_auto_optimized import AutoOptimizedPipeline

def main():
    parser = argparse.ArgumentParser(
        description="🚀 자동 최적화 비디오 익명화 - 하드웨어에 맞는 최적 설정 자동 적용",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 자동 최적화 실행
  python -m anonymizer.cli_auto --input video.mp4 --output result.mp4
  
  # 특정 설정 오버라이드
  python -m anonymizer.cli_auto --input video.mp4 --output result.mp4 \\
    --override batch_size=16 --override confidence=0.25
  
  # 하드웨어 정보만 확인
  python -m anonymizer.cli_auto --hardware-info
        """
    )
    
    # 필수 인자
    parser.add_argument("--input", "-i", type=str, 
                       help="입력 비디오 파일 경로")
    parser.add_argument("--output", "-o", type=str,
                       help="출력 비디오 파일 경로")
    
    # 기본 설정
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="기본 설정 파일 (기본값: configs/default.yaml)")
    parser.add_argument("--style", type=str, default="mosaic",
                       choices=["mosaic", "gaussian", "boxblur", "pixelate"],
                       help="익명화 스타일 (기본값: mosaic)")
    parser.add_argument("--safety-margin", type=int, default=12,
                       help="안전 여백 픽셀 (기본값: 12)")
    
    # 자동 최적화 관련
    parser.add_argument("--override", action="append", metavar="KEY=VALUE",
                       help="자동 설정 오버라이드 (예: batch_size=16, confidence=0.25)")
    parser.add_argument("--hardware-info", action="store_true",
                       help="하드웨어 정보만 출력하고 종료")
    parser.add_argument("--no-cache", action="store_true",
                       help="설정 캐시 사용하지 않기")
    parser.add_argument("--force-cpu", action="store_true",
                       help="GPU 사용하지 않고 CPU만 사용")
    
    # 고급 옵션
    parser.add_argument("--benchmark", action="store_true",
                       help="여러 배치 크기로 벤치마크 실행")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="상세 출력")
    
    args = parser.parse_args()
    
    # 하드웨어 정보만 출력하는 경우
    if args.hardware_info:
        from .auto_optimizer import HardwareProfiler, AutoConfig
        
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
        
        print("\n⚙️ 권장 설정 생성 중...")
        auto_config = AutoConfig()
        optimal_settings = auto_config.generate_optimal_config()
        
        return 0
    
    # 벤치마크 모드
    if args.benchmark:
        print("🏃‍♂️ 벤치마크 모드는 아직 구현되지 않았습니다.")
        return 1
    
    # 필수 인자 검증
    if not args.input or not args.output:
        print("❌ 오류: --input과 --output 인자가 필요합니다.")
        print("도움말: python -m anonymizer.cli_auto --help")
        return 1
    
    # 입력 파일 존재 확인
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ 오류: 입력 파일을 찾을 수 없습니다: {args.input}")
        return 1
    
    # 출력 디렉토리 생성
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # 기본 설정 로드
        print(f"⚙️ 기본 설정 로드 중: {args.config}")
        cfg = Config.from_yaml(args.config)
        
        # CLI 인자로 설정 오버라이드
        cfg.style = args.style
        cfg.safety_margin_px = args.safety_margin
        
        # 사용자 오버라이드 파싱
        user_overrides = {}
        if args.override:
            for override in args.override:
                try:
                    key, value = override.split('=', 1)
                    # 타입 추론
                    if value.lower() in ('true', 'false'):
                        user_overrides[key] = value.lower() == 'true'
                    elif value.replace('.', '').isdigit():
                        user_overrides[key] = float(value) if '.' in value else int(value)
                    else:
                        user_overrides[key] = value
                except ValueError:
                    print(f"⚠️ 경고: 잘못된 오버라이드 형식 무시됨: {override}")
        
        # 강제 CPU 모드
        if args.force_cpu:
            user_overrides['gpu_available'] = False
            print("🖥️ 강제 CPU 모드 활성화")
        
        # 캐시 비활성화
        if args.no_cache:
            user_overrides['use_cache'] = False
            print("🚫 설정 캐시 비활성화")
        
        # 상세 출력
        if args.verbose:
            print(f"📁 입력: {input_path}")
            print(f"📁 출력: {output_path}")
            print(f"🎨 스타일: {args.style}")
            print(f"🛡️ 안전 여백: {args.safety_margin}px")
            if user_overrides:
                print(f"🔧 사용자 오버라이드: {user_overrides}")
        
        # 자동 최적화 파이프라인 실행
        print("\n🚀 자동 최적화 파이프라인 시작")
        print("="*60)
        
        pipeline = AutoOptimizedPipeline(cfg, user_overrides)
        pipeline.run(str(input_path), str(output_path))
        
        print("\n✅ 처리 완료!")
        print(f"📁 결과 파일: {output_path}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️ 사용자에 의해 중단됨")
        return 1
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())