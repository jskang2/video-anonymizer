# 📂 프로젝트 구조

## 전체 구조 개요

```
video-anonymizer/
├── 📁 anonymizer/              # 핵심 익명화 모듈
├── 📁 configs/                 # 설정 파일들
├── 📁 data/                    # 입력 비디오 데이터
├── 📁 docs/                    # 문서화
├── 📁 models/                  # YOLO 모델 파일들
├── 📁 output/                  # 처리된 비디오 출력
├── 📁 scripts/                 # 유틸리티 스크립트
├── 📁 tests/                   # 테스트 파일들
├── 🐳 Dockerfile.gpu           # GPU Docker 환경
├── 🐳 Dockerfile.cpu           # CPU Docker 환경
├── ⚙️ Makefile                 # 빌드 및 실행 명령어
├── 📋 README.md                # 프로젝트 메인 가이드
├── 📋 CLAUDE.md                # Claude Code용 가이드
├── 📦 requirements-gpu.txt     # GPU 환경 의존성
├── 📦 requirements-cpu.txt     # CPU 환경 의존성
└── 🤖 auto_config_cache.json  # AI 자동 최적화 캐시
```

## 📁 핵심 모듈 (`anonymizer/`)

### 🎯 메인 컴포넌트
```
anonymizer/
├── 🚀 pipeline.py                    # 기본 익명화 파이프라인
├── 🚀 pipeline_auto_optimized.py     # AI 자동 최적화 파이프라인
├── 🚀 pipeline_speed_optimized.py    # 속도 최적화 파이프라인
├── 🚀 pipeline_ultra_optimized.py    # 품질 최적화 파이프라인
├── 🤖 auto_optimizer.py              # AI 자동 최적화 시스템
├── 🎯 detectors.py                   # YOLO + Haar 검출기
├── 📐 roi.py                         # ROI(관심영역) 관리
├── ⚙️ config.py                      # 설정 관리
├── 🎬 video_io.py                    # 비디오 입출력
├── 🎨 viz.py                         # 시각화 도구
└── ⚡ gpu_accelerated_ops.py         # GPU 가속 연산
```

### 🖥️ CLI 인터페이스
```
anonymizer/
├── 💻 cli.py                         # 기본 CLI
├── 💻 cli_auto.py                    # 자동 최적화 CLI
├── 💻 cli_ultra.py                   # 품질 최적화 CLI
└── 💻 cli_ultra_auto.py              # 울트라 자동 최적화 CLI
```

## 📁 설정 파일 (`configs/`)

```
configs/
├── ⚙️ default.yaml                   # 기본 설정
├── ⚙️ auto_optimized.yaml            # 자동 최적화 설정
└── ⚙️ gpu_optimized.yaml             # GPU 최적화 설정
```

## 📁 AI 모델 (`models/`)

```
models/
├── 🤖 yolov8n-pose.pt               # YOLO 나노 모델 (6.6MB)
└── 🤖 yolov8s-pose.pt               # YOLO 스몰 모델 (23MB)
```

**모델 자동 선택 로직:**
- GPU 메모리 ≥4GB → `yolov8s-pose.pt` (고성능)
- GPU 메모리 <4GB → `yolov8n-pose.pt` (호환성)

## 📁 문서화 (`docs/`)

```
docs/
├── 📋 DEVELOPMENT.md                 # 개발자 가이드
├── 📋 PROJECT_STRUCTURE.md           # 프로젝트 구조 (본 문서)
├── 📋 WSL2-가이드.md                 # WSL2 설정 가이드
├── 📋 개발환경-세팅.md               # 개발 환경 설정
├── 📋 컨셉.md                        # 프로젝트 컨셉
├── 📋 테스트-방법.md                 # 테스트 방법론
└── 📋 프로젝트-구조.md               # 구조 설명 (한국어)
```

## 📁 테스트 (`tests/`)

```
tests/
└── 🧪 test_smoke.py                  # 스모크 테스트
```

## 🐳 Docker 환경

### GPU Docker (`Dockerfile.gpu`)
- **기반:** PyTorch 2.0.1 + CUDA 11.7 + cuDNN 8
- **크기:** 13GB
- **성능:** 93+ FPS (RTX 3060 Ti 기준)
- **특징:** AI 자동 최적화 완전 지원

### CPU Docker (`Dockerfile.cpu`)  
- **기반:** Python 3.10-slim
- **크기:** 훨씬 가벼움
- **성능:** ~5 FPS
- **용도:** GPU 없는 환경, 호환성 테스트

## ⚙️ 자동화 시스템

### 🤖 AI 자동 최적화
- **하드웨어 감지:** CPU, GPU, RAM 자동 분석
- **설정 최적화:** 배치크기, 워커수, 임계값 자동 계산
- **캐시 관리:** `auto_config_cache.json`에 설정 저장
- **성능 모니터링:** 실시간 리소스 사용률 추적

### 📦 의존성 관리
- **GPU 환경:** `requirements-gpu.txt` - PyTorch GPU 버전
- **CPU 환경:** `requirements-cpu.txt` - 가벼운 CPU 버전

## 🏗️ 아키텍처 흐름

```
입력 비디오 → VideoReader → Detection (YOLO+Haar) → ROI 생성 
    → TTL 로직 → 마스크 생성 → 익명화 처리 → VideoWriter → 출력
```

### 🎯 핵심 검출 시스템
1. **YOLO Pose Detection:** 팔꿈치 키포인트 검출
2. **Haar Cascade:** 얼굴/눈 검출
3. **TTL 로직:** 검출 실패시 이전 ROI 유지 (깜빡임 방지)
4. **적응형 ROI:** 팔꿈치-손목 거리 기반 동적 크기 조정

### ⚡ 성능 최적화
- **배치 처리:** GPU 메모리 기반 배치 크기 자동 조정
- **병렬 처리:** CPU 코어 기반 워커 수 최적화
- **메모리 관리:** 프레임별 처리로 메모리 효율성 확보
- **모델 캐싱:** 컨테이너 재시작시 모델 재다운로드 방지

## 🚀 주요 실행 명령어

```bash
# 🤖 AI 자동 최적화 (권장)
make run-auto-speed IN=video.mp4      # 최고속도
make run-auto-ultra IN=video.mp4      # 최고품질

# 🔧 수동 최적화 (전문용)
make run-ultra IN=video.mp4           # 품질 우선
make run-speed IN=video.mp4           # 속도 우선

# 📊 시스템 정보
make hardware-info                    # 하드웨어 감지
make container-status                 # 컨테이너 상태
```

이 구조는 **AI 자동 최적화**를 통해 사용자 하드웨어에 맞춰 자동으로 최적 성능을 달성하도록 설계되었습니다.