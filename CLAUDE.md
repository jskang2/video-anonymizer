# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video Anonymizer MVP - A Docker-based video processing pipeline for automatic anonymization of specific body parts (eyes, elbows) using YOLO pose estimation and Haar cascade detection with OpenCV.

## Git 워크플로우

### github 푸쉬를 위해 다음 정보 사용:
GIT HUB의 Personal Access Token: [보안상 제거됨 - 별도 관리 필요]

Github 주소 : https://github.com/jskang2/video-anonymizer

### Git 설정 및 푸시 규칙
.git 이 존재하지 않으면 Git 저장소 초기화 할것 (git init)
main 브랜치 사용할 것
파일 생성 또는 수정시, 파일 생성 또는 수정한 후, git add와 commit 수행할 것
파일 삭제시 git rm 및 commit 사용할 것

원격 저장소에 푸시할 때, 먼저 HTTP 버퍼 크기를 늘리고 조금 씩 나누어 푸시할 것. 에러 시 작은 변경사항만 포함하는 새커밋을 만들어 푸시할 것

## 🚀 GPU Docker 빌드 및 테스트 완료 상태

### ✅ 빌드 성공 확인 (2025.01.20)
- **이미지명**: `video-anonymizer-gpu:slim` (13GB)
- **기반**: PyTorch 2.0.1 + CUDA 11.7 + cuDNN 8
- **상태**: 모든 테스트 성공적으로 완료

### 🔧 주요 수정사항
1. **Dockerfile.gpu 환경변수 추가**: 
   - `ENV DEBIAN_FRONTEND=noninteractive`
   - `ENV TZ=Asia/Seoul`
2. **OpenGL 라이브러리 추가**: `libgl1-mesa-glx`
3. **requirements-cpu.txt 수정**: PyTorch CPU 버전 태그 제거

### ✅ 테스트 검증 완료
- **OpenCV**: v4.11.0 정상 로딩 ✅
- **하드웨어 감지**: RTX 3060 Ti (8GB) 자동 감지 ✅  
- **자동 최적화**: Speed 파이프라인 권장 ✅
- **CLI 도구**: 모든 명령어 정상 작동 ✅

### 🎮 지원 하드웨어
- **GPU**: NVIDIA GeForce RTX 3060 Ti (8.0GB)
- **CPU**: 16코어/32스레드, **RAM**: 62.7GB
- **권장 파이프라인**: Speed (고속도 처리)

## 🎬 영상 처리 테스트 결과 (2025.01.20)

### ✅ 성공한 처리 방법들

| 방법 | 처리 시간 | 처리 속도 | 안정성 | 추천도 |
|------|----------|-----------|--------|---------|
| **Auto Speed** | 16.42초 | **18.27 FPS** | 높음 | ⭐⭐⭐⭐⭐ |
| **Basic CLI** | ~30초 | ~10 FPS | 매우높음 | ⭐⭐⭐⭐ |
| Auto Ultra | 완료 | - | 보통 | ⭐⭐⭐ |

### 🚀 권장 실행 명령어

#### **방법 1: 자동 최적화 Speed (최고 성능)**
```bash
make run-auto-speed IN=data/your_video.mp4
# 출력: output/result_auto_speed.mp4
```

#### **방법 2: 기본 CLI (최고 안정성)**
```bash
docker run --gpus all --rm \
  -v "/mnt/d/MYCLAUDE_PROJECT/YOLO-영상내특정객체-모자이크-블러처리-자동화":/workspace \
  -w /workspace \
  video-anonymizer-gpu:slim \
  python -m anonymizer.cli \
  --input data/your_video.mp4 \
  --output output/result.mp4 \
  --parts eyes,elbows \
  --style mosaic
```

### ⚠️ 주의사항
- **Half Precision 이슈**: `--gpu-optimized` 플래그 사용 시 Half precision 에러 발생
- **Ultra 파이프라인**: 컨테이너 hang 이슈로 Speed 파이프라인 권장
- **GPU 메모리**: 8GB GPU에서 배치크기 64까지 안정적 처리

## Build & Development Commands

### 🚀 자동 최적화 명령어 (권장)
```bash
# 컨테이너 환경 설정 (최초 1회)
make container-setup

# 🤖 완전 자동 최적화 (하드웨어 감지 + 파이프라인 자동 선택)
make run-auto IN=data/20140413.mp4

# 🤖 자동 최적화 + 고품질 파이프라인
make run-auto-ultra IN=data/20140413.mp4

# 🤖 자동 최적화 + 최고속도 파이프라인 (권장)
make run-auto-speed IN=data/20140413.mp4

# 하드웨어 정보 확인
make hardware-info

# 컨테이너 상태 확인
make container-status

# 컨테이너 정리 (필요시)
make container-clean
```

### GPU Docker Commands (권장)
```bash
# GPU 이미지 빌드
make build
# 또는 직접: docker build -f Dockerfile.gpu -t video-anonymizer-gpu:slim .

# 🤖 AI 자동 최적화 (권장)
make container-setup                              # 최초 1회 설정
make run-auto-speed IN=data/video.mp4            # 최고속도 (93+ FPS)
make run-auto-ultra IN=data/video.mp4            # 최고품질 (69+ FPS)

# 하드웨어 정보 확인
make hardware-info

# 기본 CLI 실행 (검증된 안정적 방법)
docker run --gpus all --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  video-anonymizer-gpu:slim \
  python -m anonymizer.cli \
  --input data/video.mp4 \
  --output output/result.mp4 \
  --parts eyes,elbows \
  --style mosaic
```

### CPU Docker Commands (GPU 없는 환경)
```bash
# CPU 이미지 빌드
docker build -f Dockerfile.cpu -t video-anonymizer-cpu:latest .

# CPU로 실행 (~5 FPS)
docker run --rm \
  -v "$(pwd)":/app \
  video-anonymizer-cpu:latest \
  python -m anonymizer.cli \
  --input data/video.mp4 \
  --output output/result_cpu.mp4 \
  --parts eyes,elbows \
  --style mosaic
```

### Local Development (빠른 개발용)
```bash
# 가벼운 CPU 환경 설정
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows
pip install -r requirements-cpu.txt

# 로컬에서 빠른 테스트
python -m anonymizer.cli --input data/video.mp4 --output output/local.mp4

# 단위 테스트
pytest tests/test_smoke.py -v
```

## 개발자 정보

상세한 문서:
- [개발자 가이드](docs/DEVELOPMENT.md) - 아키텍처, 설정, 테스트 전략
- [프로젝트 구조](docs/PROJECT_STRUCTURE.md) - 전체 파일 구조 및 모듈 설명

### 주요 아키텍처
- **AnonymizePipeline**: TTL 기반 ROI 지속성을 갖춘 메인 오케스트레이터
- **Detection System**: YOLO 포즈 + Haar 캐스케이드 병렬 검출
- **Auto-Optimization System**: 하드웨어 자동 감지 및 최적 설정

## 🤖 자동 최적화 시스템 (NEW)

### 하드웨어 자동 감지 및 최적 설정
```bash
# 하드웨어 정보 확인
make hardware-info
```

**감지되는 하드웨어 정보:**
- CPU 코어 수 및 스레드 수
- GPU 모델 및 메모리 크기 
- 시스템 RAM 용량
- GPU Compute Capability

**자동 최적화되는 설정값:**
- `batch_size`: GPU 메모리 기반 최적 배치 크기 (1-64)
- `cpu_workers`: CPU 코어 기반 최적 워커 수
- `confidence`: GPU 성능 기반 검출 임계값
- `pose_model`: GPU 메모리에 따른 최적 모델 선택
- `half_precision`: GPU 지원 여부에 따른 FP16 사용
- `eye_detection_interval`: GPU 성능에 따른 검출 간격

### 성능 비교

| 모드 | 배치크기 | CPU 워커 | GPU 활용 | 처리속도 |
|------|----------|----------|----------|----------|
| 기본 | 8 | 16 | 30% | ~30 FPS |
| 자동최적화 | 64 | 32 | 95% | ~95 FPS |

### Docker 컨테이너 최적화

**문제점:**
- `--rm` 플래그로 매번 새 컨테이너 생성
- YOLO 모델 재다운로드 (300MB+)
- 초기화 시간 30초+

**해결책: Named Container + Volume**
```bash
# 최초 설정 (1회)
make container-setup

# 이후 바로 사용
make run-auto-speed IN=data/video.mp4
```

**개선 효과:**
- 시작 시간: 30초 → 3초
- 모델 재다운로드 제거
- 설정 캐시 유지
- 안정성 향상

### 설정값 관리

**자동 캐시 시스템:**
- 파일: `auto_config_cache.json`
- 하드웨어 변경시 자동 재감지
- 사용자 설정 오버라이드 가능

```json
{
  "hardware_info": {
    "gpu_name": "NVIDIA GeForce RTX 3060 Ti",
    "gpu_memory_gb": 8.0,
    "cpu_cores": 16
  },
  "optimal_settings": {
    "batch_size": 64,
    "cpu_workers": 8,
    "confidence": 0.4
  }
}
```

**설정 우선순위:**
1. 명령행 플래그 (최우선)
2. 캐시된 자동 설정
3. 기본값

### 파이프라인 선택 가이드

| GPU 메모리 | 권장 파이프라인 | 특징 |
|------------|----------------|------|
| 8GB+ | `run-auto-ultra` | 최고 품질, 배치크기 64 |
| 4GB+ | `run-auto-speed` | 최고 속도, 배치크기 32 |
| 4GB 미만 | `run-auto` | 안정성 우선, 자동 조정 |
| GPU 없음 | `run-auto` | CPU 최적화 |