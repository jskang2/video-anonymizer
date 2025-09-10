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

### Docker Commands (기본)
```bash
# Build the container
make build
docker build -t video-anonymizer-mvp:latest .

# Download sample video for testing
make demo

# Run video processing (기본 파이프라인)
make run IN=data/in.mp4 OUT=data/out.mp4 PARTS=eyes,elbows STYLE=mosaic

# Run tests
make test
docker run --rm -v $(PWD):/app video-anonymizer-mvp:latest pytest -q

# Direct CLI usage
docker run --rm -v $(PWD):/app video-anonymizer-mvp:latest \
  python -m anonymizer.cli --input data/in.mp4 --output data/out.mp4 \
  --parts eyes,elbows --style mosaic --safety 12 --ttl 5
```

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run directly without Docker
python -m anonymizer.cli --config configs/default.yaml

# Run single test
pytest tests/test_smoke.py -v
pytest tests/test_smoke.py::test_smoke -v
```

## Core Architecture

### Processing Pipeline Flow
1. **VideoReader** → frame-by-frame processing
2. **Detection Phase** → parallel detection of eyes (Haar cascades) and elbows (YOLO pose keypoints)
3. **ROI Generation** → convert detections to regions of interest with safety margins
4. **TTL Logic** → maintain previous ROIs for specified frames when detection fails (reduces flicker)
5. **Mask Creation** → soft masks with feathering for smooth blending
6. **Anonymization** → apply style (mosaic/gaussian/boxblur/pixelate) to masked regions
7. **VideoWriter** → output processed frames

### Key Components

**AnonymizePipeline** (`pipeline.py`) - Main orchestrator that coordinates the entire processing flow with TTL-based ROI persistence.

**Detection System** (`detectors.py`):
- `PoseDetector`: YOLOv8-pose wrapper for 17-point COCO keypoints, extracts elbow positions (indices 7,8)
- `FaceEyeDetector`: Haar cascade wrapper for face/eye detection with nested ROI processing

**ROI System** (`roi.py`):
- Converts keypoints to circular ROIs (elbows) and bounding boxes to elliptical ROIs (eyes)
- Implements adaptive sizing based on elbow-to-wrist distance
- Creates soft masks with Gaussian feathering for natural blending

**Configuration** (`config.py`): Dataclass-based config with YAML loading and CLI override support.

### Critical Architecture Details

**TTL Anti-Flicker Logic**: When detections fail, the pipeline reuses previous ROIs for `ttl_frames` to prevent flickering. This is essential for consistent anonymization across frames.

**ROI Coordinate System**: All ROIs use image pixel coordinates (float) with adaptive sizing:
- Elbows: Circle radius = 0.5 × elbow-to-wrist distance (min 12px) + safety margin
- Eyes: Ellipse dimensions = bounding box dimensions + safety margin

**Processing Styles**:
- `mosaic/pixelate`: Downsample to 1/16 resolution then upscale with nearest interpolation
- `gaussian`: 25×25 kernel Gaussian blur
- `boxblur`: 25×25 box filter

**Memory Efficiency**: Frame-by-frame processing with immediate output writing, no batch loading.

## Configuration

Default configuration in `configs/default.yaml` with CLI override support. Key parameters:
- `safety_margin_px`: Expands ROI boundaries (default: 12px)
- `ttl_frames`: Frames to maintain ROI when detection fails (default: 5)
- `pose_model`: YOLOv8 variant (default: yolov8n-pose.pt)
- `style`: Anonymization method (mosaic|gaussian|boxblur|pixelate)

## Testing Strategy

**Smoke Test** (`tests/test_smoke.py`): Creates synthetic 10-frame black video, processes through full pipeline, verifies output file creation and non-zero size.

For development, use synthetic videos for consistent testing since real videos may have detection variance.

## Development Notes

**Model Dependencies**: YOLO models download automatically on first use. Eye/face cascades are included with OpenCV.

**Video I/O**: Uses OpenCV VideoCapture/VideoWriter with MP4V codec. FFmpeg included in Docker for broader format support.

**Error Handling**: Missing keypoints return `None`, empty detections trigger TTL fallback logic.

**Performance**: CPU-optimized with yolov8n-pose.pt (nano model). GPU version available via Dockerfile.gpu.

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