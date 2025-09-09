# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Video Anonymizer MVP - A Docker-based video processing pipeline for automatic anonymization of specific body parts (eyes, elbows) using YOLO pose estimation and Haar cascade detection with OpenCV.

## Git 워크플로우

### github 푸쉬를 위해 다음 정보 사용:
Github 주소 : https://github.com/jskang2/video-anonymizer

### Git 설정 및 푸시 규칙
.git 이 존재하지 않으면 Git 저장소 초기화 할것 (git init)
main 브랜치 사용할 것
파일 생성 또는 수정시, 파일 생성 또는 수정한 후, git add와 commit 수행할 것
파일 삭제시 git rm 및 commit 사용할 것

원격 저장소에 푸시할 때, 먼저 HTTP 버퍼 크기를 늘리고 조금 씩 나누어 푸시할 것. 에러 시 작은 변경사항만 포함하는 새커밋을 만들어 푸시할 것


## Build & Development Commands

### Docker Commands
```bash
# Build the container
make build
docker build -t video-anonymizer-mvp:latest .

# Download sample video for testing
make demo

# Run video processing
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