# 개발자 가이드

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

**Auto-Optimization System** (`auto_optimizer.py`):
- `HardwareProfiler`: Automatic hardware detection and analysis
- `AutoTuner`: Optimal settings calculation based on hardware capabilities
- `RuntimeOptimizer`: Dynamic performance monitoring and adjustment
- `AutoConfig`: Configuration generation and caching system

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

**Performance**: Auto-optimized based on hardware capabilities. GPU version uses Dockerfile.gpu.