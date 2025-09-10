from dataclasses import dataclass, field
from typing import List, Optional
import yaml

@dataclass
class Config:
    input: str
    output: str
    parts: List[str] = field(default_factory=lambda: ["eyes", "elbows"])
    style: str = "mosaic"  # mosaic|gaussian|boxblur|pixelate
    safety_margin_px: int = 12
    ttl_frames: int = 5
    pose_model: str = "yolov8n-pose.pt"
    eye_cascade: str = "haarcascade_eye.xml"
    face_cascade: str = "haarcascade_frontalface_default.xml"
    log_every: int = 30
    
    # GPU 최적화 파라미터
    batch_size: int = 1
    confidence: float = 0.25
    iou_threshold: float = 0.7
    max_det: int = 300
    imgsz: int = 640
    half_precision: bool = False
    device: str = "cuda:0"  # GPU 디바이스 ID

    @staticmethod
    def from_yaml(path: Optional[str], overrides: dict | None = None) -> "Config":
        base = {}
        if path:
            with open(path, "r", encoding="utf-8") as f:
                base = yaml.safe_load(f) or {}
        if overrides:
            base.update({k: v for k, v in overrides.items() if v is not None})
        return Config(**base)
