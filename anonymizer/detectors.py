from __future__ import annotations
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict

# COCO keypoint index mapping for YOLOv8-pose (17 pts)
COCO_KPTS = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

class PoseDetector:
    def __init__(self, model_name: str = "yolov8n-pose.pt", **kwargs):
        import torch
        self.model = YOLO(model_name)
        
        # GPU 최적화 설정 (config에서 전달받음)
        device_id = kwargs.get('device', 0 if torch.cuda.is_available() else 'cpu')
        
        # 디바이스 문자열 정규화
        if device_id == 'cpu':
            self.device = 'cpu'
        elif isinstance(device_id, str) and device_id.isdigit():
            self.device = f"cuda:{device_id}"  # '0' -> 'cuda:0'
        elif isinstance(device_id, int):
            self.device = f"cuda:{device_id}"  # 0 -> 'cuda:0'
        elif isinstance(device_id, str) and device_id.startswith('cuda'):
            self.device = device_id  # 'cuda:0' 그대로 유지
        else:
            self.device = 'cuda:0'  # 기본값
        self.batch_size = kwargs.get('batch_size', 1)
        self.confidence = kwargs.get('confidence', 0.25)
        self.iou_threshold = kwargs.get('iou_threshold', 0.7)
        self.max_det = kwargs.get('max_det', 300)
        self.imgsz = kwargs.get('imgsz', 640)
        self.half_precision = kwargs.get('half_precision', False)
        
        # GPU 디바이스 설정
        if torch.cuda.is_available() and self.device != 'cpu':
            try:
                self.model.to(self.device)
                if self.half_precision:
                    self.model.model.half()
                    print(f"[GPU] Half precision enabled on device: {self.device}")
                print(f"[GPU] YOLO model loaded on device: {self.device}")
                print(f"[GPU] Settings: conf={self.confidence}, half={self.half_precision}")
            except Exception as e:
                print(f"[GPU] Failed to load on GPU: {e}, falling back to CPU")
                self.device = 'cpu'
        else:
            self.device = 'cpu'
            print("[CPU] Using CPU inference")

    def infer(self, frame: np.ndarray) -> List[np.ndarray]:
        # returns list of (17, 2) keypoints per person (float xy in image coords)
        results = self.model.predict(
            source=frame, 
            verbose=False,
            device=self.device,
            conf=self.confidence,
            iou=self.iou_threshold,
            max_det=self.max_det,
            imgsz=self.imgsz,
            half=self.half_precision
        )[0]
        
        if results.keypoints is None:
            return []
        xy = results.keypoints.xy  # (n,17,2)
        return [kp.cpu().numpy() for kp in xy]
    
    def infer_batch(self, frames: List[np.ndarray]) -> List[List[np.ndarray]]:
        """배치 처리로 GPU 활용도 증가"""
        if not frames:
            return []
            
        results = self.model.predict(
            source=frames,
            verbose=False,
            device=self.device,
            conf=self.confidence,
            iou=self.iou_threshold,
            max_det=self.max_det,
            imgsz=self.imgsz,
            half=self.half_precision
        )
        
        batch_keypoints = []
        for result in results:
            if result.keypoints is None:
                batch_keypoints.append([])
            else:
                xy = result.keypoints.xy  # (n,17,2)
                batch_keypoints.append([kp.cpu().numpy() for kp in xy])
        
        return batch_keypoints

class FaceEyeDetector:
    def __init__(self, face_cascade_name: str, eye_cascade_name: str):
        self.face = cv2.CascadeClassifier(cv2.data.haarcascades + face_cascade_name)
        self.eye = cv2.CascadeClassifier(cv2.data.haarcascades + eye_cascade_name)
        if self.face.empty() or self.eye.empty():
            raise RuntimeError("Failed to load Haar cascades. Check filenames.")

    def detect(self, gray: np.ndarray):
        # returns faces, eyes as lists of (x,y,w,h)
        faces = self.face.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60,60))
        eyes = []
        for (x,y,w,h) in faces:
            roi = gray[y:y+h, x:x+w]
            es = self.eye.detectMultiScale(roi, scaleFactor=1.1, minNeighbors=3, minSize=(20,20))
            for (ex,ey,ew,eh) in es:
                eyes.append((x+ex, y+ey, ew, eh))
        return list(faces), eyes
