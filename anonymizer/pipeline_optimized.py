from __future__ import annotations
import cv2
import numpy as np
from typing import List, Dict
from .detectors import PoseDetector, FaceEyeDetector
from .roi import elbows_from_keypoints, eyes_from_boxes, draw_soft_mask, apply_anonymize
from .video_io import VideoReader, VideoWriter

class OptimizedAnonymizePipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        
        # GPU 최적화 설정 추출
        gpu_settings = {
            'device': getattr(cfg, 'device', '0'),
            'batch_size': getattr(cfg, 'batch_size', 1),
            'confidence': getattr(cfg, 'confidence', 0.25),
            'iou_threshold': getattr(cfg, 'iou_threshold', 0.7),
            'max_det': getattr(cfg, 'max_det', 300),
            'imgsz': getattr(cfg, 'imgsz', 640),
            'half_precision': getattr(cfg, 'half_precision', False)
        }
        
        self.pose = PoseDetector(cfg.pose_model, **gpu_settings)
        self.faceeye = FaceEyeDetector(cfg.face_cascade, cfg.eye_cascade)
        self.prev_rois: List[List[Dict]] = []  # 배치용 이전 ROI
        self.miss_counts: List[int] = []       # 배치용 miss count
        self.batch_size = gpu_settings['batch_size']
        
        print(f"[Optimized] Batch size: {self.batch_size}, GPU: {gpu_settings['device']}")

    def _build_rois_single(self, frame: np.ndarray) -> List[Dict]:
        """단일 프레임 ROI 구성 (기존 방식)"""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rois: List[Dict] = []

        # eyes via cascades
        if "eyes" in self.cfg.parts:
            faces, eyes = self.faceeye.detect(gray)
            rois.extend(eyes_from_boxes(eyes, self.cfg.safety_margin_px))

        # elbows via pose
        if "elbows" in self.cfg.parts:
            kpts_list = self.pose.infer(frame)
            for kpts in kpts_list:
                rois.extend(elbows_from_keypoints(kpts, self.cfg.safety_margin_px))

        return rois
    
    def _build_rois_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """배치 프레임 ROI 구성 (GPU 최적화)"""
        batch_rois = []
        
        # 배치 eyes detection via cascades
        eyes_batch = []
        if "eyes" in self.cfg.parts:
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces, eyes = self.faceeye.detect(gray)
                eyes_rois = eyes_from_boxes(eyes, self.cfg.safety_margin_px)
                eyes_batch.append(eyes_rois)
        else:
            eyes_batch = [[] for _ in frames]
        
        # 배치 elbows detection via pose (GPU 최적화 핵심)
        elbows_batch = []
        if "elbows" in self.cfg.parts:
            batch_keypoints = self.pose.infer_batch(frames)
            for kpts_list in batch_keypoints:
                elbows_rois = []
                for kpts in kpts_list:
                    elbows_rois.extend(elbows_from_keypoints(kpts, self.cfg.safety_margin_px))
                elbows_batch.append(elbows_rois)
        else:
            elbows_batch = [[] for _ in frames]
        
        # ROI 결합
        for eyes_rois, elbows_rois in zip(eyes_batch, elbows_batch):
            batch_rois.append(eyes_rois + elbows_rois)
        
        return batch_rois

    def run(self, input_path: str, output_path: str):
        rd = VideoReader(input_path)
        wr = VideoWriter(output_path, rd.w, rd.h, rd.fps)

        ttl = int(self.cfg.ttl_frames)
        frame_buffer = []
        frame_idx = 0
        
        # 배치 크기 초기화
        if len(self.prev_rois) < self.batch_size:
            self.prev_rois.extend([[] for _ in range(self.batch_size - len(self.prev_rois))])
            self.miss_counts.extend([0 for _ in range(self.batch_size - len(self.miss_counts))])

        for frame in rd:
            frame_buffer.append(frame)
            
            # 배치가 차거나 마지막 프레임인 경우 처리
            if len(frame_buffer) == self.batch_size or frame_idx == rd.n - 1:
                if self.batch_size > 1 and len(frame_buffer) > 1:
                    # 배치 처리 (GPU 최적화)
                    batch_rois = self._build_rois_batch(frame_buffer)
                else:
                    # 단일 프레임 처리
                    batch_rois = [self._build_rois_single(f) for f in frame_buffer]
                
                # TTL 로직 적용 및 출력 프레임 생성
                for i, (frame, rois) in enumerate(zip(frame_buffer, batch_rois)):
                    batch_idx = i % self.batch_size
                    
                    if not rois:
                        # use previous rois for a few frames to avoid flicker
                        if self.prev_rois[batch_idx] and self.miss_counts[batch_idx] < ttl:
                            rois = self.prev_rois[batch_idx]
                            self.miss_counts[batch_idx] += 1
                        else:
                            self.prev_rois[batch_idx] = []
                            self.miss_counts[batch_idx] = 0
                    else:
                        self.prev_rois[batch_idx] = rois
                        self.miss_counts[batch_idx] = 0

                    mask = draw_soft_mask(frame.shape[:2], rois, feather=3)
                    out = apply_anonymize(frame, mask, style=self.cfg.style)
                    wr.write(out)

                    if frame_idx % max(1, self.cfg.log_every) == 0:
                        print(f"[Anonymize] frame {frame_idx}/{rd.n or '?'} rois={len(rois)} batch={len(frame_buffer)}")
                    
                    frame_idx += 1
                
                # 버퍼 초기화
                frame_buffer = []

        wr.release()
        print(f"[Done] saved: {output_path}")