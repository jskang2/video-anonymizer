from __future__ import annotations
import cv2
import numpy as np
from typing import List, Dict
from .detectors import PoseDetector, FaceEyeDetector
from .roi import elbows_from_keypoints, eyes_from_boxes, draw_soft_mask, apply_anonymize
from .video_io import VideoReader, VideoWriter

class AnonymizePipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        
        # GPU 최적화 설정 전달
        gpu_settings = {}
        if hasattr(cfg, 'device'): gpu_settings['device'] = cfg.device
        if hasattr(cfg, 'confidence'): gpu_settings['confidence'] = cfg.confidence
        if hasattr(cfg, 'iou_threshold'): gpu_settings['iou_threshold'] = cfg.iou_threshold
        if hasattr(cfg, 'max_det'): gpu_settings['max_det'] = cfg.max_det
        if hasattr(cfg, 'imgsz'): gpu_settings['imgsz'] = cfg.imgsz
        if hasattr(cfg, 'half_precision'): gpu_settings['half_precision'] = cfg.half_precision
        if hasattr(cfg, 'batch_size'): gpu_settings['batch_size'] = cfg.batch_size
        
        self.pose = PoseDetector(cfg.pose_model, **gpu_settings)
        self.faceeye = FaceEyeDetector(cfg.face_cascade, cfg.eye_cascade)
        self.prev_rois: List[Dict] = []
        self.miss_count = 0

    def _build_rois(self, frame: np.ndarray) -> List[Dict]:
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

    def run(self, input_path: str, output_path: str):
        rd = VideoReader(input_path)
        wr = VideoWriter(output_path, rd.w, rd.h, rd.fps)

        ttl = int(self.cfg.ttl_frames)
        for i, frame in enumerate(rd):
            rois = self._build_rois(frame)

            if not rois:
                # use previous rois for a few frames to avoid flicker
                if self.prev_rois and self.miss_count < ttl:
                    rois = self.prev_rois
                    self.miss_count += 1
                else:
                    self.prev_rois = []
                    self.miss_count = 0
            else:
                self.prev_rois = rois
                self.miss_count = 0

            mask = draw_soft_mask(frame.shape[:2], rois, feather=3)
            out = apply_anonymize(frame, mask, style=self.cfg.style)
            wr.write(out)

            if i % max(1, self.cfg.log_every) == 0:
                print(f"[Anonymize] frame {i}/{rd.n or '?'} rois={len(rois)}")

        wr.release()
        print(f"[Done] saved: {output_path}")
