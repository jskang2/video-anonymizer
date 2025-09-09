from __future__ import annotations
import cv2
import numpy as np
from typing import List, Dict
from .detectors import PoseDetector, FaceEyeDetector
from .roi import elbows_from_keypoints, eyes_from_boxes, draw_soft_mask, apply_anonymize
from .video_io import VideoReader, VideoWriter
import torch

class BatchOptimizedPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        
        # GPU 최적화 설정
        gpu_settings = {
            'device': getattr(cfg, 'device', 0),
            'confidence': getattr(cfg, 'confidence', 0.25),
            'batch_size': getattr(cfg, 'batch_size', 8),  # 배치 크기 증가
            'half_precision': getattr(cfg, 'half_precision', False)
        }
        
        self.pose = PoseDetector(cfg.pose_model, **gpu_settings)
        self.faceeye = FaceEyeDetector(cfg.face_cascade, cfg.eye_cascade)
        self.batch_size = gpu_settings['batch_size']
        
        # GPU 메모리 사전 할당
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print(f"[Batch] GPU 배치 크기: {self.batch_size}")

    def _process_batch_rois(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """배치 단위로 ROI 처리"""
        batch_rois = []
        
        # Eyes detection (CPU)
        eyes_batch = []
        if "eyes" in self.cfg.parts:
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces, eyes = self.faceeye.detect(gray)
                eyes_rois = eyes_from_boxes(eyes, self.cfg.safety_margin_px)
                eyes_batch.append(eyes_rois)
        else:
            eyes_batch = [[] for _ in frames]
        
        # Elbows detection (GPU 배치 처리)
        elbows_batch = []
        if "elbows" in self.cfg.parts:
            # GPU 배치 추론 (핵심 최적화)
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
        prev_rois_buffer = []
        miss_counts = []
        frame_idx = 0
        
        print(f"[BatchOptimized] 처리 시작: 배치 크기 {self.batch_size}")

        for frame in rd:
            frame_buffer.append(frame)
            
            # 배치가 찼거나 마지막 프레임인 경우
            if len(frame_buffer) == self.batch_size or frame_idx == rd.n - 1:
                # GPU 배치 처리
                batch_rois = self._process_batch_rois(frame_buffer)
                
                # 배치 결과 처리
                for i, (frame, rois) in enumerate(zip(frame_buffer, batch_rois)):
                    # TTL 로직 (기존과 동일)
                    if len(prev_rois_buffer) <= i:
                        prev_rois_buffer.append([])
                        miss_counts.append(0)
                    
                    if not rois:
                        if prev_rois_buffer[i] and miss_counts[i] < ttl:
                            rois = prev_rois_buffer[i]
                            miss_counts[i] += 1
                        else:
                            prev_rois_buffer[i] = []
                            miss_counts[i] = 0
                    else:
                        prev_rois_buffer[i] = rois
                        miss_counts[i] = 0

                    # 후처리 및 출력
                    mask = draw_soft_mask(frame.shape[:2], rois, feather=3)
                    out = apply_anonymize(frame, mask, style=self.cfg.style)
                    wr.write(out)

                    if frame_idx % max(1, self.cfg.log_every) == 0:
                        print(f"[Batch] frame {frame_idx}/{rd.n or '?'} rois={len(rois)} "
                              f"batch_size={len(frame_buffer)} gpu_util=↑")
                    
                    frame_idx += 1
                
                # 버퍼 초기화
                frame_buffer = []
                # GPU 메모리 정리 (주기적)
                if frame_idx % (self.batch_size * 10) == 0:
                    torch.cuda.empty_cache()

        wr.release()
        print(f"[Done] 배치 처리 완료: {output_path}")