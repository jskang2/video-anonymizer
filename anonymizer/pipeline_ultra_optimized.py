from __future__ import annotations
import cv2
import numpy as np
from typing import List, Dict
import threading
import queue
import time
import torch
from .detectors import PoseDetector, FaceEyeDetector
from .roi import elbows_from_keypoints, eyes_from_boxes
from .video_io import VideoReader, VideoWriter
from .gpu_accelerated_ops import GPUAcceleratedOps

class UltraOptimizedPipeline:
    """모든 최적화가 적용된 파이프라인"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        
        # GPU 최적화 설정
        gpu_settings = {
            'device': getattr(cfg, 'device', 0),
            'confidence': getattr(cfg, 'confidence', 0.25),
            'batch_size': getattr(cfg, 'batch_size', 8),  # 큰 배치 크기
            'half_precision': False  # 안정성을 위해 비활성화
        }
        
        self.pose = PoseDetector(cfg.pose_model, **gpu_settings)
        self.faceeye = FaceEyeDetector(cfg.face_cascade, cfg.eye_cascade)
        self.batch_size = gpu_settings['batch_size']
        
        # GPU 가속 연산
        self.gpu_ops = GPUAcceleratedOps(device=f"cuda:{gpu_settings['device']}")
        
        # 큐 설정 (메모리 효율성을 위해 크기 제한)
        self.frame_queue = queue.Queue(maxsize=self.batch_size * 3)
        self.result_queue = queue.Queue(maxsize=self.batch_size * 3)
        
        # 성능 통계
        self.stats = {
            'frames_read': 0,
            'frames_processed': 0,
            'frames_written': 0,
            'gpu_utilization': []
        }
        
        print(f"[UltraOptimized] 초기화 완료:")
        print(f"  배치 크기: {self.batch_size}")
        print(f"  GPU 가속: {self.gpu_ops.device}")
        print(f"  멀티스레딩: 활성화")

    def _read_frames_async(self, input_path: str):
        """비동기 프레임 읽기"""
        rd = VideoReader(input_path)
        
        try:
            for frame_idx, frame in enumerate(rd):
                # GPU 메모리 최적화를 위한 전처리
                if frame.dtype != np.uint8:
                    frame = frame.astype(np.uint8)
                
                self.frame_queue.put((frame_idx, frame), timeout=10)
                self.stats['frames_read'] += 1
                
                if frame_idx % 500 == 0:
                    print(f"[Reader] {frame_idx} 프레임 읽음, "
                          f"큐 크기: {self.frame_queue.qsize()}")
        
        except Exception as e:
            print(f"[Reader] 오류: {e}")
        finally:
            self.frame_queue.put(None)  # 종료 신호

    def _process_frames_batch(self):
        """GPU 배치 처리"""
        frame_buffer = []
        idx_buffer = []
        
        while True:
            try:
                item = self.frame_queue.get(timeout=2)
                if item is None:
                    break
                
                frame_idx, frame = item
                frame_buffer.append(frame)
                idx_buffer.append(frame_idx)
                
                # 배치 처리
                if len(frame_buffer) >= self.batch_size:
                    self._process_batch_gpu(frame_buffer, idx_buffer)
                    frame_buffer = []
                    idx_buffer = []
                    
                    # GPU 메모리 정리 (주기적)
                    if self.stats['frames_processed'] % 100 == 0:
                        torch.cuda.empty_cache()
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Processor] 오류: {e}")
        
        # 남은 프레임 처리
        if frame_buffer:
            self._process_batch_gpu(frame_buffer, idx_buffer)
        
        self.result_queue.put(None)  # 종료 신호

    def _process_batch_gpu(self, frames: List[np.ndarray], indices: List[int]):
        """GPU 최적화 배치 처리"""
        start_time = time.time()
        
        # 1. Eyes detection (CPU - 빠른 Haar cascade)
        eyes_batch = []
        if "eyes" in self.cfg.parts:
            for frame in frames:
                # GPU 가속 색상 변환 (선택적)
                gray = self.gpu_ops.bgr_to_gray_gpu(frame)
                faces, eyes = self.faceeye.detect(gray)
                eyes_rois = eyes_from_boxes(eyes, self.cfg.safety_margin_px)
                eyes_batch.append(eyes_rois)
        else:
            eyes_batch = [[] for _ in frames]
        
        # 2. Elbows detection (GPU 배치 - 핵심 최적화)
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
        
        # 3. 후처리 (GPU 가속)
        for i, (frame, idx) in enumerate(zip(frames, indices)):
            rois = eyes_batch[i] + elbows_batch[i]
            
            # GPU 가속 마스크 생성
            mask = self.gpu_ops.draw_mask_gpu(frame.shape[:2], rois)
            
            # GPU 가속 익명화
            out = self.gpu_ops.apply_anonymize_gpu(frame, mask, style=self.cfg.style)
            
            self.result_queue.put((idx, out, len(rois)))
            self.stats['frames_processed'] += 1
        
        # 성능 통계 업데이트
        batch_time = time.time() - start_time
        fps = len(frames) / batch_time
        gpu_util = min(100, fps * 2)  # 추정치
        self.stats['gpu_utilization'].append(gpu_util)
        
        if self.stats['frames_processed'] % self.cfg.log_every == 0:
            avg_gpu = np.mean(self.stats['gpu_utilization'][-10:])
            print(f"[GPU Batch] {self.stats['frames_processed']} 프레임 처리, "
                  f"FPS: {fps:.1f}, 예상 GPU: {avg_gpu:.0f}%")

    def _write_frames_async(self, output_path: str, total_frames: int):
        """비동기 프레임 쓰기"""
        wr = None
        result_buffer = {}
        next_expected_idx = 0
        
        while True:
            try:
                item = self.result_queue.get(timeout=2)
                if item is None:
                    break
                
                frame_idx, out_frame, roi_count = item
                
                # VideoWriter 초기화
                if wr is None:
                    h, w = out_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    # 높은 품질 설정
                    wr = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))
                
                # 순서 보장
                result_buffer[frame_idx] = (out_frame, roi_count)
                
                # 순서대로 쓰기
                while next_expected_idx in result_buffer:
                    frame_data, rois = result_buffer.pop(next_expected_idx)
                    wr.write(frame_data)
                    self.stats['frames_written'] += 1
                    next_expected_idx += 1
                    
            except queue.Empty:
                continue
        
        # 남은 프레임 쓰기
        for idx in sorted(result_buffer.keys()):
            frame_data, _ = result_buffer[idx]
            wr.write(frame_data)
            self.stats['frames_written'] += 1
        
        if wr:
            wr.release()

    def run(self, input_path: str, output_path: str):
        """최적화된 실행"""
        print(f"[UltraOptimized] 처리 시작")
        
        # 총 프레임 수
        rd = VideoReader(input_path)
        total_frames = rd.n
        rd.cap.release()
        
        start_time = time.time()
        
        # 3개 스레드 실행
        threads = [
            threading.Thread(target=self._read_frames_async, args=(input_path,)),
            threading.Thread(target=self._process_frames_batch),
            threading.Thread(target=self._write_frames_async, args=(output_path, total_frames))
        ]
        
        for t in threads:
            t.start()
        
        # 진행상황 모니터링
        last_processed = 0
        while any(t.is_alive() for t in threads):
            time.sleep(5)
            current = self.stats['frames_processed']
            if current > last_processed:
                progress = (current / total_frames) * 100
                elapsed = time.time() - start_time
                eta = (elapsed / current * total_frames - elapsed) if current > 0 else 0
                avg_gpu = np.mean(self.stats['gpu_utilization'][-20:]) if self.stats['gpu_utilization'] else 0
                
                print(f"[Progress] {progress:.1f}% ({current}/{total_frames}), "
                      f"ETA: {eta/60:.1f}분, GPU: {avg_gpu:.0f}%")
                last_processed = current
        
        # 스레드 완료 대기
        for t in threads:
            t.join()
        
        total_time = time.time() - start_time
        avg_gpu = np.mean(self.stats['gpu_utilization']) if self.stats['gpu_utilization'] else 0
        
        print(f"[완료] Ultra Optimized 처리 완료:")
        print(f"  총 시간: {total_time:.1f}초 ({total_time/60:.1f}분)")
        print(f"  처리 속도: {total_frames/total_time:.1f} FPS")
        print(f"  평균 GPU 사용률: {avg_gpu:.0f}%")
        print(f"  읽기/처리/쓰기: {self.stats['frames_read']}/{self.stats['frames_processed']}/{self.stats['frames_written']}")