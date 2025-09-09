from __future__ import annotations
import cv2
import numpy as np
from typing import List, Dict
import threading
import queue
import time
import torch
import os
from concurrent.futures import ThreadPoolExecutor, Future
from .detectors import PoseDetector, FaceEyeDetector
from .roi import elbows_from_keypoints, eyes_from_boxes
from .video_io import VideoReader, VideoWriter
from .gpu_accelerated_ops import GPUAcceleratedOps

class SpeedOptimizedPipeline:
    """최고 속도에 중점을 두고 재설계된 파이프라인 (v3, 교착 상태 해결)"""

    def __init__(self, cfg):
        self.cfg = cfg
        self.batch_size = getattr(cfg, 'batch_size', 32)
        self.confidence = getattr(cfg, 'confidence', 0.5)
        self.eye_detection_interval = 4
        self.cpu_workers = os.cpu_count()

        gpu_settings = {
            'device': getattr(cfg, 'device', 0),
            'confidence': self.confidence,
            'batch_size': self.batch_size,
            'half_precision': False
        }

        self.pose_detector = PoseDetector(cfg.pose_model, **gpu_settings)
        self.gpu_ops = GPUAcceleratedOps(device=f"cuda:{gpu_settings['device']}")

        self.read_queue = queue.Queue(maxsize=self.batch_size * 2)
        self.cpu_queue = queue.Queue(maxsize=self.batch_size * 2)
        self.gpu_queue = queue.Queue(maxsize=self.batch_size * 2)
        self.writer_queue = queue.Queue(maxsize=self.batch_size * 2)

        self.cpu_executor = ThreadPoolExecutor(max_workers=self.cpu_workers, thread_name_prefix='cpu_worker')
        self.stop_event = threading.Event()

        print(f"[Speed v3] 초기화 완료:")
        print(f"  CPU 병렬 워커: {self.cpu_workers}")
        print(f"  GPU 배치 크기: {self.batch_size}")

    def _read_frames(self, input_path: str):
        try:
            rd = VideoReader(input_path)
            for frame_idx, frame in enumerate(rd):
                if self.stop_event.is_set(): break
                self.read_queue.put((frame_idx, frame))
        finally:
            self.read_queue.put(None)
            print("[Reader] 완료")

    def _cpu_task_wrapper(self, gray_frame: np.ndarray) -> tuple[list, list]:
        face_eye_detector = FaceEyeDetector(self.cfg.face_cascade, self.cfg.eye_cascade)
        return face_eye_detector.detect(gray_frame)

    def _dispatch_cpu_tasks(self):
        try:
            while not self.stop_event.is_set():
                item = self.read_queue.get()
                if item is None: break
                frame_idx, frame = item
                if frame_idx % self.eye_detection_interval == 0:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    future = self.cpu_executor.submit(self._cpu_task_wrapper, gray_frame)
                else:
                    future = Future()
                    future.set_result(([], []))
                self.cpu_queue.put((frame_idx, frame, future))
        finally:
            self.cpu_queue.put(None)
            print("[CPU Dispatcher] 완료")

    def _collect_cpu_results(self):
        try:
            while not self.stop_event.is_set():
                item = self.cpu_queue.get()
                if item is None: break
                frame_idx, frame, future = item
                _, eyes = future.result()
                eye_rois = eyes_from_boxes(eyes, self.cfg.safety_margin_px)
                self.gpu_queue.put((frame_idx, frame, eye_rois))
        finally:
            self.gpu_queue.put(None)
            print("[CPU Collector] 완료")

    def _process_gpu_batch(self):
        try:
            while not self.stop_event.is_set():
                batch_data = []
                sentinel_received = False
                try:
                    item = self.gpu_queue.get(timeout=1)
                    if item is None:
                        sentinel_received = True
                    else:
                        batch_data.append(item)
                        while len(batch_data) < self.batch_size:
                            item = self.gpu_queue.get_nowait()
                            if item is None:
                                self.gpu_queue.put(None) # Put it back for the next loop
                                sentinel_received = True
                                break
                            batch_data.append(item)
                except queue.Empty:
                    pass

                if batch_data:
                    frames = [item[1] for item in batch_data]
                    batch_keypoints = self.pose_detector.infer_batch(frames)
                    for i, (frame_idx, frame, eye_rois) in enumerate(batch_data):
                        kpts = batch_keypoints[i][0] if i < len(batch_keypoints) and batch_keypoints[i] else []
                        elbow_rois = elbows_from_keypoints(kpts, self.cfg.safety_margin_px)
                        all_rois = eye_rois + elbow_rois
                        mask = self.gpu_ops.draw_mask_gpu(frame.shape[:2], all_rois)
                        out_frame = self.gpu_ops.apply_anonymize_gpu(frame, mask, style=self.cfg.style)
                        self.writer_queue.put((frame_idx, out_frame))
                
                if sentinel_received:
                    break
        finally:
            self.writer_queue.put(None)
            print("[GPU Processor] 완료")

    def _write_frames(self, output_path: str, total_frames: int, w: int, h: int, fps: float):
        wr = VideoWriter(output_path, w, h, fps)
        buffer = {}
        next_idx = 0
        processed_count = 0
        try:
            while True:
                item = self.writer_queue.get()
                if item is None: break
                frame_idx, frame = item
                buffer[frame_idx] = frame
                while next_idx in buffer:
                    out_frame = buffer.pop(next_idx)
                    wr.write(out_frame)
                    processed_count += 1
                    if processed_count % 100 == 0:
                        print(f"[Writer] 진행률: {processed_count}/{total_frames}")
                    next_idx += 1
        finally:
            for idx in sorted(buffer.keys()):
                wr.write(buffer[idx])
                processed_count += 1
            print(f"[Writer] 최종 프레임 쓰기 완료. 총 {processed_count} 프레임.")
            if wr: wr.release()
            print("[Writer] 완료")

    def run(self, input_path: str, output_path: str):
        rd = VideoReader(input_path)
        total_frames, w, h, fps = rd.n, rd.w, rd.h, rd.fps
        rd.cap.release()
        if total_frames <= 0: return

        start_time = time.time()
        threads = [
            threading.Thread(target=self._read_frames, args=(input_path,), name="Reader"),
            threading.Thread(target=self._dispatch_cpu_tasks, name="CPU-Dispatcher"),
            threading.Thread(target=self._collect_cpu_results, name="CPU-Collector"),
            threading.Thread(target=self._process_gpu_batch, name="GPU-Processor"),
            threading.Thread(target=self._write_frames, args=(output_path, total_frames, w, h, fps), name="Writer")
        ]

        for t in threads: t.start()
        threads[-1].join()
        self.stop_event.set()
        for t in threads[:-1]:
            t.join()

        self.cpu_executor.shutdown(wait=True)
        total_time = time.time() - start_time
        final_fps = total_frames / total_time if total_time > 0 else 0
        print(f"\n[완료] 처리 시간: {total_time:.2f}초, 처리 속도: {final_fps:.2f} FPS")
