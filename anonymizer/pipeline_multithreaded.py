from __future__ import annotations
import cv2
import numpy as np
from typing import List, Dict
import threading
import queue
import time
from .detectors import PoseDetector, FaceEyeDetector
from .roi import elbows_from_keypoints, eyes_from_boxes, draw_soft_mask, apply_anonymize
from .video_io import VideoReader, VideoWriter

class MultithreadedPipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        
        # GPU 설정
        gpu_settings = {
            'device': getattr(cfg, 'device', 0),
            'confidence': getattr(cfg, 'confidence', 0.25),
            'batch_size': getattr(cfg, 'batch_size', 4)
        }
        
        self.pose = PoseDetector(cfg.pose_model, **gpu_settings)
        self.faceeye = FaceEyeDetector(cfg.face_cascade, cfg.eye_cascade)
        self.batch_size = gpu_settings['batch_size']
        
        # 큐 설정 (버퍼 크기로 메모리 사용량 제어)
        self.frame_queue = queue.Queue(maxsize=20)  # 입력 프레임 큐
        self.result_queue = queue.Queue(maxsize=20)  # 처리 결과 큐
        
        # 종료 플래그
        self.stop_reading = threading.Event()
        self.stop_processing = threading.Event()
        self.stop_writing = threading.Event()

    def _read_frames(self, input_path: str):
        """프레임 읽기 스레드 (Producer)"""
        print("[Thread] 프레임 읽기 스레드 시작")
        rd = VideoReader(input_path)
        frame_idx = 0
        
        try:
            for frame in rd:
                if self.stop_reading.is_set():
                    break
                    
                # 큐가 가득 찬 경우 대기
                self.frame_queue.put((frame_idx, frame), timeout=5)
                frame_idx += 1
                
                if frame_idx % 100 == 0:
                    print(f"[Reader] 읽은 프레임: {frame_idx}")
        
        except Exception as e:
            print(f"[Reader] 오류: {e}")
        finally:
            # 읽기 완료 신호
            self.frame_queue.put(None)
            print(f"[Reader] 프레임 읽기 완료: {frame_idx} 프레임")

    def _process_frames(self):
        """프레임 처리 스레드 (GPU 추론)"""
        print("[Thread] GPU 처리 스레드 시작")
        frame_buffer = []
        idx_buffer = []
        
        while not self.stop_processing.is_set():
            try:
                # 프레임 읽기
                item = self.frame_queue.get(timeout=1)
                if item is None:  # 읽기 완료
                    break
                    
                frame_idx, frame = item
                frame_buffer.append(frame)
                idx_buffer.append(frame_idx)
                
                # 배치가 찼거나 마지막인 경우 처리
                if len(frame_buffer) >= self.batch_size:
                    results = self._process_batch(frame_buffer, idx_buffer)
                    
                    # 결과를 출력 큐에 전송
                    for result in results:
                        self.result_queue.put(result)
                    
                    frame_buffer = []
                    idx_buffer = []
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Processor] 오류: {e}")
        
        # 남은 프레임 처리
        if frame_buffer:
            results = self._process_batch(frame_buffer, idx_buffer)
            for result in results:
                self.result_queue.put(result)
        
        # 처리 완료 신호
        self.result_queue.put(None)
        print("[Processor] GPU 처리 완료")

    def _process_batch(self, frames: List[np.ndarray], indices: List[int]):
        """배치 처리"""
        results = []
        
        # Eyes detection
        eyes_batch = []
        if "eyes" in self.cfg.parts:
            for frame in frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces, eyes = self.faceeye.detect(gray)
                eyes_rois = eyes_from_boxes(eyes, self.cfg.safety_margin_px)
                eyes_batch.append(eyes_rois)
        else:
            eyes_batch = [[] for _ in frames]
        
        # Elbows detection (GPU 배치)
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
        
        # 결과 생성
        for i, (frame, idx) in enumerate(zip(frames, indices)):
            rois = eyes_batch[i] + elbows_batch[i]
            mask = draw_soft_mask(frame.shape[:2], rois, feather=3)
            out = apply_anonymize(frame, mask, style=self.cfg.style)
            results.append((idx, out, len(rois)))
        
        return results

    def _write_frames(self, output_path: str, total_frames: int):
        """프레임 쓰기 스레드 (Consumer)"""
        print("[Thread] 프레임 쓰기 스레드 시작")
        
        # VideoWriter는 메인 스레드에서 초기화된 정보 사용
        wr = None
        processed_count = 0
        
        # 순서 보장을 위한 버퍼
        result_buffer = {}
        next_expected_idx = 0
        
        while not self.stop_writing.is_set():
            try:
                item = self.result_queue.get(timeout=1)
                if item is None:  # 처리 완료
                    break
                
                frame_idx, out_frame, roi_count = item
                
                # VideoWriter 초기화 (첫 프레임에서)
                if wr is None:
                    h, w = out_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    wr = cv2.VideoWriter(output_path, fourcc, 30.0, (w, h))
                
                # 순서대로 쓰기 위해 버퍼링
                result_buffer[frame_idx] = (out_frame, roi_count)
                
                # 순서대로 쓰기
                while next_expected_idx in result_buffer:
                    frame_data, rois = result_buffer.pop(next_expected_idx)
                    wr.write(frame_data)
                    processed_count += 1
                    
                    if processed_count % max(1, self.cfg.log_every) == 0:
                        print(f"[Writer] frame {processed_count}/{total_frames} rois={rois} "
                              f"queue_size={self.result_queue.qsize()}")
                    
                    next_expected_idx += 1
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[Writer] 오류: {e}")
        
        # 남은 프레임 쓰기
        for idx in sorted(result_buffer.keys()):
            frame_data, _ = result_buffer[idx]
            wr.write(frame_data)
            processed_count += 1
        
        if wr:
            wr.release()
        print(f"[Writer] 프레임 쓰기 완료: {processed_count} 프레임")

    def run(self, input_path: str, output_path: str):
        print(f"[Multithreaded] 멀티스레딩 처리 시작")
        
        # 총 프레임 수 계산
        rd = VideoReader(input_path)
        total_frames = rd.n
        rd.cap.release()
        
        start_time = time.time()
        
        # 3개 스레드 시작
        reader_thread = threading.Thread(target=self._read_frames, args=(input_path,))
        processor_thread = threading.Thread(target=self._process_frames)
        writer_thread = threading.Thread(target=self._write_frames, args=(output_path, total_frames))
        
        # 스레드 시작
        reader_thread.start()
        processor_thread.start()
        writer_thread.start()
        
        # 스레드 완료 대기
        reader_thread.join()
        processor_thread.join()
        writer_thread.join()
        
        total_time = time.time() - start_time
        print(f"[Done] 멀티스레딩 처리 완료: {total_time:.1f}초")
        print(f"       처리 속도: {total_frames/total_time:.1f} FPS")