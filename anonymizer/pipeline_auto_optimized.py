"""
자동 최적화 파이프라인 - 하드웨어에 따라 자동으로 최적 설정을 찾아 실행
Ultra와 Speed 파이프라인의 장점을 결합하고 자동 최적화 기능 추가
"""
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
from .auto_optimizer import AutoConfig, OptimalSettings

class AutoOptimizedPipeline:
    """하드웨어 자원을 자동으로 감지하고 최적화하는 파이프라인"""

    def __init__(self, cfg, user_overrides: Dict = None):
        print("[AutoPipeline] 🔍 하드웨어 분석 및 최적화 중...")
        
        # 자동 최적화 설정 생성
        self.auto_config = AutoConfig()
        self.optimal_settings = self.auto_config.generate_optimal_config(user_overrides)
        
        # 원본 설정 객체 업데이트
        self.cfg = cfg
        self._apply_optimal_settings()
        
        # 성능 모니터링 변수
        self.frame_count = 0
        self.start_time = None
        self.last_memory_check = 0
        self.performance_samples = []
        
        # GPU 최적화 설정
        gpu_settings = {
            'device': getattr(cfg, 'device', 0),
            'confidence': self.optimal_settings.confidence,
            'batch_size': self.optimal_settings.batch_size,
            'half_precision': self.optimal_settings.half_precision
        }

        # 검출기 초기화
        self.pose_detector = PoseDetector(self.optimal_settings.pose_model, **gpu_settings)
        
        # GPU 가속 연산 초기화 (가능한 경우)
        if torch.cuda.is_available():
            try:
                # Device 정규화: 이미 cuda: 포함된 경우 중복 방지
                device = gpu_settings['device']
                if not str(device).startswith('cuda:') and device != 'cpu':
                    device = f"cuda:{device}"
                self.gpu_ops = GPUAcceleratedOps(device=str(device))
                print(f"[AutoPipeline] ⚡ GPU 가속 연산 활성화")
            except Exception as e:
                print(f"[AutoPipeline] ⚠️ GPU 가속 실패, CPU 모드 사용: {e}")
                self.gpu_ops = None
        else:
            self.gpu_ops = None

        # 동적 큐 크기 설정
        queue_size = self.optimal_settings.queue_size
        self.read_queue = queue.Queue(maxsize=queue_size)
        self.cpu_queue = queue.Queue(maxsize=queue_size)
        self.gpu_queue = queue.Queue(maxsize=queue_size)
        self.writer_queue = queue.Queue(maxsize=queue_size)

        # CPU 워커 풀
        self.cpu_executor = ThreadPoolExecutor(
            max_workers=self.optimal_settings.cpu_workers, 
            thread_name_prefix='auto_cpu_worker'
        )
        self.stop_event = threading.Event()
        
        # OpenCV 스레드 최적화
        cv2.setNumThreads(self.optimal_settings.opencv_threads)

        print(f"[AutoPipeline] ✅ 초기화 완료 - {self._get_config_summary()}")

    def _apply_optimal_settings(self):
        """최적화된 설정을 원본 cfg 객체에 적용"""
        self.cfg.batch_size = self.optimal_settings.batch_size
        self.cfg.confidence = self.optimal_settings.confidence
        self.cfg.eye_detection_interval = self.optimal_settings.eye_detection_interval
        
        # 새로운 설정 속성들 추가
        if not hasattr(self.cfg, 'cpu_workers'):
            self.cfg.cpu_workers = self.optimal_settings.cpu_workers
        if not hasattr(self.cfg, 'queue_size'):
            self.cfg.queue_size = self.optimal_settings.queue_size

    def _get_config_summary(self) -> str:
        """설정 요약 문자열 생성"""
        return (f"배치:{self.optimal_settings.batch_size}, "
                f"워커:{self.optimal_settings.cpu_workers}, "
                f"모델:{self.optimal_settings.pose_model}, "
                f"눈간격:{self.optimal_settings.eye_detection_interval}")

    def _read_frames(self, input_path: str):
        """프레임 읽기 (성능 모니터링 포함)"""
        try:
            rd = VideoReader(input_path)
            self.start_time = time.time()
            
            for frame_idx, frame in enumerate(rd):
                if self.stop_event.is_set(): 
                    break
                    
                self.read_queue.put((frame_idx, frame))
                self.frame_count = frame_idx + 1
                
                # 주기적 성능 체크 및 조정
                if frame_idx % 100 == 0 and frame_idx > 0:
                    self._check_and_adapt_performance()
                    
        except Exception as e:
            print(f"[Reader] 오류: {e}")
            self.stop_event.set()
        finally:
            self.read_queue.put(None)
            print("[Reader] 완료")

    def _check_and_adapt_performance(self):
        """성능 체크 및 런타임 조정"""
        if self.start_time is None:
            return
            
        current_time = time.time()
        elapsed = current_time - self.start_time
        current_fps = self.frame_count / elapsed if elapsed > 0 else 0
        
        # GPU 메모리 사용량 체크
        gpu_allocated, gpu_cached = self.auto_config.runtime_optimizer.monitor_gpu_memory()
        
        # 런타임 설정 조정
        try:
            adapted_settings = self.auto_config.adapt_runtime_settings(current_fps, gpu_allocated)
            
            # 배치 크기가 변경된 경우
            if adapted_settings.batch_size != self.optimal_settings.batch_size:
                self.optimal_settings = adapted_settings
                self.cfg.batch_size = adapted_settings.batch_size
                print(f"[AutoPipeline] 🔄 런타임 조정 적용됨")
                
        except Exception as e:
            print(f"[AutoPipeline] 성능 조정 중 오류: {e}")

    def _cpu_task_wrapper(self, gray_frame: np.ndarray) -> tuple[list, list]:
        """스레드 안전한 CPU 작업 래퍼 (에러 처리 강화)"""
        try:
            face_eye_detector = FaceEyeDetector(self.cfg.face_cascade, self.cfg.eye_cascade)
            return face_eye_detector.detect(gray_frame)
        except cv2.error as cv_err:
            print(f"[CPU Worker] OpenCV 오류: {cv_err}")
            return ([], [])
        except Exception as e:
            print(f"[CPU Worker] 예상치 못한 오류: {e}")
            return ([], [])

    def _dispatch_cpu_tasks(self):
        """CPU 작업 분배 (적응형 눈 검출 간격)"""
        try:
            while not self.stop_event.is_set():
                item = self.read_queue.get()
                if item is None:
                    break
                    
                frame_idx, frame = item
                
                # 동적 눈 검출 간격 적용
                should_detect_eyes = (frame_idx % self.optimal_settings.eye_detection_interval == 0)
                
                if should_detect_eyes:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    future = self.cpu_executor.submit(self._cpu_task_wrapper, gray_frame)
                else:
                    # 눈 검출 건너뛰기
                    future = Future()
                    future.set_result(([], []))
                    
                self.cpu_queue.put((frame_idx, frame, future))
                
        except Exception as e:
            print(f"[CPU Dispatcher] 오류: {e}")
            self.stop_event.set()
        finally:
            self.cpu_queue.put(None)
            print("[CPU Dispatcher] 완료")

    def _collect_cpu_results(self):
        """CPU 결과 수집"""
        try:
            while not self.stop_event.is_set():
                item = self.cpu_queue.get()
                if item is None:
                    break
                    
                frame_idx, frame, future = item
                
                try:
                    _, eyes = future.result(timeout=5.0)  # 5초 타임아웃
                    eye_rois = eyes_from_boxes(eyes, self.cfg.safety_margin_px)
                except Exception as e:
                    print(f"[CPU Collector] Future 결과 오류: {e}")
                    eye_rois = []
                    
                self.gpu_queue.put((frame_idx, frame, eye_rois))
                
        except Exception as e:
            print(f"[CPU Collector] 오류: {e}")
            self.stop_event.set()
        finally:
            self.gpu_queue.put(None)
            print("[CPU Collector] 완료")

    def _process_gpu_batch(self):
        """GPU 배치 처리 (OOM 처리 및 메모리 관리 포함)"""
        processed_batches = 0
        
        try:
            while not self.stop_event.is_set():
                batch_data = []
                sentinel_received = False
                
                try:
                    # 배치 수집
                    item = self.gpu_queue.get(timeout=1)
                    if item is None:
                        sentinel_received = True
                    else:
                        batch_data.append(item)
                        
                        # 동적 배치 크기까지 수집
                        while len(batch_data) < self.optimal_settings.batch_size:
                            try:
                                item = self.gpu_queue.get_nowait()
                                if item is None:
                                    sentinel_received = True
                                    break
                                batch_data.append(item)
                            except queue.Empty:
                                break
                                
                except queue.Empty:
                    continue

                if not batch_data and sentinel_received:
                    break
                    
                if batch_data:
                    try:
                        # GPU 배치 처리
                        frames = [item[1] for item in batch_data]
                        batch_keypoints = self.pose_detector.infer_batch(frames)

                        for i, (frame_idx, frame, eye_rois) in enumerate(batch_data):
                            kpts = batch_keypoints[i][0] if i < len(batch_keypoints) and batch_keypoints[i] else []
                            elbow_rois = elbows_from_keypoints(kpts, self.cfg.safety_margin_px)
                            all_rois = eye_rois + elbow_rois
                            
                            # GPU 가속 처리 (사용 가능한 경우)
                            if self.gpu_ops:
                                try:
                                    mask = self.gpu_ops.draw_mask_gpu(frame.shape[:2], all_rois)
                                    out_frame = self.gpu_ops.apply_anonymize_gpu(frame, mask, style=self.cfg.style)
                                except Exception as gpu_err:
                                    print(f"[GPU] GPU 처리 실패, CPU 폴백: {gpu_err}")
                                    # CPU 폴백
                                    from .roi import draw_soft_mask, apply_anonymize
                                    mask = draw_soft_mask(frame.shape[:2], all_rois)
                                    out_frame = apply_anonymize(frame, mask, style=self.cfg.style)
                            else:
                                # CPU 처리
                                from .roi import draw_soft_mask, apply_anonymize
                                mask = draw_soft_mask(frame.shape[:2], all_rois)
                                out_frame = apply_anonymize(frame, mask, style=self.cfg.style)
                                
                            self.writer_queue.put((frame_idx, out_frame))

                        processed_batches += 1
                        
                        # 주기적 GPU 메모리 정리
                        if processed_batches % 10 == 0 and torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            
                    except torch.cuda.OutOfMemoryError:
                        # OOM 처리
                        print("[GPU] 🚨 GPU 메모리 부족 감지!")
                        self.auto_config.handle_oom()
                        self.optimal_settings = self.auto_config.optimal_settings
                        
                        # 현재 배치 건너뛰기
                        for frame_idx, frame, _ in batch_data:
                            # CPU 폴백 처리
                            from .roi import draw_soft_mask, apply_anonymize
                            mask = draw_soft_mask(frame.shape[:2], [])
                            out_frame = apply_anonymize(frame, mask, style=self.cfg.style)
                            self.writer_queue.put((frame_idx, out_frame))
                            
                if sentinel_received:
                    break
                    
        except Exception as e:
            print(f"[GPU Processor] 오류: {e}")
            self.stop_event.set()
        finally:
            self.writer_queue.put(None)
            print("[GPU Processor] 완료")

    def _write_frames(self, output_path: str, total_frames: int, w: int, h: int, fps: float):
        """프레임 쓰기 (버퍼 크기 제한 포함)"""
        wr = VideoWriter(output_path, w, h, fps)
        buffer = {}
        next_idx = 0
        processed_count = 0
        max_buffer_size = min(100, self.optimal_settings.queue_size * 2)  # 버퍼 크기 제한
        
        try:
            while True:
                item = self.writer_queue.get()
                if item is None: 
                    break
                
                frame_idx, frame = item
                buffer[frame_idx] = frame

                # 버퍼 크기 제한 (메모리 보호)
                if len(buffer) > max_buffer_size:
                    # 가장 오래된 프레임 드롭
                    oldest_key = min(buffer.keys())
                    dropped_frame = buffer.pop(oldest_key)
                    print(f"[Writer] ⚠️ 버퍼 오버플로우, 프레임 {oldest_key} 드롭")

                # 순차적 프레임 쓰기
                while next_idx in buffer:
                    out_frame = buffer.pop(next_idx)
                    wr.write(out_frame)
                    processed_count += 1
                    
                    if processed_count % 100 == 0:
                        elapsed = time.time() - self.start_time if self.start_time else 1
                        current_fps = processed_count / elapsed
                        print(f"[Writer] 진행률: {processed_count}/{total_frames} ({current_fps:.1f} FPS)")
                        
                    next_idx += 1
                    
        except Exception as e:
            print(f"[Writer] 오류: {e}")
        finally:
            # 남은 프레임 처리
            for idx in sorted(buffer.keys()):
                wr.write(buffer[idx])
                processed_count += 1
                
            print(f"[Writer] 최종 완료: {processed_count} 프레임")
            if wr: 
                wr.release()

    def run(self, input_path: str, output_path: str):
        """메인 실행 함수"""
        # 입력 비디오 정보 확인
        rd = VideoReader(input_path)
        total_frames, w, h, fps = rd.n, rd.w, rd.h, rd.fps
        rd.cap.release()

        if total_frames <= 0: 
            print("[AutoPipeline] ❌ 유효하지 않은 입력 비디오")
            return

        print(f"[AutoPipeline] 🎬 처리 시작: {total_frames}프레임 ({w}x{h} @ {fps:.1f}fps)")
        start_time = time.time()

        # 5단계 병렬 스레드 실행
        threads = [
            threading.Thread(target=self._read_frames, args=(input_path,), name="Reader"),
            threading.Thread(target=self._dispatch_cpu_tasks, name="CPU-Dispatcher"),
            threading.Thread(target=self._collect_cpu_results, name="CPU-Collector"),
            threading.Thread(target=self._process_gpu_batch, name="GPU-Processor"),
            threading.Thread(target=self._write_frames, args=(output_path, total_frames, w, h, fps), name="Writer")
        ]

        # 스레드 시작
        for t in threads: 
            t.start()
        
        # Writer 스레드 완료 대기
        threads[-1].join()
        self.stop_event.set()
        
        # 나머지 스레드 정리
        for t in threads[:-1]:
            t.join()

        # 리소스 정리
        self.cpu_executor.shutdown(wait=True)
        
        # 최종 성능 리포트
        total_time = time.time() - start_time
        final_fps = total_frames / total_time if total_time > 0 else 0
        
        print(f"\n🎉 [AutoPipeline] 처리 완료!")
        print(f"   ⏱️  총 시간: {total_time:.2f}초")
        print(f"   ⚡ 처리 속도: {final_fps:.2f} FPS")
        print(f"   🔧 최종 설정: {self._get_config_summary()}")
        
        # 성능 개선 제안
        if final_fps < 30:
            print(f"   💡 성능 개선 제안: 더 낮은 해상도나 confidence 값 증가를 고려해보세요")
        elif final_fps > 100:
            print(f"   💡 품질 개선 제안: 더 큰 모델이나 confidence 값 감소를 고려해보세요")