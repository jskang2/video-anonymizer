"""
자동 하드웨어 최적화 시스템
컴퓨터 환경에 따라 최적의 배치 크기와 설정값을 자동으로 찾아 설정
"""
from __future__ import annotations
import os
import time
import psutil
import torch
import cv2
import numpy as np
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class HardwareInfo:
    """하드웨어 정보 클래스"""
    cpu_cores: int
    cpu_threads: int
    total_ram_gb: float
    available_ram_gb: float
    gpu_available: bool
    gpu_name: str = ""
    gpu_memory_gb: float = 0.0
    gpu_compute_capability: Tuple[int, int] = (0, 0)

@dataclass
class OptimalSettings:
    """최적화된 설정값 클래스"""
    batch_size: int
    cpu_workers: int
    queue_size: int
    confidence: float
    pose_model: str
    eye_detection_interval: int
    half_precision: bool
    opencv_threads: int

class HardwareProfiler:
    """하드웨어 정보 수집 및 분석"""
    
    @staticmethod
    def detect_hardware() -> HardwareInfo:
        """시스템 하드웨어 정보 자동 감지"""
        # CPU 정보
        cpu_cores = psutil.cpu_count(logical=False)
        cpu_threads = psutil.cpu_count(logical=True)
        
        # 메모리 정보
        memory = psutil.virtual_memory()
        total_ram_gb = memory.total / (1024**3)
        available_ram_gb = memory.available / (1024**3)
        
        # GPU 정보
        gpu_available = torch.cuda.is_available()
        gpu_name = ""
        gpu_memory_gb = 0.0
        gpu_compute_capability = (0, 0)
        
        if gpu_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                gpu_compute_capability = torch.cuda.get_device_capability(0)
                print(f"[GPU] 감지됨: {gpu_name} ({gpu_memory_gb:.1f}GB)")
            except Exception as e:
                print(f"[GPU] 정보 수집 실패: {e}")
                gpu_available = False
        
        return HardwareInfo(
            cpu_cores=cpu_cores,
            cpu_threads=cpu_threads,
            total_ram_gb=total_ram_gb,
            available_ram_gb=available_ram_gb,
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            gpu_memory_gb=gpu_memory_gb,
            gpu_compute_capability=gpu_compute_capability
        )
    
    @staticmethod
    def estimate_memory_usage(batch_size: int, image_size: Tuple[int, int] = (640, 640)) -> float:
        """배치 크기별 예상 GPU 메모리 사용량 계산 (GB)"""
        h, w = image_size
        # YOLOv8 모델 + 입력 이미지 + 중간 결과물 예상
        model_memory = 0.5  # 기본 모델 메모리 (GB)
        input_memory = batch_size * h * w * 3 * 4 / (1024**3)  # float32 기준
        intermediate_memory = input_memory * 3  # 중간 처리 결과
        
        return model_memory + input_memory + intermediate_memory

class AutoTuner:
    """자동 설정 최적화"""
    
    def __init__(self, hardware_info: HardwareInfo):
        self.hw = hardware_info
    
    def calculate_optimal_batch_size(self) -> int:
        """GPU 메모리 기반 최적 배치 크기 계산"""
        if not self.hw.gpu_available:
            return 1
        
        # GPU 메모리의 80%를 사용 목표
        target_memory = self.hw.gpu_memory_gb * 0.8
        
        # 이진 탐색으로 최적 배치 크기 찾기
        min_batch, max_batch = 1, 64
        optimal_batch = 1
        
        for batch_size in [1, 2, 4, 8, 16, 32, 48, 64]:
            estimated_usage = HardwareProfiler.estimate_memory_usage(batch_size)
            if estimated_usage <= target_memory:
                optimal_batch = batch_size
            else:
                break
        
        print(f"[AutoTuner] 최적 배치 크기: {optimal_batch} (예상 메모리: {HardwareProfiler.estimate_memory_usage(optimal_batch):.1f}GB)")
        return optimal_batch
    
    def calculate_cpu_workers(self) -> int:
        """CPU 코어 기반 최적 워커 수 계산"""
        # CPU 코어의 75% 사용 (시스템 예약 고려)
        optimal_workers = max(1, int(self.hw.cpu_cores * 0.75))
        return min(optimal_workers, 8)  # 최대 8개로 제한
    
    def calculate_queue_sizes(self) -> int:
        """RAM 기반 최적 큐 크기 계산"""
        # 사용 가능한 RAM의 10% 사용
        available_memory_mb = self.hw.available_ram_gb * 1024 * 0.1
        
        # 프레임당 대략 2MB 가정 (1080p 기준)
        frame_size_mb = 2
        queue_size = max(8, int(available_memory_mb / frame_size_mb / 4))  # 4개 큐로 분산
        
        return min(queue_size, 64)  # 최대 64로 제한
    
    def select_optimal_model(self) -> str:
        """GPU 성능에 따른 최적 모델 선택"""
        if not self.hw.gpu_available:
            return "models/yolov8n-pose.pt"
        
        # GPU 메모리와 컴퓨트 능력에 따라 모델 선택
        if self.hw.gpu_memory_gb >= 8 and self.hw.gpu_compute_capability[0] >= 7:
            return "models/yolov8m-pose.pt"  # 중간 모델
        elif self.hw.gpu_memory_gb >= 4:
            return "models/yolov8s-pose.pt"  # 작은 모델
        else:
            return "models/yolov8n-pose.pt"  # 나노 모델
    
    def should_use_half_precision(self) -> bool:
        """Half precision 사용 여부 결정"""
        if not self.hw.gpu_available:
            return False
        
        # Compute Capability 7.0 이상에서 half precision 지원
        return self.hw.gpu_compute_capability[0] >= 7

class RuntimeOptimizer:
    """런타임 성능 모니터링 및 동적 조정"""
    
    def __init__(self):
        self.performance_history = []
        self.oom_count = 0
        self.last_adjustment_time = 0
        
    def monitor_gpu_memory(self) -> Tuple[float, float]:
        """GPU 메모리 사용량 모니터링"""
        if not torch.cuda.is_available():
            return 0.0, 0.0
        
        try:
            allocated = torch.cuda.memory_allocated(0) / (1024**3)
            cached = torch.cuda.memory_reserved(0) / (1024**3)
            return allocated, cached
        except:
            return 0.0, 0.0
    
    def should_reduce_batch_size(self, current_fps: float, target_fps: float = 30.0) -> bool:
        """배치 크기 감소 여부 판단"""
        # OOM이 발생했거나 성능이 너무 낮은 경우
        if self.oom_count > 0:
            return True
        
        # 최근 성능이 목표 FPS의 50% 미만인 경우
        if len(self.performance_history) >= 5:
            recent_avg = sum(self.performance_history[-5:]) / 5
            if recent_avg < target_fps * 0.5:
                return True
        
        return False
    
    def record_performance(self, fps: float, memory_usage: float):
        """성능 기록"""
        self.performance_history.append(fps)
        
        # 최근 20개 기록만 유지
        if len(self.performance_history) > 20:
            self.performance_history.pop(0)
    
    def handle_oom_error(self):
        """OOM 에러 처리"""
        self.oom_count += 1
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[RuntimeOptimizer] OOM 발생 (총 {self.oom_count}회)")

class AutoConfig:
    """자동 최적화 설정 생성기"""
    
    def __init__(self, cache_file: str = "auto_config_cache.json"):
        self.cache_file = Path(cache_file)
        self.hardware_info = None
        self.optimal_settings = None
        self.runtime_optimizer = RuntimeOptimizer()
    
    def load_or_detect_hardware(self) -> HardwareInfo:
        """하드웨어 정보 로드 또는 새로 감지"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file, 'r') as f:
                    cached_data = json.load(f)
                    
                # 캐시된 하드웨어 정보와 현재 정보 비교
                current_hw = HardwareProfiler.detect_hardware()
                cached_hw = HardwareInfo(**cached_data.get('hardware_info', {}))
                
                # 주요 하드웨어가 변경되지 않았으면 캐시 사용
                if (current_hw.gpu_memory_gb == cached_hw.gpu_memory_gb and 
                    current_hw.cpu_cores == cached_hw.cpu_cores):
                    print("[AutoConfig] 캐시된 하드웨어 정보 사용")
                    return cached_hw
            except Exception as e:
                print(f"[AutoConfig] 캐시 로드 실패: {e}")
        
        print("[AutoConfig] 하드웨어 정보 새로 감지")
        return HardwareProfiler.detect_hardware()
    
    def generate_optimal_config(self, user_overrides: Dict[str, Any] = None) -> OptimalSettings:
        """최적화된 설정 생성"""
        self.hardware_info = self.load_or_detect_hardware()
        tuner = AutoTuner(self.hardware_info)
        
        # 자동 계산된 최적값
        optimal_batch_size = tuner.calculate_optimal_batch_size()
        optimal_cpu_workers = tuner.calculate_cpu_workers()
        optimal_queue_size = tuner.calculate_queue_sizes()
        optimal_model = tuner.select_optimal_model()
        optimal_half_precision = tuner.should_use_half_precision()
        
        # 성능 기반 confidence 조정
        if self.hardware_info.gpu_memory_gb >= 8:
            confidence = 0.25  # 고성능 GPU는 낮은 threshold
        elif self.hardware_info.gpu_memory_gb >= 4:
            confidence = 0.3
        else:
            confidence = 0.4  # 저사양 GPU는 높은 threshold
        
        # Eye detection interval (GPU 성능에 따라)
        if self.hardware_info.gpu_memory_gb >= 6:
            eye_interval = 1  # 모든 프레임
        elif self.hardware_info.gpu_memory_gb >= 4:
            eye_interval = 2  # 2프레임마다
        else:
            eye_interval = 4  # 4프레임마다
        
        self.optimal_settings = OptimalSettings(
            batch_size=optimal_batch_size,
            cpu_workers=optimal_cpu_workers,
            queue_size=optimal_queue_size,
            confidence=confidence,
            pose_model=optimal_model,
            eye_detection_interval=eye_interval,
            half_precision=optimal_half_precision,
            opencv_threads=min(optimal_cpu_workers, 8)
        )
        
        # 사용자 오버라이드 적용
        if user_overrides:
            for key, value in user_overrides.items():
                if hasattr(self.optimal_settings, key):
                    setattr(self.optimal_settings, key, value)
                    print(f"[AutoConfig] 사용자 설정 적용: {key}={value}")
        
        # 설정 캐시 저장
        self._save_cache()
        
        # 설정 출력
        self._print_settings()
        
        return self.optimal_settings
    
    def _save_cache(self):
        """설정 캐시 저장"""
        try:
            cache_data = {
                'hardware_info': self.hardware_info.__dict__,
                'optimal_settings': self.optimal_settings.__dict__,
                'timestamp': time.time()
            }
            with open(self.cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            print(f"[AutoConfig] 캐시 저장 실패: {e}")
    
    def _print_settings(self):
        """최적화된 설정 출력"""
        print("\n" + "="*50)
        print("🚀 자동 최적화된 설정")
        print("="*50)
        print(f"💻 CPU: {self.hardware_info.cpu_cores}코어 → {self.optimal_settings.cpu_workers}워커")
        print(f"🎮 GPU: {self.hardware_info.gpu_name}")
        print(f"   └─ 메모리: {self.hardware_info.gpu_memory_gb:.1f}GB → 배치크기: {self.optimal_settings.batch_size}")
        print(f"   └─ 모델: {self.optimal_settings.pose_model}")
        print(f"   └─ Half Precision: {self.optimal_settings.half_precision}")
        print(f"🔧 성능 설정:")
        print(f"   └─ Confidence: {self.optimal_settings.confidence}")
        print(f"   └─ Eye Interval: {self.optimal_settings.eye_detection_interval}")
        print(f"   └─ Queue Size: {self.optimal_settings.queue_size}")
        print("="*50)
    
    def adapt_runtime_settings(self, current_fps: float, memory_allocated: float):
        """런타임 설정 적응형 조정"""
        self.runtime_optimizer.record_performance(current_fps, memory_allocated)
        
        # 5초마다 한번씩만 조정 (너무 빈번한 조정 방지)
        current_time = time.time()
        if current_time - self.runtime_optimizer.last_adjustment_time < 5.0:
            return self.optimal_settings
        
        if self.runtime_optimizer.should_reduce_batch_size(current_fps):
            # 배치 크기 50% 감소
            new_batch_size = max(1, self.optimal_settings.batch_size // 2)
            if new_batch_size != self.optimal_settings.batch_size:
                print(f"[RuntimeOptimizer] 배치 크기 조정: {self.optimal_settings.batch_size} → {new_batch_size}")
                self.optimal_settings.batch_size = new_batch_size
                self.runtime_optimizer.last_adjustment_time = current_time
        
        return self.optimal_settings
    
    def handle_oom(self):
        """OOM 에러 발생시 설정 조정"""
        self.runtime_optimizer.handle_oom_error()
        
        # 배치 크기 절반으로 감소
        new_batch_size = max(1, self.optimal_settings.batch_size // 2)
        print(f"[AutoConfig] OOM 처리: 배치크기 {self.optimal_settings.batch_size} → {new_batch_size}")
        self.optimal_settings.batch_size = new_batch_size
        
        return self.optimal_settings

# 편의 함수
def create_auto_optimized_config(user_overrides: Dict[str, Any] = None) -> OptimalSettings:
    """자동 최적화된 설정을 생성하는 편의 함수"""
    auto_config = AutoConfig()
    return auto_config.generate_optimal_config(user_overrides)

if __name__ == "__main__":
    # 테스트 실행
    print("🔍 하드웨어 정보 감지 중...")
    hw_info = HardwareProfiler.detect_hardware()
    
    print("\n📊 감지된 하드웨어:")
    print(f"CPU: {hw_info.cpu_cores}코어 / {hw_info.cpu_threads}스레드")
    print(f"RAM: {hw_info.total_ram_gb:.1f}GB (사용가능: {hw_info.available_ram_gb:.1f}GB)")
    if hw_info.gpu_available:
        print(f"GPU: {hw_info.gpu_name} ({hw_info.gpu_memory_gb:.1f}GB)")
    else:
        print("GPU: 없음")
    
    print("\n⚙️ 최적화된 설정 생성 중...")
    optimal_config = create_auto_optimized_config()