# 🚀 빠른 시작 가이드 (v3.0)

이 프로젝트는 `make` 명령어를 사용하여 모든 주요 기능을 실행할 수 있습니다.

## 1. 빌드 (최초 1회)

GPU가 장착된 환경을 기준으로 합니다.
```bash
make build
```

## 2. 실행 (GPU 필수)

두 가지 고성능 파이프라인 중 선택하여 실행할 수 있습니다.

### 품질 우선 실행 (권장)

모든 프레임을 정밀하게 분석하여 높은 품질의 결과물을 생성합니다. (처리 속도: ~70 FPS)
```bash
make run-ultra
```

### 최고 속도 실행

일부 연산을 최적화하여 최고 속도로 결과물을 생성합니다. (처리 속도: ~97 FPS)
```bash
make run-speed
```

## 3. 옵션 조정하여 실행하기

입력/출력 파일이나 성능/품질 관련 옵션을 직접 지정할 수 있습니다.

**예시 1: 다른 파일을 처리**
```bash
make run-ultra IN=path/to/my_video.mp4 OUT=path/to/my_result.mp4
```

**예시 2: 품질 조정 (탐지를 더 민감하게, 마스킹 영역을 더 넓게)**
```bash
make run-ultra CONFIDENCE=0.25 SAFETY_MARGIN=20
```

**예시 3: 성능 조정 (배치 크기를 64로 늘려서 실행)**
```bash
make run-speed BATCH_SIZE=64
```

### 주요 조정 파라미터

- `IN`: 입력 영상 경로
- `OUT`: 출력 영상 경로
- `CONFIDENCE`: 탐지 민감도 (0.1 ~ 1.0 사이, 낮을수록 많이 탐지)
- `SAFETY_MARGIN`: 마스킹 영역 여백 (px)
- `BATCH_SIZE`: GPU 배치 크기 (GPU 메모리에 따라 조절)
