# 🎬 Video Anonymizer 실제 사용법 가이드

## 🚀 빠른 시작

### 1. 전제 조건
- **Docker** 설치
- **NVIDIA Docker Runtime** 설치 (GPU 사용 시)
- **GPU**: NVIDIA GPU 권장 (RTX 3060 Ti 8GB 검증 완료)

### 2. 프로젝트 클론
```bash
git clone https://github.com/jskang2/video-anonymizer.git
cd video-anonymizer
```

### 3. Docker 이미지 빌드
```bash
# GPU 버전 빌드 (권장)
make build

# 또는 직접 빌드
docker build -f Dockerfile.gpu -t video-anonymizer-gpu:slim .
```

## 🎯 실제 사용 방법

### **방법 1: 자동 최적화 (가장 쉬운 방법)**

```bash
# 1. 컨테이너 환경 설정 (최초 1회만)
make container-setup

# 2. 영상 파일을 data/ 폴더에 배치
cp your_video.mp4 data/

# 3. 자동 최적화 실행
make run-auto-speed IN=data/your_video.mp4

# 결과는 output/result_auto_speed.mp4 에서 확인
```

### **방법 2: 기본 CLI (가장 안정적인 방법)**

```bash
# 영상 처리 실행 (검증된 성공 명령어)
docker run --gpus all --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  video-anonymizer-gpu:slim \
  python -m anonymizer.cli \
  --input data/your_video.mp4 \
  --output output/result.mp4 \
  --parts eyes,elbows \
  --style mosaic
```

### **방법 3: 수동 Docker 실행 (상세 옵션)**

```bash
# 전체 경로 지정 방식 (성공 검증됨)
docker run --gpus all --rm \
  -v "/mnt/d/MYCLAUDE_PROJECT/YOLO-영상내특정객체-모자이크-블러처리-자동화":/workspace \
  -w /workspace \
  video-anonymizer-gpu:slim \
  python -m anonymizer.cli \
  --input data/20140413_10sec.mp4 \
  --output output/result_manual.mp4 \
  --parts eyes,elbows \
  --style gaussian \
  --safety 15 \
  --ttl 3
```

### **방법 4: Make 명령어 사용 (추천)**

```bash
# Makefile을 통한 실행 (가장 편리함)
make run IN=data/your_video.mp4 OUT=output/result.mp4 PARTS=eyes,elbows STYLE=mosaic
```

## 🎛️ 설정 옵션

### 익명화할 신체 부위 (`--parts`)
- `eyes`: 눈 부위
- `elbows`: 팔꿈치 부위
- `eyes,elbows`: 눈과 팔꿈치 모두

### 익명화 스타일 (`--style`)
- `mosaic`: 모자이크 효과 (기본값)
- `gaussian`: 가우시안 블러
- `boxblur`: 박스 블러
- `pixelate`: 픽셀화

### 기타 옵션
- `--safety`: 안전 여백 (픽셀, 기본값: 12)
- `--ttl`: TTL 프레임 수 (기본값: 5)

## 📊 성능 비교

| 방법 | 처리 속도 | 안정성 | 설정 난이도 | 추천 상황 |
|------|----------|--------|-------------|-----------|
| **Auto Speed** | 18+ FPS | 높음 | 쉬움 | 일반적인 사용 |
| **Basic CLI** | 10+ FPS | 매우높음 | 보통 | 안정성 중시 |
| Manual Docker | 변동 | 높음 | 어려움 | 세부 조정 필요 |

## 🔧 문제 해결

### 일반적인 문제들

#### 1. GPU 감지 안됨
```bash
# NVIDIA Docker 런타임 확인
docker run --gpus all --rm nvidia/cuda:11.7-base nvidia-smi

# 하드웨어 정보 확인
make hardware-info
```

#### 2. 메모리 부족 에러
```bash
# 기본 CLI로 실행 (검증된 안정적인 방법)
docker run --gpus all --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  video-anonymizer-gpu:slim \
  python -m anonymizer.cli \
  --input data/input.mp4 \
  --output output/result.mp4 \
  --parts eyes,elbows \
  --style mosaic
```

#### 3. Half Precision 에러 해결
```bash
# 기본 CLI 사용 (--gpu-optimized 플래그 없이)
docker run --gpus all --rm \
  -v "$(pwd)":/workspace \
  -w /workspace \
  video-anonymizer-gpu:slim \
  python -m anonymizer.cli \
  --input data/input.mp4 \
  --output output/result.mp4 \
  --parts eyes,elbows \
  --style mosaic
```

#### 4. 컨테이너 Hang 현상
```bash
# 멈춰있는 컨테이너 종료
docker kill video-anonymizer-persistent

# 기본 CLI 방법으로 재시도
```

## 📁 파일 구조

```
project/
├── data/                    # 입력 영상 파일들
│   ├── your_video.mp4
│   └── 20140413_10sec.mp4   # 테스트 샘플
├── output/                  # 출력 결과 파일들
│   ├── result_auto_speed.mp4
│   └── result_basic.mp4
├── Dockerfile.gpu          # GPU 지원 Docker 설정
├── Makefile               # 빌드 및 실행 명령어
└── CLAUDE.md              # 개발자 문서
```

## ⚡ 성능 최적화 팁

### 1. GPU 메모리 활용
- **8GB GPU**: 배치크기 64까지 안정적
- **4GB GPU**: 배치크기 32 권장
- **4GB 미만**: 배치크기 16 또는 CPU 사용

### 2. 처리 속도 향상
- **Speed 파이프라인** 사용 (`make run-auto-speed`)
- **짧은 영상**으로 먼저 테스트
- **필요한 부위만** 선택 (`--parts eyes` 또는 `--parts elbows`)

### 3. 품질 vs 속도
- **고품질**: `--style gaussian` + 높은 안전 여백
- **고속도**: `--style mosaic` + 낮은 안전 여백

## 📞 지원 및 문의

- **GitHub Issues**: [https://github.com/jskang2/video-anonymizer/issues](https://github.com/jskang2/video-anonymizer/issues)
- **검증 환경**: RTX 3060 Ti 8GB, 16코어 CPU, 62GB RAM
- **테스트 완료**: 2025.01.20

---

## 📝 예제 명령어 요약

```bash
# 가장 간단한 사용법
make run-auto-speed IN=data/my_video.mp4

# 가장 안정적인 사용법  
docker run --gpus all --rm -v "$(pwd)":/workspace -w /workspace \
video-anonymizer-gpu:slim python -m anonymizer.cli \
--input data/my_video.mp4 --output output/result.mp4 --parts eyes,elbows --style mosaic
```