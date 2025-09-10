.PHONY: build run help test demo run-ultra run-speed run-auto-ultra run-auto-speed run-auto hardware-info container-setup container-clean container-status

# 컨테이너 설정
CONTAINER_NAME ?= video-anonymizer-persistent
VOLUME_NAME ?= video-anonymizer-cache

help:
	@echo "make build           # Docker 이미지 빌드"
	@echo "make container-setup # 영구 컨테이너 및 볼륨 생성"
	@echo "make container-clean # 컨테이너 및 볼륨 정리"
	@echo "make container-status # 컨테이너 상태 확인"
	@echo "make run             # 기본 파이프라인 실행"
	@echo "make run-ultra       # 고품질 최적화 파이프라인 실행 (GPU 필수)"
	@echo "make run-speed       # 최고 속도 최적화 파이프라인 실행 (GPU 필수)"
	@echo "make run-auto-ultra  # 🤖 자동 최적화 + 고품질 파이프라인 (권장)"
	@echo "make run-auto-speed  # 🤖 자동 최적화 + 최고 속도 파이프라인"
	@echo "make run-auto        # 🤖 완전 자동 최적화 (파이프라인 자동 선택)"
	@echo "make hardware-info   # 하드웨어 정보 출력"
	@echo "make test            # 스모크 테스트"
	@echo "make demo            # 데모 영상 다운로드"

# 컨테이너 관리
container-setup:
	@echo "🔧 영구 컨테이너 및 볼륨 설정 중..."
	@docker volume create $(VOLUME_NAME) || true
	@docker stop $(CONTAINER_NAME) 2>/dev/null || true
	@docker rm $(CONTAINER_NAME) 2>/dev/null || true
	@echo "✅ 컨테이너 환경 준비 완료"

container-clean:
	@echo "🧹 컨테이너 및 볼륨 정리 중..."
	@docker stop $(CONTAINER_NAME) 2>/dev/null || true
	@docker rm $(CONTAINER_NAME) 2>/dev/null || true
	@docker volume rm $(VOLUME_NAME) 2>/dev/null || true
	@echo "✅ 정리 완료"

container-status:
	@echo "📊 컨테이너 상태:"
	@docker ps -a --filter name=$(CONTAINER_NAME) --format "table {{.Names}}\t{{.Status}}\t{{.CreatedAt}}" || echo "컨테이너 없음"
	@echo "📊 볼륨 상태:"
	@docker volume ls --filter name=$(VOLUME_NAME) --format "table {{.Name}}\t{{.Driver}}\t{{.CreatedAt}}" || echo "볼륨 없음"

build:
	docker build -t video-anonymizer-mvp:latest .

demo:
	bash scripts/download_demo.sh

run:
	@[ -n "$(IN)" ] || (echo "IN=<입력영상> 지정 필요" && exit 1)
	docker run --rm -v $(PWD):/app video-anonymizer-mvp:latest \
	  python -m anonymizer.cli \
	  --input $(IN) \
	  --output $(OUT) \
	  --parts $(PARTS) \
	  --style $(STYLE)

test:
	docker run --rm -v $(PWD):/app video-anonymizer-mvp:latest \
	  pytest -q

# --- 고성능 파이프라인 --- #

# 변수 설정 (make 실행 시 오버라이드 가능)
# 예: make run-ultra IN=my.mp4 BATCH_SIZE=32
IN ?= data/20140413.mp4
OUT_ULTRA ?= output/result_ultra.mp4
OUT_SPEED ?= output/result_speed.mp4

# Ultra-Optimized GPU Run (품질 중심)
# -------------------------------------
BATCH_SIZE_ULTRA ?= 16
CONFIDENCE_ULTRA ?= 0.3
SAFETY_MARGIN_ULTRA ?= 12

run-ultra:
	@echo "🚀 Running Ultra-Optimized Pipeline (Quality Focus)..."
	@echo "Batch Size: $(BATCH_SIZE_ULTRA), Confidence: $(CONFIDENCE_ULTRA), Safety Margin: $(SAFETY_MARGIN_ULTRA)px"
	docker run --gpus all -v "/mnt/d/MYCLAUDE_PROJECT/YOLO-영상내특정객체-모자이크-블러처리-자동화":/workspace -w /workspace video-anonymizer-gpu:slim python3 -m anonymizer.cli_ultra \
		--input "$(IN)" \
		--output "$(OUT_ULTRA)" \
		--pipeline ultra \
		--batch-size $(BATCH_SIZE_ULTRA) \
		--confidence $(CONFIDENCE_ULTRA) \
		--safety-margin $(SAFETY_MARGIN_ULTRA)

# Speed-Optimized GPU Run (속도 중심)
# -------------------------------------
BATCH_SIZE_SPEED ?= 32
CONFIDENCE_SPEED ?= 0.5
SAFETY_MARGIN_SPEED ?= 12

run-speed:
	@echo "⚡ Running Speed-Optimized Pipeline (Speed Focus)..."
	@echo "Batch Size: $(BATCH_SIZE_SPEED), Confidence: $(CONFIDENCE_SPEED), Safety Margin: $(SAFETY_MARGIN_SPEED)px"
	docker run --gpus all -v "/mnt/d/MYCLAUDE_PROJECT/YOLO-영상내특정객체-모자이크-블러처리-자동화":/workspace -w /workspace video-anonymizer-gpu:slim python3 -m anonymizer.cli_ultra \
		--input "$(IN)" \
		--output "$(OUT_SPEED)" \
		--pipeline speed \
		--batch-size $(BATCH_SIZE_SPEED) \
		--confidence $(CONFIDENCE_SPEED) \
		--safety-margin $(SAFETY_MARGIN_SPEED)

# --- 🤖 자동 최적화 파이프라인 --- #

# 자동 최적화 출력 경로
OUT_AUTO_ULTRA ?= output/result_auto_ultra.mp4
OUT_AUTO_SPEED ?= output/result_auto_speed.mp4
OUT_AUTO ?= output/result_auto.mp4

# Docker 실행 헬퍼 함수 (Named Container 사용)
define run_persistent_container
	@echo "🚀 영구 컨테이너로 실행 중..."
	@docker stop $(CONTAINER_NAME) 2>/dev/null || true
	@docker run --gpus all --name $(CONTAINER_NAME) \
		-v "/mnt/d/MYCLAUDE_PROJECT/YOLO-영상내특정객체-모자이크-블러처리-자동화":/workspace \
		-v $(VOLUME_NAME):/cache \
		-w /workspace \
		--rm \
		video-anonymizer-gpu:slim $(1)
endef

# 하드웨어 정보 출력
hardware-info:
	@echo "🔍 하드웨어 정보 감지 중..."
	$(call run_persistent_container,python3 -m anonymizer.cli_ultra_auto --hardware-info)

# 자동 최적화 + Ultra 파이프라인 (품질 우선 + 자동 설정)
run-auto-ultra:
	@echo "🤖 Running Auto-Optimized Ultra Pipeline (Quality + Auto-Config)..."
	@echo "🔍 하드웨어 자동 감지 및 최적 설정 적용 중..."
	$(call run_persistent_container,python3 -m anonymizer.cli_ultra_auto --input "$(IN)" --output "$(OUT_AUTO_ULTRA)" --pipeline ultra --auto --max-performance)

# 자동 최적화 + Speed 파이프라인 (속도 우선 + 자동 설정)
run-auto-speed:
	@echo "🤖 Running Auto-Optimized Speed Pipeline (Speed + Auto-Config)..."
	@echo "🔍 하드웨어 자동 감지 및 최적 설정 적용 중..."
	$(call run_persistent_container,python3 -m anonymizer.cli_ultra_auto --input "$(IN)" --output "$(OUT_AUTO_SPEED)" --pipeline speed --auto --max-performance)

# 완전 자동 최적화 (파이프라인 자동 선택 + 자동 설정)
run-auto:
	@echo "🤖 Running Fully Auto-Optimized Pipeline (Complete Automation)..."
	@echo "🔍 하드웨어 분석 후 최적 파이프라인 자동 선택 및 설정..."
	$(call run_persistent_container,python3 -m anonymizer.cli_ultra_auto --input "$(IN)" --output "$(OUT_AUTO)" --pipeline auto --auto --max-performance)