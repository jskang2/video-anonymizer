.PHONY: build run help test demo run-ultra run-speed

help:
	@echo "make build      # Docker 이미지 빌드"
	@echo "make run        # 기본 파이프라인 실행"
	@echo "make run-ultra  # 고품질 최적화 파이프라인 실행 (GPU 필수)"
	@echo "make run-speed  # 최고 속도 최적화 파이프라인 실행 (GPU 필수)"
	@echo "make test       # 스모크 테스트"
	@echo "make demo       # 데모 영상 다운로드"

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