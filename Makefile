.PHONY: build run help test demo

help:
	@echo "make build      # Docker 이미지 빌드"
	@echo "make run IN=... OUT=... PARTS=eyes,elbows STYLE=mosaic  # 처리 실행"
	@echo "make test       # 스모크 테스트"
	@echo "make demo       # 데모 영상 다운로드"

build:
	docker build -t video-anonymizer-mvp:latest .

demo:
	bash scripts/download_demo.sh

run:
	@[ - n "$(IN)" ] || (echo "IN=<입력영상> 지정 필요" && exit 1)
	docker run --rm -v $(PWD):/app video-anonymizer-mvp:latest \
	  python -m anonymizer.cli \
	  --input $(IN) \
	  --output $(OUT) \
	  --parts $(PARTS) \
	  --style $(STYLE)

test:
	docker run --rm -v $(PWD):/app video-anonymizer-mvp:latest \
	  pytest -q
