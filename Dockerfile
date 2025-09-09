# syntax=docker/dockerfile:1
FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENCV_LOG_LEVEL=ERROR \
    PIP_NO_CACHE_DIR=1

# system deps (ffmpeg for video i/o, libgl for opencv)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

# default command shows CLI help
CMD ["python", "-m", "anonymizer.cli", "--help"]
