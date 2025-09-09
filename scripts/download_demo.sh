#!/usr/bin/env bash
set -euo pipefail
mkdir -p data
# 공개 데모(짧은 클립) — 필요시 임의 영상으로 교체하세요.
curl -L -o data/in.mp4       https://sample-videos.com/video321/mp4/720/big_buck_bunny_720p_1mb.mp4

echo "다운로드 완료: data/in.mp4"
