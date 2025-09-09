#!/usr/bin/env bash
set -euo pipefail
python -m anonymizer.cli       --config configs/default.yaml       "$@"
