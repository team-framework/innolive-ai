#!/usr/bin/env bash
# FHD WebRTC TensorRT face-swap lab launcher.
# Usage: ./run.sh [port]
# Override paths when needed, e.g. PORT=9090 SOURCE_IMAGE=~/face.png ./run.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PORT="${1:-${PORT:-8088}}"
HOST="${HOST:-0.0.0.0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DETECTOR_ENGINE="${DETECTOR_ENGINE:-models/best_swap_b4.engine}"
SWAPPER_ONNX="${SWAPPER_ONNX:-models/face_swap/inswapper_128.onnx}"
SWAPPER_ENGINE="${SWAPPER_ENGINE:-models/face_swap/inswapper_128_trt11_fp32.engine}"
SOURCE_IMAGE="${SOURCE_IMAGE:-$HOME/Documents/input.png}"
TARGET_ALIGNER="${TARGET_ALIGNER:-insightface}"

for required_path in "$DETECTOR_ENGINE" "$SWAPPER_ONNX" "$SWAPPER_ENGINE" "$SOURCE_IMAGE"; do
  if [[ ! -f "$required_path" ]]; then
    echo "Missing required file: $required_path" >&2
    exit 1
  fi
done

exec "$PYTHON_BIN" -m experiments.trt_swap_client.app \
  --host "$HOST" \
  --port "$PORT" \
  --detector "$DETECTOR_ENGINE" \
  --swapper "$SWAPPER_ONNX" \
  --swapper-engine "$SWAPPER_ENGINE" \
  --source "$SOURCE_IMAGE" \
  --swapper-backend tensorrt \
  --target-aligner "$TARGET_ALIGNER"
