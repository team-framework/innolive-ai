#!/usr/bin/env bash
# Mac: best.pt + metadata + MobileFaceSwap on MPS, SCRFD on CoreML CPU/GPU.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
cd "$PROJECT_ROOT"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

"$PYTHON" - <<'PY'
from pathlib import Path
import importlib.util
import torch
if not torch.backends.mps.is_available():
    raise SystemExit("This launcher requires a Mac with PyTorch MPS available.")
for module in ("paddle", "onnxruntime"):
    if importlib.util.find_spec(module) is None:
        raise SystemExit("Install: .venv/bin/python -m pip install -r requirements-mobilefaceswap.txt")
for name in (
    "models/best.pt", "models/face_metadata_extracter.pt", "config/face_presets.json",
    "models/mobilefaceswap/MobileFaceSwap_224.pdparams",
    "models/mobilefaceswap/arcface.pdparams",
    "models/mobilefaceswap/landmarks/scrfd_10g_bnkps.onnx",
):
    if not Path(name).is_file():
        raise SystemExit(f"Missing artifact: {name}")
from service.face_presets import PresetCatalog, DEFAULT_PRESET_MANIFEST
catalog = PresetCatalog(DEFAULT_PRESET_MANIFEST)
if not catalog.presets or any(not p.image.is_file() for p in catalog.presets.values()):
    raise SystemExit("Install the preset images referenced by config/face_presets.json first.")
PY

for port in 50051 8001; do
  if nc -z 127.0.0.1 "$port" 2>/dev/null; then
    echo "Port $port is already in use; stop the existing server first." >&2
    exit 1
  fi
done

"$PYTHON" -u ai_processor_server.py --host 127.0.0.1 --backend pytorch \
  --model models/best.pt --device mps --face-metadata-device mps \
  --face-swap-backend mobilefaceswap --face-swap-provider mps &
GRPC_PID=$!
WEB_PID=""
cleanup() {
  if [[ -n "$WEB_PID" ]]; then kill "$WEB_PID" 2>/dev/null || true; fi
  kill "$GRPC_PID" 2>/dev/null || true
  if [[ -n "$WEB_PID" ]]; then wait "$WEB_PID" 2>/dev/null || true; fi
  wait "$GRPC_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM

ready=0
for _ in $(seq 1 360); do
  if ! kill -0 "$GRPC_PID" 2>/dev/null; then
    echo "AI server exited during startup." >&2
    exit 1
  fi
  if nc -z 127.0.0.1 50051 2>/dev/null; then ready=1; break; fi
  sleep 0.5
done
if [[ "$ready" != 1 ]]; then echo "AI server startup timed out." >&2; exit 1; fi

echo "MobileFaceSwap-224: http://127.0.0.1:8001 (image / video / camera)"
echo "best.pt + metadata + synthesis: MPS | SCRFD: CoreML | Ctrl+C stops both servers"
"$PYTHON" -u server.py --host 127.0.0.1 --port 8001 \
  --grpc-target 127.0.0.1:50051 --anonymization-mode face_swap &
WEB_PID=$!
# Exit and clean up both children if either server stops.
while kill -0 "$GRPC_PID" 2>/dev/null && kill -0 "$WEB_PID" 2>/dev/null; do sleep 1; done
exit 1
