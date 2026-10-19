#!/usr/bin/env bash
# Dedicated CUDA service; systemd may supervise this launcher and both children.
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
PYTHON="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export GRPC_PORT="${GRPC_PORT:-50061}"
export WEB_PORT="${WEB_PORT:-8443}"
export AI_BACKEND="${AI_BACKEND:-tensorrt}"
export AI_ENGINE="${AI_ENGINE:-$PROJECT_ROOT/models/best_b1.engine}"
export MOBILEFACESWAP_PRECISION="${MOBILEFACESWAP_PRECISION:-fp16}"

"$PYTHON" - <<'PY'
import os
import socket
from pathlib import Path
import torch
import onnxruntime as ort
from service.runtime import RuntimeConfig, select_runtime
from service.face_presets import PresetCatalog, DEFAULT_PRESET_MANIFEST
if not torch.cuda.is_available():
    raise SystemExit("PyTorch CUDA is unavailable")
if "CUDAExecutionProvider" not in ort.get_available_providers():
    raise SystemExit("Install requirements-mobilefaceswap-cuda.txt without CPU onnxruntime")
select_runtime(RuntimeConfig(backend=os.environ["AI_BACKEND"], device="0", engine=Path(os.environ["AI_ENGINE"])))
for name in ("MobileFaceSwap_224.pdparams", "arcface.pdparams", "landmarks/scrfd_10g_bnkps.onnx"):
    if not (Path("models/mobilefaceswap") / name).is_file():
        raise SystemExit(f"Missing MobileFaceSwap artifact: {name}")
catalog = PresetCatalog(DEFAULT_PRESET_MANIFEST)
if not catalog.presets or any(not p.image.is_file() for p in catalog.presets.values()):
    raise SystemExit("Install the preset images referenced by config/face_presets.json")
for key in ("GRPC_PORT", "WEB_PORT"):
    with socket.socket() as sock:
        # A stopped server can leave TIME_WAIT sockets without an active listener.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", int(os.environ[key])))
print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
PY

TLS_ARGS=()
if [[ -n "${SSL_CERTFILE:-}" || -n "${SSL_KEYFILE:-}" ]]; then
  [[ -f "${SSL_CERTFILE:-}" && -f "${SSL_KEYFILE:-}" ]] || { echo "Both TLS files are required" >&2; exit 1; }
  TLS_ARGS=(--ssl-certfile "$SSL_CERTFILE" --ssl-keyfile "$SSL_KEYFILE")
fi

"$PYTHON" -u ai_processor_server.py --host 127.0.0.1 --port "$GRPC_PORT" \
  --backend "$AI_BACKEND" --engine "$AI_ENGINE" --model models/best.pt --device 0 \
  --face-metadata-device cuda:0 --face-swap-backend mobilefaceswap --face-swap-provider cuda \
  --mobilefaceswap-precision "$MOBILEFACESWAP_PRECISION" &
GRPC_PID=$!
WEB_PID=""
cleanup() {
  if [[ -n "$WEB_PID" ]]; then kill "$WEB_PID" 2>/dev/null || true; fi
  kill "$GRPC_PID" 2>/dev/null || true
  if [[ -n "$WEB_PID" ]]; then wait "$WEB_PID" 2>/dev/null || true; fi
  wait "$GRPC_PID" 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 0' TERM

"$PYTHON" - <<'PY'
import asyncio
import os
import time
import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
async def wait():
    async with grpc.aio.insecure_channel("127.0.0.1:" + os.environ["GRPC_PORT"]) as channel:
        stub = health_pb2_grpc.HealthStub(channel)
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                result = await stub.Check(health_pb2.HealthCheckRequest(service="AiProcessor"), timeout=2)
                if result.status == health_pb2.HealthCheckResponse.SERVING:
                    return
            except grpc.RpcError:
                pass
            await asyncio.sleep(1)
        raise SystemExit("AI server did not become ready within 180 seconds")
asyncio.run(wait())
PY

"$PYTHON" -u server.py --host "${WEB_HOST:-0.0.0.0}" --port "$WEB_PORT" \
  --grpc-target "127.0.0.1:$GRPC_PORT" --anonymization-mode face_swap \
  --target-fps "${TARGET_FPS:-30}" --client-window "${CLIENT_WINDOW:-2}" "${TLS_ARGS[@]}" &
WEB_PID=$!
while kill -0 "$GRPC_PID" 2>/dev/null && kill -0 "$WEB_PID" 2>/dev/null; do sleep 1; done
exit 1
