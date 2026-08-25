#!/usr/bin/env bash
# Start the local gRPC AI server and the browser demo together.
#
# Usage: ./scripts/run_local.sh
#
# The gRPC server (ai_processor_server.py, ${HOST:-127.0.0.1}:50051) runs in the
# background and the browser demo gateway (server.py, http://${HOST:-127.0.0.1}:8001)
# runs in the foreground. Ctrl+C stops both. On Apple Silicon the gRPC server
# uses the auto backend and falls back to PyTorch/MPS because TensorRT needs
# a Linux x86_64 NVIDIA host.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PROJECT_ROOT}/.venv/bin/python"
GRPC_PORT=50051
WEB_PORT=8001
BIND_HOST="${HOST:-127.0.0.1}"

if [[ ! -x "$PYTHON" ]]; then
  echo "virtualenv python not found: $PYTHON" >&2
  exit 1
fi

cd "$PROJECT_ROOT"

for port in "$GRPC_PORT" "$WEB_PORT"; do
  if nc -z 127.0.0.1 "$port" 2>/dev/null; then
    echo "port ${port} is already in use; stop the process holding it first" >&2
    lsof -nP -iTCP:"$port" -sTCP:LISTEN >&2 || true
    exit 1
  fi
done

"$PYTHON" ai_processor_server.py --host "$BIND_HOST" &
GRPC_PID=$!

cleanup() {
  kill "$GRPC_PID" 2>/dev/null || true
  wait "$GRPC_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "starting gRPC server on ${BIND_HOST}:${GRPC_PORT} (pid ${GRPC_PID})..."

ready=0
for _ in $(seq 1 360); do
  if ! kill -0 "$GRPC_PID" 2>/dev/null; then
    set +e
    wait "$GRPC_PID"
    status=$?
    set -e
    cleanup
    echo "gRPC server exited during startup (exit ${status})" >&2
    exit "$status"
  fi
  if nc -z 127.0.0.1 "$GRPC_PORT" 2>/dev/null; then
    ready=1
    break
  fi
  sleep 0.5
done
if [[ $ready -ne 1 ]]; then
  cleanup
  echo "gRPC server did not open port ${GRPC_PORT} within 180s" >&2
  exit 1
fi

echo "gRPC server:  ${BIND_HOST}:${GRPC_PORT}"
echo "browser demo: http://${BIND_HOST}:${WEB_PORT}"

"$PYTHON" server.py --host "$BIND_HOST" --grpc-target "127.0.0.1:${GRPC_PORT}"
