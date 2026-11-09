#!/usr/bin/env bash
# Main gRPC + browser test client with the K22741 MobileFSGAN renderer.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export FACE_SWAP_BACKEND=mobilefsgan
export OPENCV_FOR_THREADS_NUM="${OPENCV_FOR_THREADS_NUM:-1}"
export GRPC_PORT="${GRPC_PORT:-50071}"
export WEB_PORT="${WEB_PORT:-8453}"
exec bash "$ROOT/scripts/run_mobilefaceswap_cuda.sh" "$@"
