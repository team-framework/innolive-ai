#!/usr/bin/env bash
# Download the reproducible AlphaFace experiment assets without adding them to Git.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/.." && pwd)"
asset_dir="$project_dir/models/face_swap/alphaface"

download_verified() {
  local source_url="$1"
  local destination="$2"
  local expected_hash="$3"
  local temporary_path="${destination}.download"

  if [[ -f "$destination" ]] && [[ "$(shasum -a 256 "$destination" | awk '{print $1}')" == "$expected_hash" ]]; then
    echo "Verified existing $(basename "$destination")"
    return
  fi

  curl --fail --location --retry 3 --output "$temporary_path" "$source_url"
  if [[ "$(shasum -a 256 "$temporary_path" | awk '{print $1}')" != "$expected_hash" ]]; then
    echo "SHA-256 verification failed for $(basename "$destination")" >&2
    exit 1
  fi
  mv "$temporary_path" "$destination"
  echo "Downloaded and verified $(basename "$destination")"
}

mkdir -p "$asset_dir"
download_verified \
  "https://github.com/kodek4/VisoMaster-Fusion/releases/download/alphaface-model-v1/alphaface_swapper_fused_norm.onnx" \
  "$asset_dir/alphaface_swapper_fused_norm.onnx" \
  "5514d967ab6cc27e1b0edc092e05ee97d235adccb4da68574a9b1a1e221a4c6a"
download_verified \
  "https://raw.githubusercontent.com/andrewyu90/Alphaface_Official/main/Models/emp.npy" \
  "$asset_dir/emp.npy" \
  "cee626bc81721d71c5d6cb1f76f830b9ae46f595514b0884dd8ae34785576764"
