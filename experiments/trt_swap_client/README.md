# TensorRT Face Swap Lab

독립 테스트 클라이언트입니다. 기존 gRPC server와 production code는 변경하지 않습니다.

3090 Linux host에서 먼저 현재 checkpoint로 **이 클라이언트 전용** dynamic FP16 engine을 만듭니다. 기존 `best_b1.engine`을 대체하지 않습니다.

```bash
python3.11 -m venv .venv-trt10-swap
.venv-trt10-swap/bin/python -m pip install --upgrade pip
.venv-trt10-swap/bin/python -m pip install -r requirements-trt-swap-client.txt
TRT_LIB_DIR="$(.venv-trt10-swap/bin/python - <<'PY'
import sysconfig
from pathlib import Path

site = Path(sysconfig.get_paths()["purelib"])
paths = list(site.rglob("libnvinfer.so.10"))
assert paths, f"TensorRT 10 runtime was not installed below {site}"
print(paths[0].parent)
PY
 )"
echo "$TRT_LIB_DIR"
test -n "$TRT_LIB_DIR"
```

`requirements-tensorrt.txt`의 TensorRT 11은 이 lab과 함께 사용하지 않는다. ONNX Runtime
1.22 TensorRT EP는 TensorRT 10.9/CUDA 12 조합을 사용하며, Python 3.14용 TensorRT 10
binding wheel은 제공되지 않는다. 위의 Python 3.11 environment는 이 lab 전용이다.

라이브러리 경로는 위에서 찾은 경로만 현재 실행에 전달한다. system CUDA/TensorRT나
기존 TensorRT 11 environment는 변경하지 않는다.

```bash
LD_LIBRARY_PATH="$TRT_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
.venv-trt10-swap/bin/python -m experiments.trt_swap_client.export_detector \
  --checkpoint models/best.pt --output models/best_swap_b4_trt10.engine \
  --max-batch 4 --workspace 8 --device 0 --force
```

```bash
LD_LIBRARY_PATH="$TRT_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
.venv-trt10-swap/bin/python -m experiments.trt_swap_client.app --host 0.0.0.0 --port 8088 \
  --detector models/best_swap_b4_trt10.engine \
  --swapper models/face_swap/inswapper_128.onnx \
  --source ~/Documents/input.png
```

브라우저에서 `http://SERVER_IP:8088`로 접근합니다. 신뢰할 수 있는 사설망에서만 실행하세요.

- YOLO26n-seg는 dynamic TensorRT engine을 사용하며 class `0=face`, `1=number_plate`만 유지한다. 이 클래스 계약이 아니면 시작을 거부한다.
- 기본 baseline은 dynamic B4 engine과 최대 4개 frame batch다. model quality는 그대로이며, B8/B16은 B4의 실측 GPU 여유가 확인된 경우에만 별도 engine으로 export한다.
- InSwapper ONNX Runtime CUDA arena는 session당 2GiB 상한(`--swap-ort-mem-gib`)과 exact-request growth를 사용한다. swap에 필요하지 않은 age/gender/106-landmark InsightFace session은 만들지 않는다.
- 기본 `--swapper-backend tensorrt`는 128 swap generator만 ONNX Runtime TensorRT EP FP16·engine/timing cache·CUDA graph로 실행한다. face analysis는 CUDA EP로 유지한다. 초기 실행은 cache build 때문에 느릴 수 있으며, 이후 재기동부터 cache를 재사용한다.
- 기본 `--target-aligner yunet_roi`는 YOLO face box 주변에서 YuNet 5-point landmark만 다시 구한다. landmark를 못 찾거나 YOLO box와 맞지 않으면 기존 InsightFace 640 analysis로 fallback한다. 시각 품질 비교용 legacy 경로는 `--target-aligner insightface`다.
- queue가 가득 차면 오래된 처리를 쌓지 않고 해당 browser frame을 drop한다.
- AdaFace whitelist가 확인된 face는 원본을 유지한다. class 0의 비화이트리스트 face만 InSwapper에 전달한다.
- class 0 mask가 `16,384px`(기본값)보다 작거나 tracking hold 상태면 generator를 실행하지 않고 local Gaussian blur fallback을 적용한다. `--swap-min-mask-area-px`로 조정할 수 있다.
- number plate와 swap 실패 face는 즉시 local Gaussian blur fallback을 적용한다.
- 한 frame의 여러 swap 후보는 InsightFace face analysis를 한 번만 실행한 뒤 YOLO box와 일대일 매칭한다.
- AdaFace는 실제 whitelist enrollment가 시작될 때만 GPU model을 load한다. whitelist가 없는 익명 swap stream은 AdaFace VRAM을 예약하지 않는다.
- Browser WebSocket은 request/response backpressure를 적용해 처리하지 못할 frame을 계속 전송하지 않는다.

응답 metadata에는 `detector_batch_ms`, `swap_ms`, `swap_alignment_ms`, `swap_generator_ms`, `small_face_fallbacks`가 포함된다. 1-session FPS가 낮을 때는 이 값을 먼저 확인한다. `swap_generator_ms`가 크면 generator model이 병목이고, `swap_alignment_ms`가 크면 target landmark 경로가 병목이다.

`/health`의 `swapper_providers.generator` 첫 값이 `TensorrtExecutionProvider`인지 확인한다. TensorRT library가 누락되어 CUDA EP로 fallback되면 client는 시작을 거부한다. 비교용 CUDA 경로만 `--swapper-backend cuda`를 명시한다.

## NVDEC/NVENC file test

FFmpeg가 `h264_nvenc`를 제공하는 3090 host에서는 같은 batcher에 file stream을 추가할 수 있습니다.

```bash
python -m experiments.trt_swap_client.app --host 0.0.0.0 --port 8088 \
  --input-video /data/input.mp4 --hls-dir /tmp/face-swap-hls
```

입력은 NVDEC로 decode하고 output은 NVENC H.264 low-latency HLS (`/hls/live.m3u8`)로 encode합니다. tracker, AdaFace, InSwapper가 OpenCV BGR array를 요구하므로 decode 후 device→host, encode 전 host→device 복사가 한 번씩 있습니다. 따라서 이 경로는 hardware codec I/O이며 **zero-copy claim은 하지 않습니다**.

## Deployment checks

- dynamic engine은 export한 동일한 NVIDIA driver/CUDA/TensorRT 계열의 3090 host에서만 사용합니다. `best.pt` 변경 시 다시 export합니다.
- `models/face_swap/inswapper_128.onnx`는 완전한 유효 ONNX 파일이어야 합니다. 이 repository의 무시된 model artifact는 자동으로 내려받거나 교체하지 않습니다.
- 이 client는 quality를 낮추는 resize, frame-skip을 추가하지 않습니다. 작은 mask fallback은 privacy-first 처리이며, 큰 face의 generator model과 input size는 유지합니다. 실제 10 clients × 30fps는 3090 host에서 browser 및 NVDEC file workload를 나누어 실측해야 합니다.

`models/best_swap_b4_trt10.engine`은 3090 Linux에서 현재 `models/best.pt`로 새로 만들어야 하며 Git에 넣지 않습니다. 기존 TensorRT 11 engine(`best_swap_b4.engine`)은 TensorRT 10 lab에서 사용하지 않습니다.
