# TensorRT Face Swap Lab

독립 테스트 클라이언트입니다. 기존 gRPC server와 production code는 변경하지 않습니다.

3090 Linux host에서 먼저 현재 checkpoint로 **이 클라이언트 전용** dynamic FP16 engine을 만듭니다. 기존 `best_b1.engine`을 대체하지 않습니다.

```bash
python -m pip install -r requirements-trt-swap-client.txt
```

```bash
python -m experiments.trt_swap_client.export_detector \
  --checkpoint models/best.pt --output models/best_swap_b4.engine \
  --max-batch 4 --workspace 8 --device 0

python -m experiments.trt_swap_client.export_swapper \
  --onnx models/face_swap/inswapper_128.onnx \
  --output models/face_swap/inswapper_128_trt11_fp32.engine \
  --workspace 2 --precision fp32 --force
```

```bash
python -m experiments.trt_swap_client.app --host 0.0.0.0 --port 8088 \
  --detector models/best_swap_b4.engine \
  --swapper models/face_swap/inswapper_128.onnx \
  --swapper-engine models/face_swap/inswapper_128_trt11_fp32.engine \
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

`/health`의 `swapper_providers.generator` 첫 값이 `TensorRTDirect`인지 확인한다. InSwapper는 ONNX Runtime TensorRT EP가 아니라 current TensorRT에서 만든 direct **FP32** engine으로 실행한다. 이 모델은 FP16 TensorRT raw output이 ONNXRuntime과 크게 달라져 화질이 무너지는 것이 확인됐으므로, runtime은 FP16 engine을 거부한다. 비교용 ONNX Runtime CUDA 경로만 `--swapper-backend cuda`를 명시한다.

## 얼굴 품질 진단

먼저 JPEG/브라우저 표시가 아닌 raw 128×128 결과를 비교한다. `--swap-debug-dir`를 지정하면 TensorRT 실행 결과는 그대로 유지하면서, 같은 blob과 mapped latent를 CPU ONNX Runtime reference에 한 번 더 넣어 최신 얼굴 하나의 진단 묶음을 저장한다.

```bash
python -m experiments.trt_swap_client.app \
  --swapper-backend tensorrt \
  --swap-debug-dir /tmp/inswapper-debug \
  --stream-jpeg-quality 100
```

`/tmp/inswapper-debug/05_onnx_raw_swap.png`와 `06_trt_raw_swap.png`가 처음 비교할 파일이며, `debug.json`에는 input/latent 통계와 ORT-vs-TRT MAE·RMSE·MAX error가 기록된다. 이어서 `07_swap_mask.png`, `08_inverse_warp_swap.png`, `10_after_blend.png`을 보면 paste-back에서 문제가 시작되는지 확인할 수 있다. 덤프가 실패해도 live swap은 blur fallback으로 바뀌지 않는다.

이 저장소의 공식 `inswapper_128.onnx`는 InsightFace metadata상 `input_mean=0`, `input_std=255`를 사용한다. 따라서 target tensor 범위는 **`[0,1]`** 이며, 다른 InSwapper 변형의 `[-1,1]` normalization을 적용하면 안 된다. 시작 시 engine binding의 이름·shape·dtype도 검증하므로, 잘못된/stale 또는 FP16 engine은 화질이 깨진 상태로 실행하지 않고 오류로 중단한다.

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

`models/best_swap_b4.engine`과 `models/face_swap/inswapper_128_trt11.engine`은 동일한 3090 Linux TensorRT 11 environment에서 만들어야 하며 Git에 넣지 않습니다.
