# TensorRT Face Swap Lab

독립 테스트 클라이언트입니다. 기존 gRPC server와 production code는 변경하지 않습니다.

3090 Linux host에서 먼저 현재 checkpoint로 **이 클라이언트 전용** dynamic FP16 engine을 만듭니다. 기존 `best_b1.engine`을 대체하지 않습니다.

```bash
pip install -r requirements-export.txt -r requirements-tensorrt.txt \
  'insightface>=0.7,<0.8' 'onnxruntime-gpu>=1.24,<2'
python -m experiments.trt_swap_client.export_detector \
  --checkpoint models/best.pt --output models/best_swap_b16.engine \
  --max-batch 16 --workspace 8 --device 0
```

```bash
pip install -r requirements-tensorrt.txt insightface
python -m experiments.trt_swap_client.app --host 0.0.0.0 --port 8088 \
  --detector models/best_swap_b16.engine \
  --swapper models/face_swap/inswapper_128.onnx \
  --source ~/Documents/input.png
```

브라우저에서 `http://SERVER_IP:8088`로 접근합니다. 신뢰할 수 있는 사설망에서만 실행하세요.

- YOLO26n-seg는 dynamic TensorRT engine을 사용하며 class `0=face`, `1=number_plate`만 유지한다. 이 클래스 계약이 아니면 시작을 거부한다.
- detector queue는 최대 16개 frame을 3ms 동안 모아 batch 처리한다. queue가 가득 차면 오래된 처리를 쌓지 않고 해당 browser frame을 drop한다.
- AdaFace whitelist가 확인된 face는 원본을 유지한다. class 0의 비화이트리스트 face만 InSwapper에 전달한다.
- number plate와 swap 실패 face는 즉시 local Gaussian blur fallback을 적용한다.
- Browser WebSocket은 request/response backpressure를 적용해 처리하지 못할 frame을 계속 전송하지 않는다.

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
- 이 client는 quality를 낮추는 resize, face-skip, frame-skip을 추가하지 않습니다. 실제 10 clients × 30fps는 3090 host에서 browser 및 NVDEC file workload를 나누어 실측해야 합니다.

`models/best_swap_b16.engine`은 3090 Linux에서 현재 `models/best.pt`로 새로 만들어야 하며 Git에 넣지 않습니다.
