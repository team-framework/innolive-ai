# InnoLive AI gRPC face anonymizer

YOLO26n-seg와 BoT-SORT로 얼굴을 추적하고 mask 영역만 Gaussian blur 처리하는 gRPC 서버입니다. gRPC가 유일한 AI 처리 계층이며, 선택 사항인 FastAPI/WebSocket 서버도 내부적으로 gRPC stream을 호출합니다.

## gRPC 처리 구조

```text
gRPC bidirectional streams
        │
        ├── grpc.aio: 연결과 HTTP/2 flow control
        └── GPU별 bounded microbatch scheduler
                 ├── 설정된 대기 시간 동안 여러 stream의 frame을 B4로 결합
                 ├── shared JPEG decode thread pool
                 ├── fixed B1/B4 TensorRT GPU inference
                 ├── stream별 ordered BoT-SORT
                 └── shared JPEG encode thread pool
                              └── blur JPEG + face metadata
```

- 클라이언트는 B4를 모으지 않고 한 frame을 즉시 보낼 수 있습니다. scheduler는 서로 다른 stream의 frame도 하나의 B4 TensorRT 호출로 합칩니다.
- 브라우저는 프레임을 캡처하는 즉시 보내며, 네트워크에는 최대 두 frame만 유지합니다. scheduler가 서로 다른 stream의 frame을 하나의 B4 TensorRT 호출에 합치므로 브라우저가 B4를 기다리며 100ms 이상 누적하지 않습니다.
- scheduler 대기 시간 안에 B4가 채워지지 않으면 마지막 입력으로 padding합니다. padding 결과는 반환하거나 tracking하지 않습니다.
- GPU별 단일 순서 queue가 detection batch 순서와 각 stream의 BoT-SORT 시간 순서를 함께 보장합니다. 같은 stream은 순차 tracking하고 서로 다른 stream의 tracker는 CPU pool에서 병렬 실행합니다. stream은 생성 시 GPU에 고정 배정됩니다.
- TensorRT worker는 tracking 완료를 기다리지 않고 다음 B4를 실행합니다. tracker future chain이 stream 순서를 보장하는 동안 이전 batch의 tracking과 JPEG encode가 다음 GPU inference와 겹쳐 실행됩니다.
- JPEG decode와 blur JPEG encode는 공유 thread pool에서 실행되고, `grpc.aio` event loop는 네트워크 I/O만 담당합니다.
- 얼굴이 검출되지 않은 프레임은 원본 JPEG를 그대로 반환해 불필요한 재인코딩과 품질 손실을 없앱니다.
- JPEG는 이미 압축된 데이터이므로 별도 gRPC 압축을 사용하지 않습니다.
- queue와 stream별 in-flight 수가 제한되어 느린 클라이언트가 과거 frame과 메모리를 계속 쌓지 못합니다.

## 설치

Python 3.10–3.14와 CUDA가 구성된 환경을 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install "tensorrt-cu13>=11,<12"
python -m grpc_tools.protoc --proto_path=./protos \
  --python_out=__generated__ --pyi_out=__generated__ \
  --grpc_python_out=__generated__ ./protos/ai_processor.proto
```

현재 `models/yolo.engine`은 RTX 3090, CUDA 13, TensorRT 11.1 환경에서 생성한 고정 B4 FP16 엔진입니다. 다른 CUDA 세대나 GPU에서는 해당 환경용 TensorRT를 설치한 뒤 엔진을 다시 생성하는 것이 안전합니다. `.pt`만 사용할 때는 TensorRT 설치를 생략할 수 있습니다.

## gRPC 서버 실행

포트는 실행 파라미터로 지정합니다.

```bash
python ai_processor_server.py --port 50051
python ai_processor_server.py --host 127.0.0.1 --port 50052 --workers 24
```

`ProcessVideo`는 양방향 stream입니다. 입력 `VideoChunk`의 `data`, `timestamp`, `frame_id`, `batch_size`가 출력 순서와 함께 유지됩니다. `batch_size`를 생략하면 즉시 처리할 단일 frame으로 간주합니다. `ProcessedVideoChunk`는 다음 값을 반환합니다.

- blur 처리된 JPEG
- 처리 상태, frame ID, timestamp, 원본 크기
- mask polygon, bbox, confidence, track ID
- 해당 frame의 queue·decode·inference·tracking·encode 처리 시간

## 설정

| 변수 | 기본값 | 의미 |
|---|---:|---|
| `AI_MODEL_PATH` | `models/yolo.engine`, 없으면 `models/yolo.pt` | 모델 경로 |
| `AI_SINGLE_MODEL_PATH` | `models/yolo_b1.engine` (존재 시) | 단일 frame 초저지연용 B1 엔진 경로 |
| `AI_DEVICES` | `0` | 쉼표로 구분한 GPU 또는 `cpu` |
| `AI_CONFIDENCE` | `0.25` | 검출 confidence |
| `AI_IMAGE_SIZE` | `640` | 추론 입력 크기 |
| `AI_DECODE_WORKERS` | CPU 수 기반 | 공유 JPEG decode thread 수 |
| `AI_TRACK_WORKERS` | CPU 수 기반 | 서로 다른 stream의 BoT-SORT 병렬 thread 수 |
| `AI_ENCODE_WORKERS` | CPU 수 기반 | 공유 JPEG encode thread 수, CLI `--workers`로 지정 가능 |
| `AI_JPEG_QUALITY` | `85` | blur 결과 JPEG 품질 |
| `AI_BATCH_WAIT_MS` | `0.5` | 교차 stream B4를 기다리는 최대 시간(ms) |
| `AI_BATCH_QUEUE_SIZE` | `32` | GPU별 inference queue 상한 |
| `GRPC_STREAM_INFLIGHT` | `4` | 한 stream에서 동시에 처리할 frame 상한 |
| `GRPC_MAX_CONCURRENT_RPCS` | `256` | 동시 RPC 상한 |
| `GRPC_MAX_MESSAGE_MB` | `16` | 송수신 message 상한 |
| `GRPC_KEEPALIVE_MS` | `60000` | client channel keepalive 간격 |
| `GRPC_MIN_RECV_PING_MS` | `30000` | server가 허용하는 최소 ping 간격 |
| `GRPC_HOST`, `GRPC_PORT` | `0.0.0.0`, `50051` | CLI 미지정 시 listen 주소 |

CLI의 `--host`, `--port`, `--workers`가 대응 환경 변수보다 우선합니다.

## TensorRT B4 엔진 생성

```bash
AI_DEVICES=0 python export_tensorrt.py --batch 4 --output models/yolo.engine
python ai_processor_server.py --port 50051
```

export는 `--batch 1` 또는 `--batch 4`를 지원하며, `dynamic=False`, `quantize=16`으로 고정됩니다. 서버는 단일 frame에는 B1, 동시 frame에는 B4 엔진을 자동 선택하며 `AI_SINGLE_MODEL_PATH`로 B1 경로를 바꿀 수 있습니다.

## 선택 사항: FastAPI/WebSocket gateway

FastAPI 서버는 테스트 UI와 브라우저 연결을 위한 얇은 gateway입니다. 모델, TensorRT, tracker를 직접 로드하지 않으며 WebSocket 연결마다 gRPC `ProcessVideo` stream 하나를 유지합니다. gRPC 서버에 연결할 수 없으면 시작에 실패합니다.

```bash
pip install -r requirements-web.txt
python ai_processor_server.py --port 50051
python web_gateway.py --grpc-target 127.0.0.1:50051 --port 8000
```

브라우저에서 `http://localhost:8000`을 엽니다. 클라이언트는 각 camera frame을 즉시 보내며 capture와 playout queue를 두 frame으로 제한합니다. queue가 가득 차면 오래된 frame을 버리고 최신 frame을 우선 표시하므로 처리 속도가 순간적으로 떨어져도 지연이 누적되지 않습니다. 네트워크에는 최대 두 요청만 동시에 유지하며 응답은 `frame_id`로 원본 캡처 시각과 연결합니다. 지원 브라우저에서는 실제 새 camera frame이 도착할 때 `requestVideoFrameCallback`으로 캡처하며 JPEG encode가 겹치지 않습니다.

gateway는 WebSocket 수신·gRPC 송신·gRPC 수신·WebSocket 전송을 독립 task로 실행하고 각 queue를 두 frame으로 제한합니다. gRPC 응답은 frame ID 순서를 유지해 브라우저에 전달하므로 작은 pipeline으로 GPU와 JPEG 처리를 겹치면서도 오래된 영상이 화면을 뒤늦게 덮어쓰지 않습니다. 브라우저는 JPEG decode가 끝난 다음 `requestAnimationFrame`에 한 번만 표시해 tearing을 피합니다.

WebSocket 요청은 `[4-byte big-endian JSON 크기][JSON][JPEG payload...]` 형식이며 한 packet에 1–4 frame을 담습니다. 응답은 frame별로 같은 형식의 메시지 하나를 사용합니다. gateway는 이를 gRPC로 전달할 뿐 모델을 직접 실행하거나 브라우저에서 다시 blur하지 않습니다.

응답 `timing.gateway`에는 gateway ingress queue, gRPC write, gRPC 응답 대기, 서버 처리시간을 제외한 gRPC residual, 응답 순서/송신 queue 시간이 포함됩니다.

latency 측정은 화면의 `Capture→display`, 캡처 JPEG, 서버 처리시간을 제외한 WebSocket·gateway·gRPC residual 전체와 세부 구간(q=ingress queue, w=gRPC write, g=gRPC 응답 대기, r=서버 처리시간을 제외한 gRPC residual, o=응답 순서/송신 queue), 서버 JPEG decode, TensorRT·tracking, blur·JPEG encode, 브라우저 JPEG decode를 분리해 표시합니다. 실제 latency는 GPU, 카메라 해상도, 네트워크에 따라 달라질 수 있으므로 정적인 측정값은 제공하지 않습니다.

## 파일 구조

```text
ai_processor_server.py   gRPC 진입점과 비동기 stream 제어
web_gateway.py           선택 사항인 FastAPI/WebSocket→gRPC gateway
service/face_processor.py  TensorRT batch scheduler와 stream tracker
service/transform_image.py mask Gaussian blur와 JPEG encode
service/protocol.py        브라우저 binary protocol
service/grpc_config.py     gRPC channel/server 설정
web/                       테스트 UI
tests/                     scheduler, protocol, blur 회귀 검사
```

## 검사

```bash
python -m unittest discover -s tests
python -m compileall service ai_processor_server.py web_gateway.py
node --check web/app.js
node tests/test_web.mjs
```
