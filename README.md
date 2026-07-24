# InnoLive AI gRPC head anonymizer

`models/best.pt`의 YOLO segmentation 모델과 BoT-SORT로 사람의 head를 추적하고
mask 영역만 Gaussian blur 처리하는 실시간 비식별화 서버입니다. 예측 클래스는
`0: head`이며, 다른 클래스 매핑의 모델은 서버 시작 단계에서 거부합니다.

gRPC가 AI 처리 계층을 담당하고, 선택 사항인 FastAPI/WebSocket gateway도 내부적으로
gRPC bidirectional stream을 사용합니다. 기존 클라이언트와의 protobuf 호환성을 위해
응답 필드명 `faces`와 `FaceMetadata`는 유지하지만 실제 값은 head detection입니다.

## 처리 구조

```text
gRPC bidirectional streams
        │
        ├── grpc.aio + HTTP/2 flow control
        └── GPU별 bounded microbatch scheduler
                 ├── 여러 stream의 frame을 고정 B4 inference로 결합
                 ├── shared JPEG decode thread pool
                 ├── best_b1/best_b4 TensorRT 또는 best.pt inference
                 ├── stream별 ordered BoT-SORT + mask hold
                 └── shared ROI blur/JPEG encode thread pool
```

- 클라이언트는 frame을 즉시 전송하며 scheduler가 서로 다른 stream의 입력을 B4로
  합칩니다. 대기 시간 안에 B4가 채워지지 않으면 마지막 입력으로 padding합니다.
- 같은 stream의 tracking 순서는 유지하고 서로 다른 stream은 CPU pool에서 병렬로
  처리합니다.
- BoT-SORT detection이 일시적으로 끊기면 Kalman filter 예측 위치로 이전 mask를
  이동해 기본 8 frame 유지합니다. 새 미확정 track도 2 frame 유지합니다.
- detection이 없는 frame은 원본 JPEG를 그대로 반환해 재인코딩과 품질 손실을
  피합니다.
- bounded queue와 stream별 in-flight 제한으로 지연 frame이 무한히 쌓이지 않습니다.

## 설치

Python 3.10–3.14와 CUDA가 구성된 환경을 권장합니다.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m grpc_tools.protoc --proto_path=./protos \
  --python_out=__generated__ --pyi_out=__generated__ \
  --grpc_python_out=__generated__ ./protos/ai_processor.proto
```

NVIDIA 서버에서는 해당 CUDA 버전에 맞는 PyTorch와 TensorRT를 설치하세요.
TensorRT engine은 GPU architecture와 TensorRT 버전에 종속되므로 배포 장비에서
직접 생성해야 합니다. `.pt` fallback만 사용할 때는 TensorRT가 필요하지 않습니다.

## TensorRT engine 생성

실시간 처리에는 고정 shape, FP16, integrated NMS engine을 기본으로 사용합니다.
B4는 여러 stream을 합치는 처리량 최적화용이고 B1은 단일 frame 저지연용입니다.

```bash
python scripts/export_tensorrt.py --device 0 --batch 4
python scripts/export_tensorrt.py --device 0 --batch 1
```

각각 `models/best_b4.engine`, `models/best_b1.engine`을 생성합니다. engine이
`best.pt`보다 최신일 때만 서버가 자동으로 사용하므로 모델을 교체한 뒤에는 engine도
다시 생성해야 합니다. 대표성 있는 calibration dataset이 있을 때는 INT8도 지원합니다.

```bash
python scripts/export_tensorrt.py \
  --device 0 --batch 4 --int8 --data /path/to/data.yaml
```

## gRPC 서버 실행

```bash
python ai_processor_server.py --host 0.0.0.0 --port 50051
python ai_processor_server.py --host 127.0.0.1 --port 50052 --workers 24
```

`ProcessVideo`는 입력 `VideoChunk`의 `timestamp`, `frame_id`, `batch_size`와 응답
순서를 유지합니다. 응답에는 blur JPEG, mask polygon, bbox, confidence, track ID와
queue·decode·inference·tracking·encode 처리 시간이 포함됩니다.

### 주요 설정

| 변수 | 기본값 | 의미 |
|---|---:|---|
| `AI_MODEL_PATH` | 최신 `models/best_b4.engine`, 없으면 `models/best.pt` | B4 모델 경로 |
| `AI_SINGLE_MODEL_PATH` | 최신 `models/best_b1.engine` | 단일 frame B1 모델 경로 |
| `AI_DEVICES` | `0` | 쉼표로 구분한 GPU 또는 `cpu` |
| `AI_CONFIDENCE` | `0.10` | BoT-SORT second-stage detection 포함 confidence |
| `AI_IMAGE_SIZE` | `640` | inference 입력 크기 |
| `AI_MASK_HOLD_FRAMES` | `8` | lost track mask 유지 frame 수 |
| `AI_UNCONFIRMED_HOLD_FRAMES` | `2` | 미확정 track mask 유지 frame 수 |
| `AI_DECODE_WORKERS` | CPU 수 기반 | JPEG decode thread 수 |
| `AI_TRACK_WORKERS` | CPU 수 기반 | stream tracking thread 수 |
| `AI_ENCODE_WORKERS` | CPU 수 기반 | blur/JPEG encode thread 수 |
| `AI_JPEG_QUALITY` | `85` | 출력 JPEG 품질 |
| `AI_BATCH_WAIT_MS` | `0.5` | B4 구성을 기다리는 최대 시간(ms) |
| `AI_BATCH_QUEUE_SIZE` | `32` | GPU별 inference queue 상한 |
| `GRPC_STREAM_INFLIGHT` | `4` | stream별 동시 처리 frame 상한 |
| `GRPC_MAX_CONCURRENT_RPCS` | `256` | 동시 RPC 상한 |
| `GRPC_HOST`, `GRPC_PORT` | `0.0.0.0`, `50051` | listen 주소 |

CLI의 `--host`, `--port`, `--workers`가 환경 변수보다 우선합니다.

## WebSocket gateway

FastAPI gateway는 모델을 직접 로드하지 않고 WebSocket 연결마다 gRPC stream 하나를
유지합니다.

```bash
pip install -r requirements-web.txt
python ai_processor_server.py --port 50051
python web_gateway.py --grpc-target 127.0.0.1:50051 --port 8000
```

브라우저에서 `http://localhost:8000`을 열면 됩니다. capture와 playout queue는 두
frame으로 제한하며, queue가 가득 차면 오래된 frame을 버려 지연 누적을 방지합니다.

## 성능 테스트

AI 처리 비용을 제외한 gRPC transport만 측정하려면 `passthrough` 모드를 사용합니다.
`--max-workers`는 기존 benchmark와의 호환을 위해 `--workers` alias로 유지됩니다.

```bash
python ai_processor_server.py --processing-mode passthrough --port 50051
python benchmarks/run_benchmarks.py --profile smoke
```

상세 사용법은 [`benchmarks/README.md`](benchmarks/README.md)를 참고하세요. 생성된
결과는 Git에서 제외된 `benchmark-results/`에 저장됩니다.

## 파일 구조

```text
ai_processor_server.py      grpc.aio 진입점과 stream 제어
service/face_processor.py   TensorRT batch scheduler와 BoT-SORT mask 안정화
service/transform_image.py  ROI Gaussian blur와 JPEG encode
scripts/export_tensorrt.py  best.pt TensorRT B1/B4 변환
web_gateway.py              WebSocket→gRPC gateway
service/protocol.py         browser binary protocol
service/grpc_config.py      gRPC channel/server 설정
web/                        camera 테스트 UI
tests/                      scheduler, tracking, protocol, blur 회귀 검사
```

## 검사

```bash
ruff check ai_processor_server.py service scripts tests
python -m unittest discover -s tests
python -m compileall service ai_processor_server.py web_gateway.py
node --check web/app.js
node tests/test_web.mjs
```
