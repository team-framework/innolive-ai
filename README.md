# InnoLive face metadata server

YOLO face segmentation과 stream-local BoT-SORT를 이용해 JPEG 영상의 얼굴 mask, bbox,
track metadata를 반환합니다. 세션별 AdaFace whitelist에 일치하는 track은 blur 대상에서
제외합니다. 운영 진입점은 bidirectional streaming gRPC
`/AiProcessor/ProcessVideo`입니다. `server.py`는 브라우저 화면을 실제 gRPC 서버에 연결하는
시연용 WebSocket-to-gRPC gateway이며 자체적으로 모델을 실행하지 않습니다.

응답에는 입력 JPEG나 처리된 pixel이 포함되지 않습니다. 화면 합성 또는 blur는 client가
보관한 원본 frame과 서버 metadata를 이용해 수행합니다.

## 추론 backend

기본값 `--backend auto --device auto`는 실행 환경에 맞춰 다음 순서로 선택합니다.

1. Linux x86_64, NVIDIA CUDA, TensorRT package, 유효한 `best_b1.engine`이 모두 있으면
   TensorRT를 사용합니다.
2. 그 외 환경에서는 `models/best.pt`를 PyTorch로 로드합니다.
3. PyTorch device는 Apple Silicon의 MPS, NVIDIA CUDA, CPU 순서로 자동 선택합니다.

`--backend tensorrt`를 명시하면 engine, manifest, checkpoint hash, TensorRT version,
CUDA device 중 하나라도 맞지 않을 때 즉시 실패합니다. 명시적 TensorRT 요청을 PyTorch로
조용히 우회하지 않습니다.

현재 체크포인트:

```text
models/best.pt
sha256 a0308be1d294a1265cd95ee1eb2111be0b5e85317a705da25716572bfda82a44
task segment
classes {0: face}
```

현재 생성된 TensorRT engine:

```text
models/best_b1.engine
sha256 db809c2f26cd0ee5a6f0b24175cc148937890a557db26abbb5c3fffb3ea60079
build NVIDIA GeForce RTX 3090 / TensorRT 11.1.0.106 / Torch 2.13.0+cu130
```

engine binary는 GPU별 로컬 산출물이므로 Git에서 제외하고 manifest만 추적합니다.

## 설치

Python 3.12 이상을 권장합니다.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Apple Silicon에서는 별도 옵션 없이 PyTorch와 MPS를 사용합니다.

AdaFace IR-18 체크포인트와 YuNet landmark detector는 별도 runtime artifact입니다.
[models/README.md](models/README.md)의 공식 출처에서 내려받아 기본 경로에 둡니다. 파일이
없거나 로드되지 않으면 영상 처리는 계속되지만 모든 얼굴을 보호하며 `AddWhitelist`는
`FAILED_PRECONDITION`을 반환합니다.

```bash
.venv/bin/python ai_processor_server.py
```

startup log에서 선택된 backend/device를 확인할 수 있습니다. CPU를 강제하려면:

```bash
.venv/bin/python ai_processor_server.py --backend pytorch --device cpu
```

## TensorRT engine 생성

TensorRT engine은 GPU 모델, NVIDIA driver, CUDA, TensorRT stack에 종속됩니다. 반드시 실제
배포 대상인 Linux x86_64 NVIDIA 장비에서 생성합니다. macOS와 Apple Silicon에서는
TensorRT engine을 만들거나 실행할 수 없습니다.

```bash
.venv/bin/pip install -r requirements-export.txt
.venv/bin/python -m scripts.export_tensorrt --device 0 --force
```

이미 생성한 engine을 실행만 하는 host는 `requirements-tensorrt.txt`를 설치하면 됩니다.

입력과 생성물:

```text
models/best.pt
  -> models/best_b1.engine
  -> models/best_b1.engine.json
```

생성 설정은 static batch 1, `imgsz=640`, FP16, `dynamic=false`, NMS 제외입니다. manifest는
checkpoint/engine SHA-256과 build GPU, Python, Torch, Ultralytics, TensorRT version을
기록합니다. checkpoint가 바뀌거나 배포 GPU/TensorRT stack이 달라지면 반드시 다시
생성하고 acceptance를 실행해야 합니다.

TensorRT만 허용하는 배포에서는 다음처럼 실행합니다.

```bash
.venv/bin/python ai_processor_server.py \
  --backend tensorrt \
  --engine models/best_b1.engine \
  --device 0
```

## gRPC 서버

기본 bind는 `127.0.0.1:50051`, 기본 동시 stream 수는 4입니다. YOLO와 AdaFace 모델은
프로세스마다 한 번만 로드되고, Tracker와 track 판정 cache는 `ProcessVideo` RPC마다
분리됩니다. AdaFace는 단일 bounded worker를 사용하며 기본 queue에서는 한 세션이 용량의
절반까지만 동시에 점유할 수 있습니다.

```bash
.venv/bin/python ai_processor_server.py \
  --host 127.0.0.1 \
  --port 50051
```

주요 옵션:

```text
--model PATH              PyTorch checkpoint, 기본 models/best.pt
--engine PATH             TensorRT engine, 기본 models/best_b1.engine
--backend auto|tensorrt|pytorch
--device auto|cpu|mps|CUDA_INDEX
--warmup-runs N
--max-streams N
--inference-timeout SECONDS
--adaface-weights PATH
--adaface-detector PATH
--adaface-device auto|cpu|mps|CUDA_INDEX
--adaface-threshold COSINE
--adaface-min-face-size PIXELS
--adaface-queue-capacity N
--adaface-revalidate-frames N
--adaface-max-pending-per-stream N
--max-sessions N
--max-whitelist-entries N
--ssl-certfile PATH --ssl-keyfile PATH
```

외부 host에 bind할 때는 신뢰된 private network나 인증 proxy 안에서 사용하고 TLS를
적용합니다. built-in TLS는 server certificate 기반 암호화이며 mTLS 인증은 제공하지
않습니다.

표준 gRPC health service는 빈 service와 `AiProcessor`를 제공합니다.

```bash
grpc_health_probe -addr=127.0.0.1:50051 -service=AiProcessor
```

## Python client

`VideoProcessorClient`는 최대 W5 요청을 전송하면서 각 응답을 client가 보관한 정확한
source JPEG와 다시 묶습니다.

```python
import asyncio
from pathlib import Path

from grpc_client import VideoProcessorClient


async def main() -> None:
    frames = [Path("frame-1.jpg").read_bytes(), Path("frame-2.jpg").read_bytes()]
    async with VideoProcessorClient("127.0.0.1:50051") as client:
        session = client.for_session("authenticated-session-id")
        await session.add_whitelist(Path("enrollment-face.jpg").read_bytes())
        async for result in session.process_jpegs(frames):
            print(result.response.frame_id, result.response.faces)
            source_jpeg = result.source_jpeg


asyncio.run(main())
```

직접 frame ID와 timestamp를 관리하려면 `VideoFrame` iterable을
`session.process_video()`에 전달합니다. `client.add_whitelist(..., session_id=...)`와
`client.process_video(..., session_id=...)` 형태도 그대로 지원합니다. 하나의 client/channel로
여러 session의 `ProcessVideo`를 동시에 실행할 수 있으며, client 종료 시 열려 있는 RPC를
모두 취소합니다. client는 JPEG 크기/형식, frame ID 단조 증가,
응답 순서, timestamp echo, metadata 크기, pixel 미포함을 검증하고 위반 시 fail-closed로
stream을 종료합니다. RPC 실패는 `VideoRpcError`의 `method`, gRPC `code`, `details`로
구분할 수 있습니다. `AddWhitelist`의 기본 deadline은 10초이고, 수명이 정해지지 않은
영상 stream의 deadline은 호출자가 `timeout`으로 지정합니다.

## gRPC 인터페이스 설계

`ProcessVideo`의 bidirectional streaming은 한 RPC 안에서 요청·응답 순서를 유지하는
장시간 영상 흐름에 맞는 표준 gRPC 형태입니다. `AddWhitelist`는 독립된 등록 작업이므로
unary RPC가 자연스럽습니다. client는 gRPC 권장 방식대로 channel과 stub을 재사용합니다.

- RPC 전체를 계속할 수 있는 frame decode 오류는 `ProcessedVideoChunk.error_code`로 돌려주고,
  잘못된 session, 자원 한도, 모델 미준비처럼 RPC를 수행할 수 없는 오류는 gRPC status를
  사용합니다.
- `FaceData`, `WhitelistResponse`, package 없는 service는 새 API라면 더 구체적인 이름과
  versioned package가 읽기 쉽지만, 현재 생성 client의 message type과 RPC 경로를 깨지 않도록
  유지했습니다. Python의 `VideoSession`이 이 legacy naming을 호출부에서 감춥니다.
- `session_id`는 stream의 모든 요청에 반복되지만 첫 frame 이후 변경을 거부합니다. 이는
  기존 streaming message에 additive field만 추가해 wire compatibility를 유지하기 위한
  선택입니다.

설계 기준은 [gRPC core concepts](https://grpc.io/docs/what-is-grpc/core-concepts/),
[performance best practices](https://grpc.io/docs/guides/performance/),
[error handling](https://grpc.io/docs/guides/error/),
[deadlines](https://grpc.io/docs/guides/deadlines/),
[Protocol Buffers compatibility guidance](https://protobuf.dev/best-practices/dos-donts/)를
따릅니다.

## wire 계약

`protos/ai_processor.proto`는 기존 client 호환을 위해 package 없이
`/AiProcessor/ProcessVideo` 경로를 유지합니다.

Request `VideoChunk`:

- `data`: 완전한 JPEG 한 장, 최대 4 MiB
- `timestamp`: server가 해석하지 않고 그대로 돌려주는 signed int64
- `frame_id`: 그대로 돌려주는 signed int64
- `batch_size`: legacy field이며 0 또는 1만 허용
- `session_id`: 비어 있지 않은 세션 식별자. 첫 메시지 값으로 stream이 고정됨

Response `ProcessedVideoChunk`:

- `data`: 항상 empty
- `status_message`: `success` 또는 `failed`
- `faces`: bbox, polygon, confidence, track ID, hold 상태, `whitelisted` 판정
- `timing`, `stats`: 단계별 시간과 detection/tracking 통계
- 오류 시 `error_code`, `error_message`

invalid JPEG와 invalid batch는 해당 frame만 실패하고 stream을 유지합니다. inference,
tracking, serialization 실패는 terminal error 후 stream을 닫습니다. inference timeout은
health를 `NOT_SERVING`으로 바꾸며, 이미 시작한 inference가 끝날 때까지 tracker와 stream
admission을 보존합니다.

Unary `/AiProcessor/AddWhitelist`는 `FaceData.session_id`와 한 장의 JPEG를 받습니다. 정확히
한 얼굴을 YuNet 5점 landmark로 정렬해 AdaFace embedding만 메모리에 보관하며 원본이나
crop은 저장하지 않습니다. 응답은 `entry_id`, 현재 `entry_count`, `whitelist_version`을
포함합니다. 빈 세션, decode 실패, 0개·복수 얼굴, 작은 얼굴, 정렬 실패는
`INVALID_ARGUMENT`; 세션·entry·queue 제한은 `RESOURCE_EXHAUSTED`입니다.

proto를 수정한 경우 생성물을 함께 갱신합니다.

```bash
.venv/bin/python -m grpc_tools.protoc \
  -I. \
  --python_out=. \
  --pyi_out=. \
  --grpc_python_out=. \
  protos/ai_processor.proto
```

이 저장소에는 Go module이나 생성된 `.pb.go`가 없습니다. Go 서버가 있는 별도 저장소에서는
기존 module mapping을 유지해 같은 proto를 다시 생성해야 합니다.

```bash
protoc -I. \
  --go_out=. --go_opt=Mprotos/ai_processor.proto=YOUR_GO_MODULE/protos \
  --go-grpc_out=. --go-grpc_opt=Mprotos/ai_processor.proto=YOUR_GO_MODULE/protos \
  protos/ai_processor.proto
```

Go 호출부는 모든 `VideoChunk`에 같은 `session_id`를 넣고, 등록 시
`AddWhitelist(FaceData{session_id, data})`를 호출해야 합니다. Python 서버는 인증을 하지
않으므로 앞단에서 검증한 세션 값만 전달해야 합니다.

## 브라우저 gRPC 시연

브라우저는 raw gRPC bidirectional stream을 직접 사용할 수 없으므로 `server.py`가 얇은
WebSocket 어댑터 역할을 합니다. JPEG와 metadata는 아래 경로를 실제로 통과합니다.

```text
browser -> ILF1 WebSocket -> VideoProcessorClient -> gRPC ProcessVideo -> AI runtime
```

먼저 gRPC 추론 서버를 실행합니다.

```bash
.venv/bin/python ai_processor_server.py \
  --host 127.0.0.1 \
  --port 50051
```

다른 터미널에서 브라우저 gateway를 실행합니다.

```bash
.venv/bin/python server.py \
  --host 127.0.0.1 \
  --port 8002 \
  --session-id local-demo-session \
  --grpc-target 127.0.0.1:50051
```

브라우저에서 `http://127.0.0.1:8002`를 엽니다. gateway는 gRPC health service의
`AiProcessor`가 `SERVING`일 때만 카메라 연결을 허용합니다. 모델과 tracker는
`ai_processor_server.py` 프로세스에만 존재하므로 GPU memory도 중복되지 않습니다.

브라우저는 profile 문자열이나 동시 stream 수를 고정값으로 비교하지 않습니다. ILF1 v1과
`ProcessVideo` gRPC 경로를 확인한 뒤 서버가 광고한 해상도, JPEG 품질, FPS, request window가
더 작은 경우 해당 한도에 맞춥니다. `--max-streams`는 gateway 전체 동시 접속 한도이며
`B1` inference batch나 `W5` stream별 request window와 별개입니다.

## 검사

```bash
.venv/bin/pip install -r requirements-test.txt
.venv/bin/ruff check ai_processor_server.py grpc_client.py server.py service scripts tests
.venv/bin/ruff format --check ai_processor_server.py grpc_client.py server.py service scripts tests
.venv/bin/python -m unittest discover -s tests
node --check web/app.js
node tests/test_web.mjs
```

세션/track 판정 자체의 Python overhead는 모델 추론과 browser blur를 제외한 microbenchmark로
비교할 수 있습니다.

```bash
.venv/bin/python -m scripts.benchmark_recognition --frames 10000 --streams 4
```

배포 GPU의 gRPC acceptance:

```bash
.venv/bin/python -m scripts.benchmark_grpc \
  --target 127.0.0.1:50051 \
  --session-id benchmark-session \
  --input /path/to/sequential-test-video.mp4 \
  --frames 120 \
  --output grpc-acceptance.json
```

gate는 30 FPS, server p95 33.3 ms 이하, W5 이하, 순서가 맞는 모든 terminal 응답,
metadata-only 응답, 지속적인 RTT 증가 없음을 검사합니다.

## 구조

```text
ai_processor_server.py          gRPC lifecycle, health, ProcessVideo
grpc_client.py                  bounded W5 async Python client
server.py                       browser-to-gRPC demo gateway
service/runtime.py              auto/TensorRT/PyTorch runtime와 backend 선택
service/tracking.py             stream-local BoT-SORT와 1-frame mask hold
service/adaface_model.py         공유 AdaFace/YuNet bounded worker
service/recognition.py           세션 whitelist와 stream-local track cache
service/frame.py                transport 공통 JPEG 검증/decode
service/protocol.py             ILF1 WebSocket codec와 payload limit
service/grpc_config.py          gRPC message, keepalive, bind 설정
protos/                         schema와 생성된 Python stub
scripts/export_tensorrt.py      배포 GPU용 TensorRT export
scripts/benchmark_*.py          transport별 acceptance
scripts/benchmark_utils.py      공통 video/statistics helper
tests/                          runtime, transport, lifecycle 회귀 테스트
web/                            개발용 browser client
config/botsort.yaml             tracker threshold 설정
pyproject.toml                  lint/format 기준
```

`B1-640-Q90-W5`는 서빙 profile이지 privacy 품질 승인 자체는 아닙니다. production 전에는
실제 sequential validation set으로 mask precision/recall/F1, mask IoU, bbox recall과 작은
얼굴, 움직임, 가림, 저조도 slice를 별도로 검증해야 합니다.
