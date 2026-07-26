# InnoLive face video processor

YOLO face segmentation과 stream-local BoT-SORT를 이용해 얼굴을 추적하고 서버에서
모자이크가 합성된 JPEG를 반환합니다. 세션별 AdaFace whitelist에 일치하는 track만
모자이크에서 제외합니다. 운영 진입점은 bidirectional streaming gRPC
`/AiProcessor/ProcessVideo`입니다. `server.py`는 브라우저 화면을 실제 gRPC 서버에 연결하는
시연용 WebSocket-to-gRPC gateway이며 자체적으로 모델을 실행하지 않습니다.

기존 생성 클라이언트는 `output_mode` 기본값에 따라 metadata-only로 계속 동작합니다. 새
클라이언트는 `MOSAIC_JPEG`를 요청하고 `mosaic_jpeg`만 화면에 사용합니다. 합성이나 전송에
실패하면 원본으로 되돌아가지 않습니다.

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

AdaFace ViT-Base KP-RPE WebFace12M 체크포인트와 YuNet landmark detector는 별도 runtime artifact입니다.
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

기본 bind는 `127.0.0.1:50051`이며 `ProcessVideo` 동시 stream 수에는 애플리케이션 admission
제한을 두지 않습니다. YOLO와 AdaFace 모델은 프로세스마다 한 번만 로드되고, Tracker와
track 판정 cache는 RPC마다 분리됩니다. 각 stream의 W5 window, 직렬 YOLO 실행 lane,
AdaFace bounded queue, 실행 1개와 대기 1개로 제한한 mosaic lane을 유지하므로 연결 수가
늘어도 frame 추론이 무한히 병렬 실행되거나 stream 하나가 frame을 무한 적재하지 않습니다.

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
--inference-timeout SECONDS
--adaface-architecture ir18|ir50|ir101|vit_base_kprpe
--adaface-weights PATH
--adaface-detector PATH
--adaface-device auto|cpu|mps|CUDA_INDEX
--adaface-threshold COSINE
--adaface-min-face-size PIXELS
--adaface-queue-capacity N
--adaface-warmup-runs N
--adaface-revalidate-frames N
--adaface-max-pending-per-stream N
--adaface-pending-timeout SECONDS
--max-sessions N          1..1024
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

`VideoProcessorClient`는 최대 W5 요청을 전송하고 서버가 합성한 JPEG를 순서대로 반환합니다.

```python
import asyncio
from pathlib import Path

from grpc_client import VideoProcessorClient


async def main() -> None:
    frames = [Path("frame-1.jpg").read_bytes(), Path("frame-2.jpg").read_bytes()]
    async with VideoProcessorClient("127.0.0.1:50051") as client:
        created = await client.create_session()
        session = client.for_session(created.session_id)
        entries = await session.add_whitelist_many(
            [
                Path("enrollment-face-1.jpg").read_bytes(),
                Path("enrollment-face-2.png").read_bytes(),
            ]
        )
        status = await session.get_whitelist_status()
        print(status.entry_count, status.whitelist_version, list(status.entry_ids))
        print([item.session_id for item in await client.list_sessions()])
        async for result in session.process_jpegs(frames):
            print(result.response.frame_id, result.response.faces)
            protected_jpeg = result.mosaic_jpeg
        await session.delete_whitelist(entries[0].entry_id)
        await session.delete()


asyncio.run(main())
```

직접 frame ID와 timestamp를 관리하려면 `VideoFrame` iterable을
`session.process_video()`에 전달합니다. `client.add_whitelist(..., session_id=...)`와
`client.process_video(..., session_id=...)` 형태도 그대로 지원합니다. 하나의 client/channel로
여러 session의 `ProcessVideo`를 동시에 실행할 수 있으며, client 종료 시 열려 있는 RPC를
모두 취소합니다. client는 JPEG 크기/형식, frame ID 단조 증가,
응답 순서, timestamp echo, payload 크기와 완전한 서버 JPEG를 검증하고 위반 시 fail-closed로
stream을 종료합니다. `source_jpeg`는 요청-응답 상관관계와 기존 호출부 호환을 위해 남아
있지만 출력 fallback으로 사용하지 않습니다. RPC 실패는 `VideoRpcError`의 `method`, gRPC `code`, `details`로
구분할 수 있습니다. `AddWhitelist`의 기본 deadline은 10초이고, 수명이 정해지지 않은
영상 stream의 deadline은 호출자가 `timeout`으로 지정합니다.

## gRPC 인터페이스 설계

`ProcessVideo`의 bidirectional streaming은 한 RPC 안에서 요청·응답 순서를 유지하는
장시간 영상 흐름에 맞는 표준 gRPC 형태입니다. `AddWhitelist`, `DeleteWhitelist`,
`GetWhitelistStatus`, `CreateSession`, `ListSessions`, `DeleteSession`은 각각 독립된 관리
작업이므로 unary RPC로 분리했습니다. 세션 registry는 최대 1024개로 제한되어 목록 응답도
bounded이므로 별도 pagination은 두지 않았습니다. client는 gRPC 권장 방식대로 channel과
stub을 재사용합니다.

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
- `output_mode`: `METADATA_ONLY` 또는 `MOSAIC_JPEG`. stream 도중 변경할 수 없음

Response `ProcessedVideoChunk`:

- `data`: 항상 empty
- `mosaic_jpeg`: `MOSAIC_JPEG` 성공 응답에만 존재하는 Q90 서버 합성 JPEG
- `status_message`: `success` 또는 `failed`
- `faces`: bbox, polygon, confidence, track ID, hold 상태, `whitelisted` 판정
- `timing`, `stats`: 단계별 시간과 detection/tracking 통계
- 오류 시 `error_code`, `error_message`

모자이크할 mask만 union한 뒤 유효 시그마를 유지한 축소 ROI에서 Gaussian Blur를 한 번
계산하고 JPEG를 인코딩합니다. whitelist mask와 보호 mask가 겹치면 보호 mask가 우선합니다.
invalid JPEG와 invalid batch는 해당 frame만 실패하고 stream을 유지합니다. inference,
tracking, mosaic, serialization 실패는 pixel 없는 terminal error 후 stream을 닫습니다. inference timeout은
health를 `NOT_SERVING`으로 바꾸며, 이미 시작한 inference가 끝날 때까지 tracker와 stream
admission을 보존합니다.

Unary `/AiProcessor/AddWhitelist`는 `FaceData.session_id`와 한 장의 static JPEG, PNG 또는
WebP를 받습니다. animated PNG/WebP는 거부합니다. 정확히 한 얼굴을 YuNet 5점 landmark로
정렬해 AdaFace embedding만 메모리에 보관하며 원본이나 crop은 저장하지 않습니다. 응답은
`entry_id`, 현재 `entry_count`, `whitelist_version`을 포함합니다. 빈 세션, decode 실패,
0개·복수 얼굴, 작은 얼굴, 정렬 실패는 `INVALID_ARGUMENT`; 세션·entry·queue 제한은
`RESOURCE_EXHAUSTED`입니다. 영상 `ProcessVideo.data`는 호환성을 위해 계속 완전한 JPEG만
허용합니다.

여러 장은 `AddWhitelist`를 이미지별로 호출하며 각 embedding이 독립 exemplar로 누적됩니다.
Python client의 `add_whitelist_many()`는 queue overflow를 피하기 위해 순차 등록합니다. 기본
한도는 세션당 32장입니다. Unary `/AiProcessor/GetWhitelistStatus`는 원본이나 embedding을
노출하지 않고 해당 세션의 `entry_count`, `whitelist_version`, 삭제에 사용할 `entry_ids`를
반환합니다. `/AiProcessor/DeleteWhitelist`는 `session_id`와 `entry_id`로 exemplar 한 개를
삭제하며 없는 세션이나 entry는 `NOT_FOUND`입니다. 존재하지 않는 유효 세션의 상태 조회는
registry slot을 생성하지 않고 빈 목록과 0/0을 반환합니다.

Unary `/AiProcessor/CreateSession`은 registry lock 안에서 충돌 없는 opaque UUID 기반 ID를
생성하고 빈 `SessionInfo`를 반환합니다. `/AiProcessor/ListSessions`는 생성된 ID와
`entry_count`, `whitelist_version`, 생성 시각, 활성 stream 수를 반환합니다.
`/AiProcessor/DeleteSession`은 유휴 세션만 삭제하며 존재하지 않는 세션은 `NOT_FOUND`, 활성
stream이 있는 세션은 `FAILED_PRECONDITION`입니다. 이 관리 RPC들과 브라우저의
`GET/POST/DELETE /api/sessions`는 로컬 시연·신뢰된 관리 경로용이며 인증을 구현하지 않습니다.
운영 data-plane에 공개하지 말고 인증된 Go proxy 또는 private control network 뒤에 둡니다.

등록은 YuNet score 0.9와 40px 기준으로 정확히 한 얼굴을 요구합니다. 이미 얼굴 track으로
좁혀진 영상 crop은 score 0.6과 24px 기준을 사용하되 복수 얼굴이면 계속 fail-closed합니다.
기본 AdaFace는 공식 CVLFace ViT-Base KP-RPE WebFace12M입니다. YuNet이 찾은 얼굴을 RGB
112×112 crop과 정규화된 5점 landmark로 만들고 좌우 반전 feature를 합성합니다. IR18/50/101
체크포인트도 같은 512차원 계약으로 선택할 수 있으며 legacy IR 계열은 BGR 5점 정렬을
사용합니다. 첫 준비 실패나 cosine 미달은 5 frame 간격으로 최대 두 번 빠르게 재검증한 뒤
기존 장기 재검증 주기로 돌아가며, 임계값 0.4는 낮추지 않습니다.
기존 whitelist 판정의 주기 재검증 중에는 그 판정에 실제로 사용된 entry가 남아 있는 동안만
모자이크 제외 상태를 유지합니다. 재검증 실패나 해당 entry 삭제는 즉시 fail-closed로
전환하고, 늦게 완료된 이전 작업 결과는 다시 적용하지 않습니다.

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

Go 호출부는 모든 `VideoChunk`에 같은 `session_id`와
`VIDEO_OUTPUT_MODE_MOSAIC_JPEG`를 넣고 `ProcessedVideoChunk.mosaic_jpeg`를 downstream으로
전달해야 합니다. 최악 조건의 JPEG와 metadata를 받을 수 있도록 channel에는 최소 5 MiB의
receive limit도 설정합니다. `data`는 deprecated이며 계속 비어 있습니다. 등록 시
`AddWhitelist(FaceData{session_id, data})`, 상태 확인 시 `GetWhitelistStatus`를 호출해야
합니다. exemplar 삭제에는 `DeleteWhitelistRequest{session_id, entry_id}`를 사용합니다.
시연용 자동 세션을 사용할 때는 `CreateSession`, 관리 목록은 `ListSessions` stub을 추가로
생성하고 유휴 세션 정리에는 `DeleteSession`을 사용합니다. `DeleteWhitelist` RPC와
`GetWhitelistStatusResponse.entry_ids = 4`, `VideoChunk.output_mode = 6`,
`ProcessedVideoChunk.mosaic_jpeg = 13`은 additive 변경이며 기존 field number와 RPC
path는 바뀌지 않습니다. 재생성 전 client는 새 JPEG를 읽을 수 없지만 기본 metadata-only
요청은 계속 유효합니다. Python 서버는
인증을 하지 않으므로 앞단에서 검증한 세션 값만 전달해야 합니다.

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
  --grpc-target 127.0.0.1:50051
```

브라우저에서 `http://127.0.0.1:8002`를 엽니다. gateway는 gRPC health service의
`AiProcessor`가 `SERVING`일 때만 카메라 연결을 허용합니다. 모델과 tracker는
`ai_processor_server.py` 프로세스에만 존재하므로 GPU memory도 중복되지 않습니다.

상단 whitelist 패널은 새 브라우저 탭마다 서버가 생성한 충돌 없는 세션을 자동 선택하고,
같은 탭의 새로고침에는 `sessionStorage`의 세션을 재사용합니다. 생성된 세션 목록과 각
등록 수와 활성 stream 수는 화면 아래 하나의 세션 목록에서 확인합니다. 이 목록에서
새 세션을 만들거나 유휴 세션을 삭제할 수 있고, 사용 중인 세션 삭제는 서버도 거부합니다.
등록된 얼굴은 entry ID 목록의 삭제 버튼으로 영상 stream 실행 중에도 개별 삭제할 수 있습니다.
JPEG, PNG, WebP를 포함해 브라우저가 decode할 수 있는 raster 이미지는 긴 변 640 이하의
JPEG로 정규화한 뒤 여러 장을 순차 등록합니다. drag-and-drop과 다중 선택을 지원하며 파일별
변환·등록·성공·실패 상태를 표시합니다. HEIC/AVIF 지원 여부는 브라우저 codec에 따릅니다.
한 파일이 decode·얼굴 수·정렬 검사에 실패해도 다음 파일을 계속 시도합니다.

브라우저는 원본을 로컬에서 blur하지 않고 gRPC 서버가 반환한 JPEG만 그립니다. WebSocket
요청은 ILF1, 성공 응답은 JSON metadata와 JPEG를 함께 담은 binary ILR1 envelope이며 오류는
JSON text입니다. 누락·손상·sequence 불일치 응답은 검은 화면으로 fail-closed합니다.

브라우저는 profile 문자열이나 동시 stream 수를 고정값으로 비교하지 않습니다. ILF1 v2와
`ProcessVideo` gRPC 경로를 확인한 뒤 서버가 광고한 해상도, JPEG 품질, FPS, request window가
더 작은 경우 해당 한도에 맞춥니다. gateway도 WebSocket stream에 별도 admission 제한을
두지 않으며 모든 연결이 공유 gRPC channel을 사용합니다.

## 검사

```bash
.venv/bin/pip install -r requirements-test.txt
.venv/bin/ruff check ai_processor_server.py grpc_client.py server.py service scripts tests
.venv/bin/ruff format --check ai_processor_server.py grpc_client.py server.py service scripts tests
.venv/bin/python -m unittest discover -s tests
node --check web/app.js
node tests/test_web.mjs
```

세션/track 판정 자체의 Python overhead는 모델 추론과 모자이크 합성을 제외한 microbenchmark로
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
서버 모자이크 JPEG, 지속적인 RTT 증가 없음을 검사합니다.

## 구조

```text
ai_processor_server.py          gRPC lifecycle, health, ProcessVideo
grpc_client.py                  bounded W5 async Python client
server.py                       browser-to-gRPC demo gateway
service/runtime.py              auto/TensorRT/PyTorch runtime와 backend 선택
service/tracking.py             stream-local BoT-SORT와 1-frame mask hold
service/adaface_model.py         공유 AdaFace/YuNet bounded worker
service/recognition.py           세션 whitelist와 stream-local track cache
service/frame.py                영상 JPEG와 등록 JPEG/PNG/WebP 검증/decode
service/mosaic.py               fail-closed mask union, blur, JPEG 합성
service/protocol.py             ILF1/ILR1 WebSocket codec와 payload limit
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
