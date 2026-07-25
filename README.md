# InnoLive B1-640-Q90-W5 face anonymizer

InnoLive 초저지연 영상 추론 서빙 표준 **1.1**의 고정 경로만 구현한 face
segmentation/tracking 서버입니다.
클라이언트는 long edge 640 JPEG Q90 frame을 ILF1 WebSocket으로 전송하고, 서버는
정적 FP16 TensorRT B1 추론과 connection-local BoT-SORT를 순서대로 실행한 뒤
metadata JSON만 반환합니다. 입력 JPEG나 원본 pixel은 응답하지 않습니다.

## 고정 serving profile

| 항목 | 값 |
|---|---|
| TensorRT | static FP16, batch 1 |
| 모델 입력 | 640 |
| 전송 frame | aspect ratio 유지, long edge 640 |
| 작은 입력 | 확대하지 않음 (`upscale_small_inputs=false`) |
| JPEG | Q90 |
| WebSocket protocol | persistent WS, binary ILF1 request / text JSON response |
| client window | in-flight 최대 5 |
| capture 대기열 | latest 1 |
| local decode | 실제 send한 exact JPEG를 send 직후 비동기 predecode |
| render | active 1 + latest-waiting 1, monotonic commit, playout 없음 |
| process당 stream/runtime | 1 / 1 |
| GPU scheduler | 직렬 B1 lane 1개 |
| detector ingress | 0.01 |
| existing-track continuation | 0.05 |
| new-track activation | 0.25 |
| mask hold | 1 frame, confidence decay 0.90 |
| 응답 | sequence가 있는 metadata JSON only |

W5는 GPU batch가 아닙니다. GPU는 한 frame씩 sequence 순서대로 처리하며 W5는
네트워크 RTT로 인한 stop-and-wait 공백만 겹칩니다. B4, dynamic engine, raw JPEG
request, batch request, 응답 JPEG, gRPC 우회 경로는 지원하지 않습니다.

WebRTC는 현재 표준 경로가 아닙니다. `capture/sent < 30`이면 capture/JPEG encode,
`sent=30`인데 result가 낮고 WebSocket `bufferedAmount`가 증가하면 network/transport,
server queue가 증가하면 runtime, `result=30`인데 display가 낮으면 local decode/render를
먼저 진단합니다.

## 구조

```text
web/app.js
  camera frame callback
    -> accumulated 30 FPS deadline + 2 ms early tolerance
    -> aspect resize (long edge 640, no upscale)
    -> one JPEG Q90 encoder
    -> latest pending frame (0..1)
    -> ILF1 sender (in-flight 0..5)
         -> socket.send(exact JPEG)
         -> start exact-JPEG bitmap decode immediately
         -> retain seq -> {JPEG, bitmap lease}
         |
         v
server.py
  boundary validation
    -> JPEG decode + dimension/pixel limit
    -> singleton warmed service/runtime.py
    -> serialized static B1 TensorRT inference
    -> connection-local service/tracking.py
    -> bounded polygon metadata JSON
         |
         v
web/app.js
  terminal result matched by seq
    -> active 1 + latest-waiting 1 renderer
    -> exact bitmap + mask blur/bbox overlay
    -> monotonic commit, close bitmap exactly once
```

핵심 코드는 다음 다섯 경계로만 구성됩니다.

- `service/protocol.py`: ILF1 request/terminal response codec와 byte limit
- `server.py`: WebSocket lifecycle, boundary, readiness, admission, metric
- `service/runtime.py`: manifest 검증, singleton TensorRT, 직렬 GPU lane
- `service/tracking.py`: connection-local BoT-SORT와 1-frame mask hold
- `web/app.js`: jitter-safe capture, send-time predecode, latest-1/W5 transport,
  bounded monotonic fail-closed render

## 설치

Python 3.12와 NVIDIA driver가 설치된 배포 GPU에서 실행합니다. 현재 engine
manifest는 RTX 3090, TensorRT `11.1.0.106`, Ultralytics `8.4.104` 환경입니다.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
```

TensorRT engine은 GPU/TensorRT stack에 종속됩니다. 다른 배포 장비에서는 반드시
그 장비에서 다시 변환하고 acceptance를 통과해야 합니다.

## B1 engine 생성

`models/best.pt`를 교체한 뒤 export 의존성을 설치하고 B1 engine만 다시 만듭니다.

```bash
.venv/bin/pip install -r requirements-export.txt
.venv/bin/python scripts/export_tensorrt.py --device 0 --force
```

생성물:

- `models/best_b1.engine`: GPU별 binary이며 Git에서 제외
- `models/best_b1.engine.json`: source/engine SHA, B1/640/FP16/static, class와
  build provenance

서버는 startup에서 engine과 manifest 존재 여부, engine/checkpoint SHA-256,
`B1`, `640`, `FP16`, `dynamic=false`, `{0: face}`, TensorRT runtime version을 모두
검증합니다. 하나라도 다르면 readiness를 `false`로 유지하고 stream을 거부합니다.

현재 tracked checkpoint SHA-256:

```text
c1b62c95c8a901411d46767a8377b9ec27d50c408b396ad15fbab9b9748ae35b
```

## 서버 실행

```bash
.venv/bin/python server.py \
  --engine models/best_b1.engine \
  --device 0 \
  --host 0.0.0.0 \
  --port 8001
```

브라우저는 `http://127.0.0.1:8001`로 접속합니다. 원격 webcam은 secure context가
필요하므로 TLS를 사용합니다.

```bash
.venv/bin/python server.py \
  --engine models/best_b1.engine \
  --host 0.0.0.0 \
  --port 8443 \
  --ssl-certfile /path/to/fullchain.pem \
  --ssl-keyfile /path/to/privkey.pem
```

`/healthz`는 profile, readiness, runtime instance, active stream, GPU
memory/utilization과 누적 stage metric을 반환합니다. `/readyz`는 완전히 warm-up된
경우에만 HTTP 200을 반환합니다.

## ILF1 wire contract

요청 binary message 하나가 frame 하나입니다.

```text
offset  size  field
0       4     ASCII "ILF1"
4       4     unsigned sequence, big endian
8       N     JPEG bytes
```

- sequence는 연결 안에서 엄격히 증가하며 `0..2^32-1`입니다.
- wrap 전에 연결을 다시 수립합니다.
- JPEG 최대 4 MiB, decoded 최소 32×32, long edge 최대 640, pixel 최대
  `640×640`입니다.
- 수락된 sequence는 정확히 하나의 `result` 또는 `error` terminal로 끝납니다.

결과는 UTF-8 JSON text이며 JPEG/pixel을 포함하지 않습니다.

```json
{
  "v": 1,
  "type": "result",
  "seq": 123,
  "width": 640,
  "height": 360,
  "objects": [],
  "timing_ms": {
    "decode": 1.2,
    "queue": 0.0,
    "inference": 7.4,
    "tracking": 0.5,
    "serialize": 0.2,
    "server_total": 9.4
  }
}
```

오류도 동일 sequence의 terminal입니다.

```json
{
  "v": 1,
  "type": "error",
  "seq": 123,
  "code": "DECODE_FAILED",
  "message": "frame could not be decoded as JPEG"
}
```

## Fail-closed 동작

- truncated header처럼 sequence를 복원할 수 없으면 protocol close합니다.
- 복원 가능한 잘못된 header/sequence/JPEG는 해당 sequence `error`를 보냅니다.
- oversized payload/frame은 inference 전에 거부합니다.
- inference, tracking, serialization 실패는 부분 결과 없이 단계별 `error`를
  반환하고 tracker 불확실성을 피하기 위해 연결을 닫습니다.
- timeout/disconnect/unknown·duplicate·regressed terminal이면 client는 모든 local
  frame을 폐기하고 output canvas를 검정색으로 덮습니다.
- 유효 result와 일치한 local frame만 렌더하며 이미 표시한 sequence보다 과거
  frame으로 돌아가지 않습니다.
- active render는 새 result 때문에 취소하지 않고, 아직 시작하지 않은 waiting 한 칸만
  최신 result로 교체합니다. in-flight 5 + active 1 + waiting 1로 bitmap 소유권을
  최대 7개로 제한하며 정상/교체/timeout/error/reset/decode-late 경로에서 정확히 한 번
  해제합니다.
- 객체에 유효한 bounded polygon이 없거나 browser blur를 사용할 수 없으면 원본을
  표시하지 않고 blackout합니다.
- 화면의 source video는 **개발용 미보호 preview**입니다. 방송/제품 출력으로
  사용하면 안 됩니다.

## 관측 metric

브라우저 diagnostics는 다음을 분리해 기록합니다.

- `capture_fps`, `encoded_fps`, `sent_fps`, `result_fps`, `display_fps`
- `pending_frames`, `inflight_requests`, `capture_dropped`, `stale_results`
- `websocket_buffered_bytes`, `bitmap_owners`, `bitmap_owner_peak`
- `jpeg_bytes`, `metadata_bytes`
- `round_trip_ms`, RTT p50/p95
- `local_jpeg_decode_ms`, `result_to_display_ms`, `capture_to_display_ms`와 p50/p95
- raw frame count, monotonic elapsed time, 반올림 전 exact capture/result/display FPS,
  exact display/capture ratio

현재 snapshot은 `window.__INNOLIVE_METRICS__`, 최대 300개의 1초 표본은
`window.__INNOLIVE_METRIC_HISTORY__`, 현재 bitmap 소유 수는
`window.__INNOLIVE_BITMAP_OWNERS__`에서 확인할 수 있습니다. 브라우저 gate 결과에는
이 값과 browser build, hardware acceleration, camera source, display refresh,
RTT/jitter를 함께 보존합니다.

서버는 decode, queue, inference, tracking, serialize, total p50/p95와 runtime
instance, readiness, active stream, GPU memory/utilization, detection/track/
low-confidence continuation/held-mask/error stage를 기록합니다. 모든 duration은 각
process 내부 monotonic clock만 사용하며 동기화되지 않은 one-way latency를 만들지
않습니다.

## 검사

```bash
.venv/bin/pip install -r requirements-test.txt
ruff check server.py service scripts tests
python -m unittest discover -s tests
node --check web/app.js
node tests/test_web.mjs
```

테스트에는 malformed/truncated/oversized request, duplicate sequence, decode/
inference/tracking/serialization failure, timeout, readiness failure, stream admission,
metadata-only response, B1 manifest와 tracker isolation/hold 검사가 포함됩니다.

## A. Server/transport result acceptance

실제 배포 hardware/network profile과 120 frame 이상의 고정 영상으로 실행합니다.

```bash
.venv/bin/python scripts/benchmark_stream.py \
  --url ws://127.0.0.1:8001/ws \
  --input /path/to/sequential-test-video.mp4 \
  --frames 120 \
  --output acceptance.json
```

도구가 자동 판정하는 gate:

- sustained result FPS `>= 30`
- server total p95 `<= 33.3 ms/frame`
- in-flight `<= 5`
- 모든 sequence가 순서대로 terminal result 하나로 종료
- 응답에 JPEG/raw pixel이 없음
- 첫/마지막 구간 RTT가 지속적으로 증가하지 않음

이 CLI gate는 camera callback, browser JPEG encode/decode, overlay와 실제 display
성능을 증명하지 않습니다.

## B. 실제 browser/app display acceptance

foreground target browser와 실제 camera 또는 30 FPS fake camera에서 최소 30초씩
3회 측정합니다. 매 run 시작 전에 페이지의 `중지` 후 `카메라 시작`으로 session metric을
초기화하고, DevTools에서 다음 값을 JSON으로 보존합니다.

```js
JSON.stringify({
  samples: window.__INNOLIVE_METRIC_HISTORY__,
  final: window.__INNOLIVE_METRICS__,
  bitmap_owners: window.__INNOLIVE_BITMAP_OWNERS__,
}, null, 2)
```

각 run의 필수 gate:

- 1초 `capture_fps`, `result_fps`, `display_fps` 표본 중앙값이 각각 30 이상
- `display_frames / capture_frames >= 0.99`
- wall-clock exact FPS의 반올림 전 값과 frame count/elapsed time 보존
- 최소 30초×3회이며 latency queue의 지속 증가와 인위적 playout delay가 없음
- commit sequence 단조 증가, JPEG/metadata mismatch, timeout, protocol error 0
- 중지 후 `bitmap_owners == 0`

`node tests/test_web.mjs`는 누적 capture deadline, active/waiting renderer와 bitmap
lifecycle 회귀 검사일 뿐 실제 browser display gate를 대체하지 않습니다.

## 모델 품질 gate는 별도

`B1-640-Q90-W5`는 서빙 방법론이며 현재 `best.pt`의 production 품질 승인서가
아닙니다. 기준 문서에 기록된 동일 checkpoint 계열의 temporally stabilized mask
recall은 train-13 `0.050025`, train-11 `0.326729`였습니다. 서빙/네트워크를
최적화해도 검출되지 않은 얼굴을 tracker가 복구할 수 없습니다.

production 배포 전에는 동일 held-out sequential dataset으로 mask precision/recall/
F1, matched mask IoU, bbox recall과 tiny/small face, 빠른 움직임, 가림, 저조도 slice를
별도로 승인해야 합니다. 이 품질 gate가 없거나 실패한 동안 방송 경로는 blur-all
또는 blackout fallback을 사용해야 합니다.

## 최종 파일 구조

```text
server.py                       FastAPI/WebSocket lifecycle와 health/readiness
service/protocol.py             ILF1 bounded codec
service/runtime.py              singleton static-B1 TensorRT runtime
service/tracking.py             connection-local BoT-SORT와 mask hold
web/app.js                      1.1 capture/predecode/W5/bounded renderer client
web/index.html, web/style.css   검증 UI
config/botsort.yaml             0.05 continuation / 0.25 activation
models/best.pt                  source checkpoint
models/best_b1.engine.json      engine manifest
scripts/export_tensorrt.py      B1-only exporter
scripts/benchmark_stream.py     release transport/performance acceptance
tests/                          protocol/runtime/server/tracker/client regression
```
