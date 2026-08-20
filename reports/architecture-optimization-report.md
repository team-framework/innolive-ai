# InnoLive AI Face Processor — 구조 진단 및 최적화 보고서

- **작성일**: 2026-08-18
- **대상**: `new-inno-live-server-python` (gRPC B1-640-Q90-W5 프로파일)
- **방법**: 코드 정독(약 3,500줄 핵심 경로) + 정적 검사 + 회귀 테스트 + **실제 모델(MPS/CPU) 부하 테스트**
- **테스트 자료**: 실제 얼굴 사진 3종(pravatar 400px)을 1920×1080 배경에 배치해 이동시키는 합성 영상 400프레임(프레임당 3~6개 얼굴)

---

## 1. 테스트 환경

| 항목 | 값 |
| --- | --- |
| CPU/GPU | Apple M4 (PyTorch MPS / CPU) |
| Python | 3.13.2 |
| 모델 | YOLO26n-seg (`best.pt`, 640 face-seg), AdaFace ViT-Base KP-RPE (WebFace12M), YuNet 2023mar |
| TensorRT | 엔진 파일 존재(Linux 대상) — macOS에서 미사용 |
| grpcio / opencv | 1.81.1 / 5.0.0 |

> **한계**: 배포 대상인 Linux NVIDIA GPU(TensorRT) 환경이 아니므로, 본 보고서의 절대 성능 수치는 MPS/CPU 기준이다. 상대적 비교와 병목 분석은 유효하다.

## 2. 검증 현황 (기존 품질)

| 항목 | 결과 |
| --- | --- |
| `ruff check` / `ruff format --check` | **전부 통과** (60 files) |
| `unittest discover -s tests` | **165 tests, 0.79s, 전부 통과** |
| 서버 시작(MPS) | YOLO warmup 포함 약 4s, AdaFace 로드 1.2s |

코드 품질·테스트 커버리지 측면에서는 건강한 상태다.

## 3. 성능测试结果

### 3.1 공식 게이트 (`scripts/benchmark_grpc.py`, 640p, 300프레임, W5)

| 백엔드 | 통과 | result FPS | server_total p50 / p95 / max | RTT p50 / p95 |
| --- | --- | --- | --- | --- |
| **MPS** (얼굴 3개/프레임) | ✅ **PASS** (게이트 8/8) | **45.4** | 20.4 / **29.7** / 330.5 ms | 105 / 135 ms |
| **CPU** (얼굴 6개/프레임) | ❌ FAIL (FPS·p95 게이트 미달) | **25.6** | 35.0 / **57.1** / 173 ms | 176 / 292 ms |

- 30 FPS·p95 33.3ms acceptance target은 **MPS급 가속기가 필요**하다. CPU(M4)에서는 약 26FPS로 30FPS에 미달(6개 얼굴 기준).
- MPS 단일 스트림은 여유 여유(45 FPS)로 target 대비 1.5배 헤드룸.
- p95 게이트 통과에도 **max 330ms** 스파이크가 존재(초기 프레임·GC 등) — 안정화 여지 있음.

### 3.2 해상도별 단계 지연 (MPS, 확장 harness)

| 시나리오 | 프레임 | server_total p50/p95 | decode | inference | tracking | blur_encode | serialize | queue |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 640p 단일 | 240 | 22.3 / 33.6 ms | 0.5 | 16.5 | 2.8 | 5.1 | 1.3 | 0 |
| **FHD 1080p 단일** | 200 | **70.5 / 100.3 ms** | 2.1 | 27.6 | 7.1 | **29.8** | 3.6 | 0 |

- FHD에서는 `blur_encode`(mask feathering + blur + JPEG Q90 인코딩)가 **프레임 비용의 42%** 차지. "detect small, blur big" 설계대로 FHD 원본 해상도에서 합성하는 것이 원인.
- RTT 관점: 640p p50 113ms, FHD p50 356ms — W5 윈도우 큐잉(5프레임 × 처리시간)이 지배적.

### 3.3 동시 스트림 (MPS, 640p, 얼굴 6개/프레임)

| 스트림 수 | 총 처리 | queue p50 / p95 | server_total p50 | RTT p50 |
| --- | --- | --- | --- | --- |
| 1 | 240f / 5.7s (42 fps) | 0 | 22.3 ms | 113 ms |
| 2 (120f×2) | 240f / 4.9s (49 fps) | **13.8 / 18.5** | 40.3 ms | 204 ms |
| 3 (90f×3) | 270f / 5.2s (52 fps) | **34.4 / 43.2** | 60.2 ms | 306 ms |

- 전역 단일 인퍼런스 레인 때문에 **큐 대기 시간이 동시 스트림 수에 선형 증가** (2배 → +14ms, 3배 → +34ms).
- 총 처리량은 소폭 상승하지만 **스트림별 RTT는 2.7배** 악화. 실시간 라이브用途에서 체감 지연이 급증한다.
- 현재 1.5s inference timeout은 이 부하에서 여유있지만, 스트림 수·프레임 비용이 커지면 큐 대기만으로 timeout이 발화될 수 있다(§4-A2).

### 3.4 Whitelist(인식) 플로우 (MPS)

| 측정 | 결과 |
| --- | --- |
| 등록(`AddWhitelist`, 400px 사진) | **178 ms** (YuNet + 정렬 + ViT-Base 임베딩) |
| 인식 메타데이터 오버헤드 (10k 프레임, `benchmark_recognition`) | **~0.02 ms/프레임** (무시 가능) |
| AdaFace 파이프라인 정확성 (동일 얼굴, 유사 스케일) | cosine **0.97** ✅ / 다른 인물 0.08 ✅ |
| **640p 스트림** (등록자 얼굴 ≈15px 폭) | **whitelist 0건** — `query_min_face_size=24px` 미만이라 AdaFace 미제출, fail-closed로 모자이크 |
| **FHD 스트림** (등록자 얼굴 ≈45×56px) | **whitelist 0.98/프레임** (1인), 타인 2.4/프레임 보호. server_total p50 66ms, **p95 237ms** (AdaFace 레인 + MPS 경합) |

**핵심 발견**:
1. 인식 계수·임베딩 파이프라인 자체는 정확하다(cosine 0.97).
2. **whitelist는 "스트림 내 얼굴 픽셀 크기"에 강하게 의존**한다. 640p 스트림에서 일반 시청자 크기의 얼굴(15~40px)은 24px floor 또는 YuNet 재탐지 한계로 인식 대상에서 빠진다. 즉, **640p에서는 whitelist가 사실상 유효하지 않을 수 있다**.
3. AdaFace 쿼리는 크롭 내에서 "얼굴 정확히 1개"를 요구한다(§4-A7).

### 3.5 등록(Enrollment) 품질 민감도

| 입력 | YuNet 결과 |
| --- | --- |
| 128px 로우퀄리티 얼굴 | **0개** (등록 불가, "expected exactly one face, found 0") |
| 1920×1080 전체 프레임 (200px 얼굴) | **0개** |
| 400px 얼굴 클로즈업 | 6장 중 4장 성공 (score 0.90~0.95), 2장은 다중/미검출 |

→ 등록 UX가 입력 품질에 취약하며, 실패 시 에러 메시지가 개선 방향을 안내하지 않는다.

---

## 4. 구조적 문제 (중요도 순)

### A1 [중] 전역 단일 인퍼런스 레인 — 스트림 간 head-of-line blocking
`service/runtime.py`의 `RuntimeManager.lane`(단일 `asyncio.Lock`)로 모든 스트림의 추론이 직렬화된다. 측정 결과 동시 스트림 3개에서 queue p95 43ms, RTT 2.7배 악화.
- 단일 레인은 결정적 순서(스트림 내 순차성)를 보장하기 위한 선택이지만, **스트림 간**에도 직렬화된다.
- **개선**: 스트림 내 직렬 + 스트림 간 병렬(예: N=2~4 레인 풀, 스트림별 순서 유지) 또는 per-stream 전용 추론. 배포 GPU에서는 TensorRT engine 병렬 컨텍스트로 대응.

### A2 [높음] inference timeout이 큐 대기를 포함 → 서비스 자기 불능화
`ai_processor_server.py`의 `_run_inference`는 `wait_for(shield(task), 1.5s)`로 **레인 대기 포함 전체**를 측정한다. 다중 스트림/서지 부하에서 큐 대기만 1.5s를 초과하면:
1. 해당 스트림 `DEADLINE_EXCEEDED` 종료,
2. **`_mark_runtime_unhealthy()` → 서비스 전체 NOT_SERVING** (건강한 모델도 서비스 정지).
- 한 번 unhealthy latch되면 **자동 복구 경로가 없다**(§A15).
- **개선**: (a) 레인 대기 시간과 인퍼런스 실행 시간을 분리 측정해 실행 시간만 timeout에 사용, (b) unhealthy 전환 전에 경보·계측 노출, (c) 복구 프로브.

### A3 [중] 동시 스트림 수 무제한
`active_streams`는 증가만 하고 한도가 없다. 각 스트림은 BOTSORT 트래커 + 인식 상태 + W5×4MB 버퍼를 보유. 악성/실수 클라이언트가 수십 스트림을 열면 메모리·레인 점유로 전체 성능이 악화된다.
- **개선**: 최대 동시 스트림(예 16) + owner(프로토 레벨)별 한도, 초과 시 `RESOURCE_EXHAUSTED`.

### A4 [중] 세션이 영구 잔존 (leak)
`ProcessVideo`는 세션이 없으면 **자동 생성**(`acquire_stream`)하고, TTL/LRU가 없다. 임시 세션 id를 쓰는 클라이언트가 1,024개를 채우면 `RESOURCE_EXHAUSTED`로 신규 스트림 불가.
- **개선**: idle TTL(예 30분) + LRU, `ListSessions`에 last_active 노출.

### A5 [높음·기능] whitelist의 실전 유효 범위: 얼굴 픽셀 크기
측정(§3.4): 640p 스트림에서 일반 크기의 얼굴은 `query_min_face_size=24px`/YuNet 한계로 인식 대상에서 제외 → **등록 인물도 모자이크**. FHD에서는 정상(0.98/프레임).
- 이는 "안전 우선"이라 옳은 방향이지만, 운영상 "whitelist가 동작하지 않는다"는 불만 원인이 된다.
- **개선**: (a) 서비스 문서·메타데이터에 `skipped_too_small` 같은 명시적 상태 추가(현재는 모자이크로만 관찰 가능), (b) `query_min_face_size`를 테스트 기반으로 재조정(24px 이하 skip은 유지하되, 24~40px 구간 신뢰도 측정), (c) 쿼리 크롭의 YuNet 실패 시 YOLO 박스 기반 정렬 fallback(§A7).

### A6 [중] AdaFace 쿼리의 "정확히 1개 얼굴" 엄격성
`_detected_face`는 크롭 내 얼굴이 정확히 1개가 아니면 `FaceCountError`(fail-closed). 실측: 400px 사진에서 2번째 얼굴(0.82)만 추가되어도 **전체 사진 쿼리가 전부 실패**, 200px 전체 사진은 2개 검출로 실패.
- YOLO가 이미 얼굴 박스를 주고 있는데, 크롭 내 인접·배경의 0.6 이상 후보 1개에 의해 인식이 사라진다.
- **개선**: "exactly 1"을 **YOLO 박스와 IoU가 최대인 얼굴 선택**으로 완화(0개일 때만 실패). 등록 경로의 0.9 단일화 유지.

### A7 [낮음] FHD 모자이크 합성 비용
`service/mosaic.py`의 `_feathered_mask`는 매 프레임 **전체 해상도**에서 dilate + `distanceTransform` 수행. FHD에서 blur_encode p50 29.8ms(프레임 비용 42%).
- **개선**: mask 바운딩박스(패딩 포함)로 제한. 예상 절감 FHD 10~15ms/프레임.

### A8 [낮음] 설정 불일치: `MOSAIC_MAX_INFLIGHT=2` vs 1 워커
`BoundedSemaphore(2)`지만 모자이크 `ThreadPoolExecutor`는 1 워커 → 실질 병렬 1. 의도(버퍼링)라면 문서화, 병렬화 의도라면 워커 2.

### A9 [낮음] 관측성 부족
`runtime.health()` / `adaface.health()`(calls, overflow, queue wait, failures)는 **프로세스 내부 전용** — gRPC/HTTP로 노출되지 않음. §A2의 queue/inference 분기 계측, SLO 모니터링을 위해 `/metrics`(Prometheus) 또는 RPC 필요.

### A10 [낮음] 기타
- **SIGTERM 처리 없음**: `asyncio.run`만 `KeyboardInterrupt`를 잡으므로 SIGTERM은 graceful shutdown(`stop(grace)`)을 거치지 않고 즉시 종료.
- **`recognition.process`가 이벤트 루프에서 동기 실행**: 매 프레임 crop 복사(`_face_crop`)·스케줄링이 루프 스레드에서 발생. FHD·다중 스트림에서 지터 원인.
- **unhealthy latch 후 복구 없음**: 일시적 MPS/CUDA 오류로 서비스 전체가 NOT_SERVING로 영구 전환(§A2와 복합).
- **인증/암호화 부재**: README가 인정(내부망/프록시 의존) — 프로토 계약상 유지하되 운영 체크리스트에 명시 권장.
- **AdaFace는 PyTorch 전용**: YOLO만 TensorRT 가속. Linux 배포에서 AdaFace ViT-Base의 CPU 부하가 GIL 경합 요인(FHD p95 237ms 관측). ONNX Runtime/TensorRT export 검토.

---

## 5. 최적화 로드맵

### 단기간 (1~2주, 리스크 낮음)
| # | 항목 | 기대 효과 | 근거 |
| --- | --- | --- | --- |
| 1 | timeout 분기: 레인 대기 ≠ 인퍼런스 실행 (A2) | 부하 시 자기 불능화 제거 | §3.3, §A2 |
| 2 | unhealthy 복구 프로브 + 경보 (A15/A2) | 단일 오류로 영구 중단 방지 | 코드 분석 |
| 3 | 모자이크 feathering을 mask bbox로 제한 (A7) | FHD -10~15ms/프레임 (p95 100→~85ms 추정) | §3.2 |
| 4 | 세션 TTL/LRU (A4) | registry exhaustion 방지 | 코드 분석 |
| 5 | 스트림 수 한도 + backpressure (A3) | DoS/실수 완화 | §3.3 |
| 6 | 등록 실패 메시지 개선("얼굴이 너무 작습니다/클로즈업 제공") + 등록 이미지 자동 크롭 옵션 (A8/§3.5) | UX | §3.5 |
| 7 | SIGTERM graceful shutdown (A10) | 배포(커스텀 컨테이너) 안정성 | 코드 분석 |

### 중기 (1개월)
| # | 항목 | 기대 효과 | 근거 |
| --- | --- | --- | --- |
| 8 | 인퍼런스 레인 풀(스트림 내 직렬 유지, N=2~4) (A1) | 3스트림 RTT 306→~180ms 목표 | §3.3 |
| 9 | AdaFace 쿼리 IoU 기반 얼굴 선택으로 완화 (A6) | 인접 후보로 인한 인식 소실 제거 | §3.4 |
| 10 | `/metrics` (queue_ms, inference_ms, adaface calls/overflow, active_streams) (A9) | SLO·캐패시티 플랜ニング | §3 |
| 11 | `recognition.process`를 워크러로 이동 (A10) | 이벤트 루프 지터 감소 | 코드 분석 |

### 장기 (분기)
| # | 항목 | 기대 효과 | 근거 |
| --- | --- | --- | --- |
| 12 | AdaFace ONNX/TensorRT (Linux) | GIL 경합 제거, FHD p95 237ms 개선 | §3.4 |
| 13 | 쿼리 최소 얼굴 크기 재보정 + `skipped_too_small` 메타데이터 (A5) | whitelist 실전 유효 범위 명확화 | §3.4 |
| 14 | YOLO raw-forward (ultralytics predict 오버헤드 제거) 또는 ONNX export 표준화 | inference 27.6ms(MPS) 추가 절감 여지 | §3.2 |
| 15 | 멀티-GPU/멀티인스턴스 분산(스트림 라우팅) | 수직 확장 한계 초과 시 | §3.3 |

---

## 6. 결론

1. **핵심 파이프라인(탐지→추적→합성→전송)은 설계대로 동작**하며, 공식 acceptance gate(MPS)를 통과한다(45.4 FPS, p95 29.7ms). 코드 품질·테스트(165건)·검증 계층(유한 바운더리, fail-closed)은 견고하다.
2. **가장 실질적인 운영 리스크**는 성능이 아니라 (a) 부하 시 **timeout→자기 불능화** 경로(A2), (b) **스트림 간 직렬화에 따른 RTT 악화**(A1), (c) **whitelist의 얼굴 크기 의존성**(A5)이다.
3. **whitelist 기능은 "동작한다"고 단정하기 어렵다**: FHD에서는 검증되었으나(0.98/프레임), 640p에서는 일반 크기의 얼굴이 인식 대상에서 제외되어 등록 인물까지 모자이크된다. 배포 목표 해상도 기준으로 유효성 재검증이 필요하다.
4. **CPU-only 배포는 30FPS 게이트에 미달**(25.6 FPS, p95 57ms) — 30FPS SLO를 유지하려면 가속기(MPS/NVIDIA+TensorRT)를 사실상 필수로 문서화할 것을 권고한다.

---

## 부록: 재현 방법

```bash
# 테스트 영상 생성 (실제 얼굴 사진 3종 + 이동 합성, 400프레임 1080p)
#   /tmp/innolive-bench/gen_video.py, harness.py, probe_adaface.py, probe_match.py

# 서버
.venv/bin/python ai_processor_server.py --port 50061                # auto → MPS
.venv/bin/python ai_processor_server.py --backend pytorch --device cpu --port 50062

# 공식 게이트
.venv/bin/python -m scripts.benchmark_grpc --target 127.0.0.1:50061 \
  --session-id bench --input bench_1080.mp4 --frames 300

# 원시 결과
#   /tmp/innolive-bench/bench_640_mps.json   (MPS, 3-face video, 공식 게이트 통과)
#   /tmp/innolive-bench/bench_640_cpu.json   (CPU, 6-face video, 게이트 미달)
#   /tmp/innolive-bench/harness_results.json (FHD / 1·2·3 스트림 / whitelist 플로우)
```

*측정 조건: Apple M4, Python 3.13.2, 단일 서버 프로세스, loopback. 절대 수치는 환경 의존이며, 보고서의 목적은 병목 위치와 상대적 영향이다.*
