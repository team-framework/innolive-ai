# TensorRT Face Swap Lab (프로덕션 연동 가이드)

독립 테스트 클라이언트이자 프로덕션 face-swap 파이프라인의 기준 구현이다.
YOLO26n-seg 검출 → 얼굴 합성 → blur fallback까지 한 프로세스에서 돌고,
WebRTC/WebSocket/file(HLS) 출력으로 서빙한다.

```bash
python -m pip install -r requirements-trt-swap-client.txt
```

```bash
python -m experiments.trt_swap_client.export_detector \
  --checkpoint models/best.pt --output models/best_swap_b4.engine \
  --max-batch 4 --workspace 8 --device 0

python -m experiments.trt_swap_client.build_mixed_onnx \
  --onnx models/face_swap/inswapper_128.onnx \
  --output models/face_swap/inswapper_128_mixed19.onnx --force

python -m experiments.trt_swap_client.export_swapper \
  --onnx models/face_swap/inswapper_128_mixed19.onnx \
  --output models/face_swap/inswapper_128_trt11_mixed.engine \
  --mixed-base models/face_swap/inswapper_128.onnx --force
```

```bash
python -m experiments.trt_swap_client.app --host 0.0.0.0 --port 8088 \
  --source ~/Documents/input.png
```

`--swapper-engine`을 생략하면 빌드된 mixed 엔진을 자동 선택하고,
없으면 stock FP32 엔진으로 떨어진다. 명시한 경로는 그대로 쓴다.
`./run.sh [포트]` / `PORT=... ./run.sh`도 같다.
브라우저는 `http://SERVER_IP:8088` (사설망에서만 실행).

## 파이프라인과 동시성 (프로덕션 연동 시 그대로 쓰는 계약)

```
 camera/file ─▶ detector batch ─▶ compose ─▶ WebRTC/WebSocket/HLS
   (YOLO TRT,      (스트림별 격리, 스레드풀 병렬)
    짧은 batch wait)
```

- **진입점**: `Settings` + `SwapLab` (`experiments/trt_swap_client/app.py`).
  `create_stream(session_id)` → `submit(frame, stream)` → `(output, meta)`.
- **얼굴 1개**: 기존 순차 경로 그대로 (prepare → forward → paste).
  다얼굴만 워커풀 파이프라인을 탄다.
- **얼굴 N개**: prepare/forward/render를 워커풀(4 스레드)로 겹친다.
  forward는 공유 TRT 컨텍스트 락으로 직렬화하고, 머지는 인덱스 순서로
  기존 blend 수식 그대로라서 순차 실행과 비트 동일하다.
- **세션 N개**: detector 배치는 유지하고, compose는 스트림별로 그룹핑해서
  그룹 간만 병렬로 돌린다. **같은 스트림은 항상 순차**라서 프레임 순서와
  tracker/recognition 상태가 깨지지 않는다. 단일 그룹이면 풀을 안 탄다.
- **스레드 안전**: TRT context·재사용 버퍼는 forward 락,
  YuNet은 스레드별 클론 4개, 통계 카운터는 락, 레지스트리는 기존 락.
  timing은 스레드별 TLS로 귀속돼서 남의 job 수치를 읽지 않는다.
- **최신 프레임 우선**: 큐가 차면 같은 세션의 대기 프레임을 교체하고,
  없으면 가장 오래된 것을 evict한다. p95 꼬리를 만들지 않는다.
- **종료**: `await lab.close()`가 worker·AdaFace·swapper 풀을 순서대로 닫는다.
- **메타**: `swap_ms`, `swap_prepare/forward/paste_ms`, `swap_batch_size`,
  `swap_faces`, `fallback_blurs`, `render_frames`/`swap_completed_frames`
  (렌더 FPS와 합성 완료 FPS 분리), `yolo_batch`, `detector_batch_ms`.
  `/health`에 p50/p95·드롭·마지막 단계 시간이 있다.

## 엔진 (재현 절차 포함)

- detector: `best_swap_b4.engine` (dynamic B1–B4 FP16). `best.pt` 변경 시 rebuild.
- swapper stock: `inswapper_128_trt11_fp32.engine` (static B1 FP32 direct).
- swapper mixed: `inswapper_128_trt11_mixed.engine` (무거운 Conv 19개만 FP16,
  forward 12.9→5.6ms, 실얼굴 MAE 1.4e-3·육안 동등).
- 시작 시 manifest 게이트가 binding 이름·shape·dtype·모델 해시를 검증한다.
  provenance 없는 엔진(구 FP16 포함)은 그대로 거부한다. mixed는
  `base_model_sha256` + `mixed_recipe`가 있어야 통과한다.
- 엔진은 Git에 넣지 않고 호스트에서 빌드한다. **다른 GPU(5070Ti 등)로
  옮기면 전량 rebuild + 아래 품질 게이트를 다시 돌린다.**
  Blackwell(sm_120)은 TRT/CUDA 버전 요구가 다르다.

## 성능 (3090 실측, 720p 합성 프레임)

| config | 베이스 | 최종 | 비고 |
| --- | --- | --- | --- |
| 1얼굴 x 1세션 | 23.4fps | **38.4fps** | 30fps선 돌파 |
| 2얼굴 x 1세션 | 14.0fps | **28.5fps** | 파이프라인 +104% |
| 4얼굴 x 1세션 | 9.9fps | **24.6fps** | 파이프라인 +148% |
| 1얼굴 x 10세션 합계 | 25.8/s | **69.4/s** | compose 병렬 +169% |
| 4얼굴 x 4세션 | 41sw/s | **123sw/s** | +200% |

- 단계 비용(1얼굴): detector 3 / yunet 10 / forward 5.6 / paste 5.
- 1얼굴 프레임은 베이스와 비트 동일 출력이다.
- YuNet landmark 인덱스 수정(#25 병합) 후 paste ROI가 정상화되면서
  뭉개짐이 사라지고 paste가 절반으로 줄었다.
- 측정 하네스: `python -m experiments.trt_swap_client.bench_lab --help`
  (base/opt 공용, `--faces/--sessions/--swapper-engine` 지정).
- 시도 후 기각: 동적 배치(정확하나 속도 선형), paste 스레드 warp(역효과),
  전체 FP16(오차), 빌드 레벨 튜닝(무이득).

## 품질 게이트 (엔진·코드 변경 시 매번)

1. 1얼굴 프레임 베이스 대비 비트 동등 (paste 수식 변경 시).
2. YOLO mask 내외 원본 보존·경계 artifact 육안 + 작은 얼굴 blur fallback.
3. 엔진 교체 시 실 얼굴 blob 기준 ORT/FP32 대비 + 디코딩 PNG 비교.
4. `pytest tests/test_trt_swap_client.py` 전체 통과.

## 동작 계약 (변경 금지 없이 유지)

- YOLO26n-seg class `0=face`, `1=number_plate`가 아니면 시작 거부.
- InSwapper CUDA arena 세션당 2GiB 상한, age/gender 세션 미생성.
- `--target-aligner yunet_roi` 기본, 실패 시 InsightFace fallback
  (`insightface`는 legacy 비교 경로).
- AdaFace whitelist 얼굴은 원본 유지, 작은 mask·hold·실패는 blur fallback.
  면적은 YOLO mask 기준이며, mask가 깨지면 면적 0으로 blur한다
  (box 사각형은 blur 영역으로만 쓴다).
- WebRTC 우선, 불가 시 WebSocket fallback. TURN 환경변수 지원.

```bash
export WEBRTC_TURN_URL='turn:turn.example.com:3478?transport=udp'
export WEBRTC_TURN_USERNAME='...'
export WEBRTC_TURN_CREDENTIAL='...'
```

## 얼굴 품질 진단·파일 테스트

```bash
python -m experiments.trt_swap_client.app \
  --swapper-backend tensorrt \
  --swap-debug-dir /tmp/inswapper-debug --swap-debug-frames 1 \
  --stream-jpeg-quality 100
```

`05_onnx_raw_swap.png` vs `06_trt_raw_swap.png` 먼저 비교, `debug.json`에
MAE·RMSE·MAX 기록. 덤프 실패는 live swap에 영향 없다.
공식 ONNX는 `input_mean=0`, `input_std=255`라 target 범위는 `[0,1]`이다.

```bash
python -m experiments.trt_swap_client.app --host 0.0.0.0 --port 8088 \
  --input-video /data/input.mp4 --hls-dir /tmp/face-swap-hls
```

NVDEC decode → NVENC HLS(`/hls/live.m3u8`). BGR 변환에 device↔host 복사가
한 번씩 있어 zero-copy가 아니다.
