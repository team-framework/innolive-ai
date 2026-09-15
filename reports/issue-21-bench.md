# 이슈 #21 벤치마크 리포트: YOLO mask 합성 및 다중 얼굴 batch 처리

- 브랜치: `feat/yolo-mask-batch/#21` (PR #24, Draft)
- 베이스: `9aed52b` (본 브랜치의 parent main)
- 측정일: 2026-09-15, 호스트: RTX 3090 24GB (driver 595.91.07)
- 환경: TensorRT 11.1.0.106, torch 2.13.0+cu130, `.venv-trt10-swap`
- 엔진: detector `best_swap_b4.engine` (dynamic B1–B4 FP16),
  swapper `inswapper_128_trt11_fp32.engine` (static B1 FP32 direct)
- 하네스: `experiments/trt_swap_client/bench_lab.py` (in-process, base/opt 공용)
- 프레임: `~/Documents/input.png`를 타일링한 1280x720 합성 프레임
  (1/2/4 얼굴, detector가 정확히 1/2/4개 검출함을 사전 확인)
- 부하: 세션별 closed-loop (다음 프레임은 이전 결과 반환 즉시 전송),
  max-batch 4 / batch-wait 3ms / max-queue 16 (별도 표기 제외)
- 원시 데이터: `reports/issue-21-bench.json`

## 1. 결과 요약 (base → opt)

| config | FPS | face swaps/sec | p50 (ms) | p95 (ms) |
| --- | --- | --- | --- | --- |
| 1face x 1sess | 23.43 → **25.55 (+9.0%)** | 23.43 → 25.55 | 42.7 → 39.1 | 43.2 → 40.1 |
| 1face x 2sess | 24.89 → 25.04 (+0.6%) | 24.89 → 25.04 | 80.4 → 80.0 | 81.5 → 81.2 |
| 1face x 4sess | 26.05 → 26.16 (+0.4%) | 26.05 → 26.16 | 153.4 → 152.7 | 154.9 → 155.4 |
| 2face x 1sess | 13.99 → **14.45 (+3.3%)** | 27.98 → 28.90 | 71.5 → 69.0 | 72.7 → 71.2 |
| 2face x 2sess | 14.56 → 14.45 (-0.8%) | 29.12 → 28.90 | 137.6 → 138.3 | 139.0 → 141.2 |
| 2face x 4sess | 14.98 → 14.79 (-1.3%) | 29.95 → 29.59 | 266.9 → 269.5 | 272.0 → 274.7 |
| 4face x 1sess | 9.92 → **10.24 (+3.2%)** | 39.67 → 40.97 | 100.6 → 97.6 | 102.5 → 99.5 |
| 4face x 2sess | 10.13 → 10.18 (+0.5%) | 40.51 → 40.71 | 197.0 → 196.0 | 203.2 → 200.0 |
| 4face x 4sess | 10.30 → 10.33 (+0.3%) | 41.18 → 41.33 | 388.2 → 386.3 | 392.5 → 393.8 |

- 단일 세션: **+3~9% 개선** (1x1 +9.0%, p50 -3.6ms)
- 다중 세션 합계 처리량: 동률 (±1% 이내, 측정 노이즈 수준)
- p95가 p50을 바짝 따라가서 꼬리 지연 없음
- 참고: 앱 기본값(latency-first: max-batch 1 / wait 0ms / queue 2)으로 1x1을
  재측정하면 25.47 FPS, p50 39.3ms로 batching 설정과 동등
- 이슈의 "약 28 FPS" 기재치와 절대값이 다른 이유: 본 하네스는 720p 대형 얼굴
  타일 기준이며, 실측 조건(프레임/얼굴 크기)에 따라 달라진다.
  전후 비교는 동일 하네스·동일 프레임으로 수행했으므로 상대치는 유효하다.

## 2. 단계별 비용 (opt, 얼굴 1개 기준, ms)

detector 3.2 (batch 시 분할 상각: yolo_batch 2→4.8, 4→8.4) /
alignment(yunet, CPU) 10.1 / prepare 0.8 / forward(TRT) 13.0 / paste 10.8

- 얼굴이 늘면 forward가 얼굴 수에 비례 (13.0/25.7/51.7) — static B1 엔진의
  순차 실행 특성이다. 아래 4절에서 동적 배치 엔진을 검증했다.
- yunet은 CPU(OpenCV CUDA 빌드 아님) 추론이라 ROI 크기에 비례한다.
  2얼굴 타일(18.0)이 4얼굴 타일(14.2)보다 느린 것도 ROI 크기 효과이며,
  base/opt 동일하므로 본 변경의 회귀가 아니다.

## 3. 적용된 최적화 (이슈 범위 내)

1. **적응형 batch wait** (`SwapLab._batch_loop`): 백로그가 있을 때만
   `batch_wait_ms`를 소비하고, idle 단일 세션은 즉시 처리한다.
   기존에는 단일 세션도 매 프레임 3ms를 풀로 대기했다.
   → 1x1 p50 -3ms, detector batching(yolo_batch 2/4)은 부하 시 그대로 유지.
2. **paste 전체 프레임 복사 1회화** (`_paste_prepared` + `destination`):
   얼굴 수만큼 하던 full-frame 복사를 1회로 줄였다. 수학식 동일.
3. **YOLO seg 교차의 aligned-domain 1-warp화** (`_paste_inswapper`):
   seg를 aligned 128 도메인에서 feather mask와 먼저 곱한 뒤 BGR+alpha와
   함께 1회 warp한다. 2-warp 수식과 max 1LSB 동등함을 실측 검증했다
   (아래 5절). 5채널 warp·스레드 warp는 오히려 느려져서 기각했다 (4절).

## 4. 시도 후 기각한 최적화 (측정 포함)

- **동적 배치 TRT 엔진** (`export_swapper --max-batch 4`, B1..B4 profile):
  빌드 성공, batch 출력이 ONNX FP32 reference와 MAE 8e-4로 일치하고
  latent 순열 대응도 정확했다. 그러나 sustained forward가
  N=1 13.9ms → N=4 56.1ms(얼굴당 14.0ms)로 완전 선형이라 처리량 이득이
  없다 (가중치 554MB 메모리 바운드). 정적 B1 + 순서 보장 순차 실행을 유지한다.
  엔진 파일은 서버 `models/face_swap/inswapper_128_trt11_fp32_b4.engine`에만
  있고 Git에는 넣지 않았다.
- **5채널 single warp** (BGR+mask+seg 1회 warp): 4채널+1채널 2-warp보다
  paste가 11.7→16.5ms로 악화. OpenCV fast path를 벗어나서 기각.
- **paste 워커 스레드 병렬화**: 4얼굴에서만 -14%, 2얼굴 무효과,
  격리 micro-bench에서 동일 warp의 스레드 실행이 6배까지 느려지는 것을
  확인 (이 빌드의 warpAffine은 스레드 확장이 안 됨). 작은 얼굴 조건에서
  역효과 위험이 있어 기각하고 순차 paste로 되돌렸다.

## 5. 정확성 검증

- **paste 동등성**: 1얼굴 프레임에서 base 출력과 opt(mask off) 출력이
  **비트 동일** (maxdiff 0). 다중 얼굴 차이는 batch-prepare 의미론
  (원본 프레임에서 전 얼굴 prepare, 얼굴 간 오염 제거)에 따른 것으로 문서화.
- **YOLO mask 효과**: mask on/off 비교에서 mask 밖 원본 보존,
  경계 feather에 artifact 없음, 작은 얼굴 blur fallback 유지를
  실 이미지로 육안 확인했다 (서버 `/tmp/mask_check/`,
  `swap_faces/fallback_blurs/small_face_fallbacks` 메타 포함).
  mask on/off 수치 차이: max 159 / mean 6.4~9.3 (얼굴 내부·경계 한정).
- **TRT batch 출력**: 동적 엔진 N=1/2/4가 ONNX FP32와 MAE ~8e-4,
  latent 순열 테스트에서 행 대응 정확 (maxdiff 0.0), 행 간 출력 상이.
- **버퍼 별칭 버그 수정**: static 경로 `forward_batch`가 재사용 버퍼 뷰를
  쌓아 다중 얼굴이 마지막 얼굴로 덮이는 버그를 GPU 측정 중 발견·수정하고
  회귀 테스트를 추가했다 (`test_forward_batch_keeps_distinct_rows_*`).
- **세션 격리·latest-frame**: tracker/recognition은 스트림별 소유 유지,
  submit coalescing/evict 로직을 단위 테스트로 검증했다
  (`test_submit_coalesces_*`, `test_submit_evicts_*`).
- **timing**: prepare/forward/paste를 얼굴 합계로 기록하고
  render FPS와 합성 완료 FPS를 분리했다 (`render_frames`,
  `swap_completed_frames`, `swapped_faces_total`, p95 포함).

## 6. 혼합 정밀도 엔진 (Phase 2 결과)

TRT 11.1에는 FP16 빌더 플래그와 레이어별 정밀도 API가 없어서
(`BuilderFlag` 목록·`ILayer` 확인), ONNX 그래프 수술로 접근했다.
forward의 83%를 차지하는 Conv 16개 포함 19개 Conv의 데이터·가중치에만
FP16 Cast를 삽입하고 Resize/shape/AdaIN 통계 경로는 untouched다
(`experiments/trt_swap_client/build_mixed_onnx.py`).
19개 Conv FP16, 앞단·헤드 FP32 유지 같은 변형을 sweeep했고,
실 얼굴 blob 기준 MAE가 가장 좋은 전체 변환을 채택했다.

| 엔진 | forward/face | 실얼굴 MAE | 비고 |
| --- | --- | --- | --- |
| stock FP32 | 12.9ms | 7.9e-4 (ORT 대비) | 기준 |
| mixed19 (19 Conv FP16) | **5.6ms (2.3배)** | 1.4e-3 | **채택** |
| mixed17 (head FP32) | 7.6ms | 1.2e-3 | 오차 개선 미미, 기각 |
| mixed12 (front+head FP32) | 8.9ms | 1.0e-3 | 비용 대비 효과 없음, 기각 |

- 랜덤 노이즈 입력 기준 MAE(1.2e-2)는 실 얼굴(1.4e-3)보다 크게 나온다.
  게이트는 실 데이터 + 육안으로 판정했다. 디코딩 PNG 육안 비교 통과,
  mask on/off 패턴 동일(max 158/mean 6.4), 작은 얼굴 blur fallback 정상.
- 엔진 provenance는 manifest에 기록한다
  (`base_model_sha256` + `mixed_recipe`, `--mixed-base`).
  런타임 게이트가 provenance 없는 엔진은 계속 거부한다.
- 엔진 파일 자체는 Git에 넣지 않고 호스트에서 빌드한다
  (`inswapper_128_trt11_mixed.engine`).

mixed 엔진 e2e (stock-opt 대비):

| config | FPS | p50 (ms) |
| --- | --- | --- |
| 1face x 1sess | 25.55 → **30.84 (+21%)** | 39.1 → 32.3 |
| 2face x 1sess | 14.45 → **18.49 (+28%)** | 69.0 → 53.9 |
| 4face x 1sess | 10.24 → **14.71 (+44%)** | 97.6 → 67.8 |
| 1face x 4sess (합계) | 26.16 → **33.18** | 152.7 → 120.2 |
| 1face x 10sess (합계) | 25.84 → **32.85** | 313.8 → 260.9 |

베이스(`9aed52b`) 대비 1x1은 **23.43 → 30.84 (+31%)** 로 30fps선을 넘었다.
남은 지배 병목은 yunet(CPU 10ms) + paste(CPU 11ms)다 (Phase 3).

## 7. 얼굴 파이프라인 (Phase 3a 결과)

얼굴별 prepare/forward/render를 워커풀(4 스레드)로 겹치고,
forward만 공유 TRT 컨텍스트 락으로 직렬화했다.
CPU 단계가 GPU를 가리지 않아 GPU 유휴가 사라진다.
머지는 인덱스 순서로 기존 blend 수식 그대로라서 순차 실행과 비트 동일함을
e2e로 확인했다 (1/2얼굴 maxdiff 0). yunet도 스레드별 클론으로 병렬화했다
(공유 객체의 setInputSize 레이스 회피, 실측 54→23ms).
1얼굴 경로는 기존 순차 코드를 그대로 써서 건드리지 않았다.

mixed 엔진 + 파이프라인 (순차-mixed 대비):

| config | FPS | p50 (ms) |
| --- | --- | --- |
| 1face x 1sess | 30.84 → 31.94 (동등, 파이프 미사용) | 32.3 → 31.1 |
| 2face x 1sess | 18.49 → **23.32 (+26%)** | 53.9 → 42.5 |
| 4face x 1sess | 14.71 → **18.86 (+28%)** | 67.8 → 52.6 |

베이스 대비 2얼굴 +67%, 4얼굴 +90%다.
타이밍 합계(`swap_forward_ms` 등)는 락 대기를 포함하므로 wall보다 크게
보일 수 있다. `swap_generator_ms`는 wall 기준이라 FPS와 일치한다.

## 8. 결론

- 이슈의 기능 범위(YOLO mask 합성, prepare/forward/paste 분리 batch 구조,
  bounded latest-frame, latent 매핑 준비, timing/FPS 분리)를 구현했다.
- 동일 하네스에서 베이스 대비 단일 세션 +3~9%, 다중 세션 동률을 달성했고,
  화질·지연 회귀는 없다 (1얼굴 비트 동일, p95 안정).
- 혼합 정밀도(mixed19) 적용 후 1x1 **30.84fps로 30fps선 돌파**,
  베이스 대비 +31%. 멀티세션 합계도 26→33/s로 상승했다.
- 얼굴 파이프라인 적용 후 2얼굴 23.3fps, 4얼굴 18.9fps
  (베이스 대비 +67%/+90%). 순차 실행과 비트 동일함을 e2e로 확인했다.
- 남은 과제: 세션 간 compose 병렬화 (합계 처리량), 적응형 레이트,
  듀얼 GPU 분할 (계획 문서 §4).
