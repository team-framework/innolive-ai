# RF-DETR 전환 추가 검증 — 2026-11-09

PR #103의 API·안정성 검증에서는 회귀를 발견하지 않았다. 다만 제공된 RF-DETR 체크포인트는
기존 모델보다 번호판 검출·분할과 얼굴 마스크 재현율이 낮았다. 따라서 **품질이 그대로
유지된다는 조건은 충족하지 못했다.** 아래 측정은 해당 표본과 운영 설정에 한정한다.

## 정답 라벨 기반 품질 비교

같은 원본 이미지와 라벨을 기존 YOLO TensorRT, RF-DETR PyTorch FP32, RF-DETR TensorRT
FP16으로 각각 평가했다. 운영 detector의 confidence 0.01·최대 100검출을 사용하고,
P/R은 confidence 0.25·IoU 0.5에서 계산했다. BoT-SORT와 mask hold 이전의 detector 출력이다.

Box와 최종 detector polygon을 공통 624×624 좌표로 변환해 COCOeval로 평가했다.
Mask는 해당 좌표에서 rasterize했다. 각 데이터셋에서 실제 라벨이 있는 class만 평가했다.
기존 모델의 입력은 기존 640 letterbox, RF-DETR은 학습 설정인 624 square resize다.
이 측정은 같은 입력 해상도의 모델 순위가 아니라 **교체 전후 운영 detector 경로의 비교**다.
과거 별도 평가의 threshold·raster·출력 방식이 달라 기존 보고서 mAP와 직접 비교하지 않는다.

| 표본 | 경로 | Box mAP50 | Box mAP50–95 | Mask mAP50 | Mask mAP50–95 | Box recall | Mask recall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 얼굴 416프레임 / 3,382개 | 기존 YOLO TRT | 60.86% | 24.13% | 57.22% | 21.33% | 57.42% | 55.85% |
| 얼굴 | RF-DETR FP32 | 66.03% | 27.11% | 57.19% | 19.18% | 54.58% | 50.44% |
| 얼굴 | RF-DETR TRT FP16 | **65.87%** | **27.22%** | 57.20% | 19.18% | **54.61%** | **50.50%** |
| 번호판 220장 / 520개 | 기존 YOLO TRT | 93.42% | 58.62% | 84.58% | 40.15% | 92.88% | 85.96% |
| 번호판 | RF-DETR FP32 | 85.98% | 46.77% | 75.81% | 33.26% | 85.58% | 78.08% |
| 번호판 | RF-DETR TRT FP16 | **86.31%** | **46.77%** | **75.89%** | **33.25%** | **85.96%** | **78.27%** |

- 기존 대비 RF-DETR의 얼굴 Box mAP50은 +5.01%p지만 Mask recall은 −5.35%p다.
- 번호판 Box mAP50은 −7.11%p, Mask mAP50은 −8.69%p, Box recall은 −6.92%p다.
- RF-DETR FP32→FP16의 두 표본 mAP50 차이는 최대 0.33%p 미만이다. 이번 품질 차이의 주원인은 FP16 변환보다 제공 체크포인트와 운영 detector 경로의 차이다.
- FP16 얼굴 Box/Mask precision은 94.96%/87.81%, 번호판은 76.02%/69.22%다. 단순 threshold 완화가 모든 지표를 회복한다고 볼 수 없다.

얼굴 표본은 세로 1080×1920의 약 13.9초 단일 영상이며, 연속 프레임이므로 416개의 독립
장면으로 해석하면 안 된다. 얼굴 라벨은 머리카락·모자를 제외한다. 번호판 표본은 기존 서버의
CAVI test 평가 뷰다. 두 데이터와 이번 체크포인트의 학습 데이터 중복 여부는 확인하지 못했다.
새로운 카메라·각도·야간·가림 조건의 일반화나 모든 개인정보 보호 성공률은 보장하지 않는다.

## API·호환성

실제 GPU의 기존 YOLO 서버와 RF-DETR 서버에 같은 요청을 전송했다.
[`verify_live_contract.py`](../scripts/verify_live_contract.py)의 24개 검증 그룹과
10개 오류 응답이 양쪽에서 통과했고 오류 code/details가 일치했다.

- JPEG/raw YUV420 × blur/face swap/face metadata × 기본/metadata-only/영상 출력의 18개 조합, 각 3프레임 확인
- frame ID·timestamp 그대로 반환, 중복 frame ID 허용, 영상 크기·픽셀 형식·polygon 제한·B1 timing 계약 확인
- Python client window 1/2/5에서 각 20프레임 순서·전달·inflight 상한 확인
- 손상 JPEG·잘못된 batch/mosaic 설정 이후 동일 stream 복구, 잘못된 stream enum·session 거부 확인
- 실제 AdaFace 등록·export/import·삭제, 승인 얼굴의 원본 통과·승인 유지·삭제 직후 재보호, 활성 session 삭제 거부, RPC 취소 후 재연결과 tracker reset 확인
- 테스트에서 만든 session을 모두 삭제하고 목록에 잔류하지 않는지 확인

`main`과 protobuf 원본·generated client·공개 transport 구현의 변경은 없다.
기존/신규 worker에서 사용한 protobuf descriptor SHA-256은 동일하다.

```text
4c921079596d1ea8140524a6edc09af080f32fbf135b8bb70468f5b3c4e6eb39
```

브라우저 ILF1/ILR1 v2와 WebRTC JavaScript 계약 검사도 통과했다. Go 실제 프로세스 및
실제 iOS 기기와 외부 네트워크 상호 운용은 실행하지 않았다. protobuf 무변경은 이들
클라이언트의 wire 호환 근거이며 실기기 검증을 대신하지 않는다.

## 안정성 수정 및 회귀 검증

- RF-DETR import가 프로세스 전체의 PyTorch matmul 정밀도를 `high`로 변경하는 부수 효과를 차단했다. 모델 로드 성공·실패 양쪽에서 기존 설정을 복원한다.
- TensorRT의 독립 stream이 평균·표준편차 등 기본 stream의 초기화를 기다리도록 의존성을 명시했다.
- 체크포인트 hash 읽기 또는 executor 생성 실패 시 RuntimeManager singleton 예약과 이미 만든 executor를 정리하여 재시작할 수 있게 했다.
- Linux 전체 일반 회귀: **489 passed, 509 subtests passed**, SCRFD artifact와 NVENC opt-in **2 skipped**. 실제 RF-DETR 체크포인트의 CPU VOD SIGTERM·재개는 별도 **1 passed**.
- 실제 GPU graph cache: 6개 해상도를 3회 순환해 상한 4개 eviction, 선택적/전체 mask 전송 동일성, 다음 추론 이후 lazy mask 18개 보존 검증 통과. 검정/흰색 12입력의 유한 출력 확인.
- 추가 NVENC opt-in WebRTC 검사 **14 passed, 3 subtests passed**. TF32 off FP32 대비 6얼굴 입력 strict 비교도 통과했다. 기존 BoT-SORT와의 360프레임 동일 출력 검증 결과는 [전환 보고서](rfdetr-migration.md)에 기록했다.

## 연속 부하 검증

FHD 4얼굴·2세션·raw YUV420·window 1 조건을 120초씩 3회 연속 측정했다.
각 회차에 5초 warmup과 신규 session을 사용했다. 측정 합계 **360초·15,029프레임**에서
응답 순서 오류·처리 실패·합성 fallback이 없었고 모든 프레임의 4얼굴이 탐지·합성됐다.
합산 처리량은 회차별 **43.116 / 41.134 / 40.992 FPS**였다.

서버 PID의 VRAM은 모든 측정 sample에서 **1,832MiB**였다. RSS는 첫 회차 시작
3,034MiB에서 두 번째 회차 종료 3,068MiB로 증가했으며 세 번째 종료는 3,068MiB였다.
측정 구간의 thermal slowdown은 0이었다. 이 결과는 6분 관찰이며 수시간·수일의 누수
부재를 증명하지 않는다. 고정 grid 영상 반복이므로 실제 이동·가림 장면의 품질 평가도 아니다.
원본은 `reports/rfdetr-102/audit-soak/resources.json`이다.

## 재현 및 원본

```bash
# 품질 평가에만 필요한 선택 의존성
.venv/bin/python -m pip install pycocotools
.venv/bin/python scripts/benchmark_detector_quality.py \
  --dataset /path/to/polygon-dataset --classes 0 --backend tensorrt \
  --output reports/quality-face-trt.json
# 번호판은 --classes 1, FP32 원본은 --backend pytorch

.venv/bin/python scripts/verify_live_contract.py \
  --target 127.0.0.1:50073 --image /path/to/single-face.jpg \
  --output reports/api-contract.json
```

GPU 경로는 `/home/gorani/apps/innolive-rfdetr-102`다. 원본 결과는
`reports/rfdetr-102/quality-{face,plate}-*.json`, `api-contract-{new,legacy}.json`,
`graph-stress.json`, `audit-numeric.json`과 `.runtime/audit-{tests,vod,graphs}.log`에 보관했다.
품질 결과 JSON은 모든 이미지·라벨의 SHA-256을 포함한다. 기존 모델 결과는 별도 기존
venv에서 추론한 detector-neutral rows를 읽으며 제품의 Ultralytics 종속성을 추가하지 않는다.

| 표본 | 서버 데이터 경로 | 이미지·라벨 목록 digest |
| --- | --- | --- |
| 얼굴 | `/home/gorani/apps/workspace/benchmark_results_v2/benchmark_v2` | `03b9ac177748c50481f58d96db5c91c104c7f72938729ce961b92e96e1b2acf6` |
| 번호판 | `/home/gorani/apps/workspace/benchmarks/v10_vehicle_context_test_20261016/eval_dataset` | `9d91a5ee74edfec26743c18a8c3e7a5c7b2dd9e643d2cd4af5d3ca7fd4c06312` |

이 평가의 FP32 reference는 matmul `highest`, cuDNN TF32 off다. 추가 진단 3입력에서
CPU PyTorch·CUDA FP32·ONNX CPU 출력을 대조해 export 자체의 큰 불일치를 발견하지 않았다.
이전 전환 보고서의 초기 FP32/FP16 비교에서는 RF-DETR import에 의해 TF32가 허용됐으므로,
엄밀한 FP32 reference가 아니었다. 새 비교 도구는 기준 정밀도를 명시한다.
