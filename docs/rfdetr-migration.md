# RF-DETR Seg Small 파이프라인 전환

2026-11-09, RTX 3090의 `gorani@gpu`에서 검증했다. 이 문서의 성능은 해당 GPU의
loopback 테스트이며, 외부 네트워크·실제 카메라·Mac 성능 보장은 아니다.

## 적용 모델과 경로

| 항목 | 값 |
| --- | --- |
| 제공 파일 | `checkpoint_best_ema.pth` |
| 설치 경로 | `models/rfdetr_seg_s.pth` |
| 체크포인트 SHA-256 | `a5da6cb2a53ff001215149bb941678a27b02acd1dba6521a419fea30defdb2cf` |
| 모델 / 라이브러리 | `RFDETRSegSmall` / `rfdetr==1.11.2` |
| 입력 / query / mask | RGB 624×624 / 100 / 156×156 logits |
| 학습 메타데이터 | epoch 12, face·plate 2 classes |
| API class | `0=face`, `1=number_plate`; logit slot 2는 background |
| 배포 엔진 | `models/rfdetr_seg_s_b1.engine`, static B1, TensorRT FP16 |

체크포인트와 엔진은 Git에 포함하지 않는다. 체크포인트 식별 정보는
[`models/rfdetr_seg_s.json`](../models/rfdetr_seg_s.json)에 보관한다.
사용자가 앞서 제공한 `inno_fs.pth`는 사용하지 않는다.

## 유지한 파이프라인

```text
gRPC / WebSocket / WebRTC / VOD
  → RF-DETR segmentation
  → stream별 BoT-SORT + GMC + 1-frame mask hold
  → 얼굴: YuNet + AdaFace whitelist / metadata / preset 선택
  → MobileFSGAN·MobileFaceSwap·InSwapper 또는 blur
  → 처리 영상 + 기존 protobuf / ILF1 metadata
```

탐지 모델을 메인 서버·VOD·실험 클라이언트·export·실행 스크립트에서 교체했다.
Ultralytics import·설치 요구 사항과 기존 `models/best.pt`를 제거했다.
GPU 검증용 새 venv에서도 `importlib.util.find_spec("ultralytics") is None`을 확인했다.
과거 측정 문서와 라이선스 기록은 당시 모델명을 보존한다.

BoT-SORT는 MIT 원본의 XYWH Kalman filter와 독립 association/GMC 구현을 사용한다.
기존 설정의 high/new threshold 0.25, low threshold 0.05, lost buffer 30,
score fusion, sparse optical flow, ReID 비활성화, stream별 ID를 유지한다.
얼굴과 번호판 간 association은 차단하여 얼굴 승인 상태가 번호판으로 넘어가지 않게 한다.
검출 공백의 mask hold는 기존과 같이 최대 1프레임이다.

공개 class 이름과 protobuf 필드는 유지한다. detector 입력은 640→624로 바뀌므로
health의 `image_size`와 profile은 `B1-624-Q90-W5`로 바뀐다.
브라우저 이미지 업로드의 장변 제한은 기존 640을 유지한다.
실험 클라이언트는 여러 frame을 순서대로 B1 추론하며, 기존 `yolo_batch` 통계 키는
클라이언트 호환용으로 남긴다. 이 값은 실제 detector tensor batch 크기가 아니다.
기존 YOLO dynamic B4 engine은 사용할 수 없다.

## 적용한 최적화

- GPU에서 BGR→RGB, square resize, ImageNet normalization을 수행하고 TensorRT 추론까지 CUDA Graph로 재사용한다.
- pinned host buffer와 고정 GPU buffer를 사용한다. frame shape별 graph cache는 최대 4개다.
- 전체 mask를 매번 복사하지 않고 선택된 query의 mask만 GPU→CPU로 전송한다.
- 실제 추적 결과에 필요한 mask만 contour로 변환한다. 다음 추론의 buffer 재사용에 영향을 받지 않도록 선택 plane을 소유한다.
- GMC와 탐지의 기존 병렬 처리, bounded inference/recognition/render lane, MobileFSGAN batch·CUDA Graph 경로를 유지한다.

RF-DETR는 square resize를 사용하며 YOLO letterbox/NMS를 적용하지 않는다.
foreground sigmoid score의 top 100을 사용하고 background slot을 제외한다.
여러 조각으로 나뉜 mask는 단일 wire polygon으로 모두 보호하도록 convex hull로 묶는다.
원본보다 넓은 영역이 포함될 수 있다. 출력 polygon은 기존처럼 최대 64점이다.
mask 경계 padding은 624 기준으로 계산하므로 FHD에서 4px이다.

## 설치와 재현

저장소 루트에서 실행한다. NVIDIA GPU가 있는 Linux에서 엔진을 새로 생성한다.

```bash
cp /path/to/checkpoint_best_ema.pth models/rfdetr_seg_s.pth
sha256sum models/rfdetr_seg_s.pth
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-export.txt
.venv/bin/python scripts/export_tensorrt.py --device 0
```

TensorRT 11은 upstream의 typed FP16 ONNX 변환을 사용하며 FP32 IO를 유지한다.
TensorRT 10 이하의 FP16 builder 경로도 제공하지만 이번 실측은 11.4에서 수행했다.
manifest schema 2가 체크포인트·엔진 SHA-256, 클래스, 해상도, 전처리를 검증한다.
잘못된 엔진은 조용히 다른 backend로 바뀌지 않고 시작을 실패시킨다.

필요한 face metadata·AdaFace·YuNet·preset·MobileFSGAN artifact를 준비한 뒤
[`MobileFSGAN CUDA 가이드`](mobilefsgan-cuda.md)의 환경으로 실행한다.

```bash
MOBILEFSGAN_ENGINE_DIR=.runtime/mobilefsgan-fast \
  bash scripts/run_mobilefsgan_cuda.sh

# FP32 원본 확인용. TensorRT를 사용하지 않는다.
.venv/bin/python ai_processor_server.py --backend pytorch --device 0 \
  --model models/rfdetr_seg_s.pth

# 개수·class·box/mask IoU·선택적 mask 전송의 동일성 검증
.venv/bin/python scripts/verify_rfdetr.py image1.jpg image2.jpg \
  --output reports/rfdetr-parity.json
```

비교 도구는 score≥0.25 검출의 개수가 같고 class별 box/mask IoU≥0.9인지를 검사한다.
미충족 시 JSON에 실패를 기록하고 exit code 1을 반환한다. score 차이도 함께 기록한다.
마이크로벤치마크 시간은 모든 candidate contour 변환까지 포함한다. 실제 서버는 추적에
선택된 contour만 만들기 때문에 candidate가 많은 이미지의 시간과 직접 비교하면 안 된다.

## 검증 결과와 정밀도 한계

| 검증 | 결과 |
| --- | --- |
| Linux 회귀 테스트 | 487 passed, 502 subtests passed; SCRFD artifact·NVENC opt-in 2 skipped |
| 실제 체크포인트 VOD | CPU 경로 SIGTERM·재개 테스트 1 passed |
| 기존 BoT-SORT 비교 | 일반 이동·가림 240프레임 + 카메라 이동 120프레임, ID·좌표·score·class·index 완전 일치 |
| 얼굴 입력 FP32/FP16 | 480p/FHD × 1/4/8얼굴의 6이미지, 총 26검출 모두 통과 |
| 얼굴 box/mask IoU 최솟값 | 0.99185 / 0.99908 |
| 선택적 mask 전송 | 얼굴·차량 총 18이미지에서 전체 mask 전송 결과와 정확히 일치 |
| 차량 스트레스 FP32/FP16 | 12이미지 중 7이미지 strict 비교 실패 |

차량 샘플은 사용자가 보유한 차량 분류 이미지에서 고정 seed로 뽑았으며, 일부 번호판이
이미 가려져 있고 얼굴 정답이 없다. 따라서 정확도/mAP 데이터셋이 아니다.
원본 FP32도 차량 일부를 얼굴로 검출하며, FP16 변환 시 score 0.25 근처의 개수와
일부 작은 mask가 달라졌다. 최대 관측 score 차이는 약 0.17이다.
번호판이 검출된 `car-07`의 plate box/mask 비교는 통과했다.
다만 동일 파일을 브라우저 이미지 API의 JPEG 재인코딩 경로로 보내면 plate가 검출되지
않았다. 이 샘플은 번호판 보호의 실영상 성공 사례로 계산하지 않는다.
이미지 API의 4얼굴 검출·합성 및 기존 640px 장변 계약은 실제 요청으로 확인했다.

정규화·attention·head를 FP32로 유지한 mixed precision과 FP32/TF32 엔진도 비교했지만,
모든 스트레스 입력의 일치를 확보하지 못했고 지연이 늘었다. 현재 기본은 검증한
upstream FP16 변환이다. 얼굴 테스트 통과를 모든 실제 장면의 FP32 동등성이나
탐지 정확도 유지로 해석하면 안 된다. 배포용 face/plate 정답 데이터셋의 recall·mAP와
실제 이동 영상의 IDF1/HOTA는 아직 검증하지 않았다. `--backend pytorch`로 FP32 원본을
직접 사용할 수 있다.

## GPU 성능

동일 preset 얼굴 grid, MobileFSGAN 엔진·설정, loopback gRPC raw YUV420, window 1로 비교했다.
각 조건은 warmup 5초 후 15초간 측정했다. 아래 값은 단일 세션의 전체 파이프라인 처리량이다.
서버 시간에는 탐지·추적·인식·합성이 포함되며 RTT는 client 왕복 시간이다.

| 영상 | 얼굴 수 | 기존 YOLO FPS | RF-DETR FPS | 변화 | RF 서버 p50 / p95 | RF RTT p50 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 640×480 | 1 | 86.3 | **83.2** | -3.6% | 10.80 / 12.19 ms | 11.79 ms |
| 640×480 | 4 | 47.8 | **46.0** | -3.8% | 20.11 / 23.66 ms | 21.31 ms |
| 640×480 | 8 | 29.5 | **29.2** | -1.1% | 32.29 / 36.72 ms | 33.65 ms |
| 1920×1080 | 1 | 46.0 | **44.7** | -2.9% | 15.43 / 16.95 ms | 21.47 ms |
| 1920×1080 | 4 | 30.6 | **31.3** | +2.4% | 27.11 / 31.21 ms | 31.20 ms |
| 1920×1080 | 8 | 20.9 | **21.1** | +1.0% | 41.93 / 49.15 ms | 46.32 ms |

이 비교에서는 기존 대비 FPS 차이가 약 −4%~+2%였다. 단일 실행의 관측치이며 통계적 우열을
의미하지 않는다. 탐지 box/mask가 달라지면 후단의 crop·렌더링 부하도 달라진다.
모든 측정 프레임에서 입력 얼굴 전체가 탐지·합성됐고 fallback은 0건이었다.

최종 detector 구간 p50은 480p에서 **4.73~5.23ms**, FHD에서 **5.77~6.03ms/frame**이었다.
같은 입력의 최초 구현 대비 선택적 mask 전송·지연 contour 변환으로 detector 구간이
약 0.4~1.6ms 줄었다. 이는 얼굴 1개당 합성 시간이나 엔진 kernel 단독 시간이 아니다.

| 영상 | 얼굴 수 | GPU busy 평균 | 전력 평균 | 서버 VRAM | 서버 CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| 480p | 1 | 60.3% | 297 W | 1.70 GiB | 108% |
| 480p | 4 | 55.2% | 298 W | 1.74 GiB | 111% |
| 480p | 8 | 55.0% | 281 W | 1.79 GiB | 112% |
| 1080p | 1 | 34.6% | 191 W | 1.75 GiB | 99% |
| 1080p | 4 | 37.8% | 242 W | 1.79 GiB | 114% |
| 1080p | 8 | 39.9% | 251 W | 1.84 GiB | 110% |

FHD 4얼굴 × 2세션은 합산 **41.27 FPS**(각 20.60/20.67 FPS), 480p는 합산 **59.93 FPS**였다.
FHD 2세션에서 GPU busy 47.7%, CPU 194.7%, 전력 290W를 기록했다.
단일 세션보다 GPU·CPU 사용률이 올라가지만 세션별 지연은 증가한다.

남은 병목은 다인 얼굴 합성·CPU 영상 처리 및 전송이다. FHD 8얼굴에서 render p50은
29.86ms이고 detector는 6.03ms다. 따라서 detector만 추가 가속해도 전체 FPS 개선 폭은 제한된다.
480p에서는 약 298W로 300W 전력 제한에 도달하므로 GPU busy가 100% 미만이라고 해서
동일 비율의 성능 여유가 있다는 뜻은 아니다. thermal slowdown은 측정 구간에서 없었다.

Chrome headless + 640×480 4얼굴 fake camera로 WebRTC를 별도로 25초 측정했다.
단일 세션의 RTP decoded FPS는 **28.82**, 화면 marker 기준 preview→output p50/p95는
**33.3/33.8ms**였다. 2세션은 각각 **27.76/27.92 decoded FPS**, marker 지연
**66.7/100ms**였다. 패킷 손실·RTP freeze는 0건이었다.

2개 headless 탭 중 한 탭에서는 화면 callback 측정이 7.62 FPS로 내려갔지만 RTP decoded
counter는 27.76 FPS를 유지했다. 따라서 dual-tab 화면 표시 FPS를 28 FPS로 주장하지 않는다.
이 측정은 loopback fake camera이며 실제 카메라의 capture 지연이나 외부 네트워크 지연은
포함하지 않는다. NVENC bitrate 변경·decodable frame 검사를 포함한 WebRTC 테스트는
별도 opt-in 실행에서 **14 passed, 3 subtests passed**였다.


## 검증 환경과 원본 결과

- RTX 3090 24GiB, NVIDIA driver 595.99.02, 전력 제한 300W, Ryzen 7900 24 logical CPU
- Python 3.14.4, Torch 2.14.0+cu126, RF-DETR 1.11.2, TensorRT 11.4.0.106, OpenCV 5
- 저장소 설치 pin은 Torch 2.13.0/Torchvision 0.28.0이며 GPU 검증은 서버의 기존 Torch 2.14 환경에서 수행
- GPU 작업 경로: `/home/gorani/apps/innolive-rfdetr-102`
- 원본 비교: `reports/rfdetr-102/{yolo-480,yolo-1080,final-480,final-1080}/resources.json`
- 정밀도: `reports/rfdetr-102/{parity,parity-final,parity-release,parity-mixed,precision-candidates}.json`
- WebRTC: `reports/rfdetr-102/browser-{1,2}/result.json`; 실제 이미지 API: `reports/rfdetr-102/api.json`
- 테스트 로그: `.runtime/{final-tests,vod-tests,nvenc-tests}.log`
- GitHub Actions: Python 3.12/3.13 설치·CPU 테스트, lint·browser contract, 협업 정책 검사 통과

검증 후 이 작업의 테스트 서버를 RF-DETR로 교체했다. GPU의 `innolive-rfdetr-102.service`가
gRPC `127.0.0.1:50071`, browser `127.0.0.1:8453`을 제공한다. 기존
`innolive-mobilefsgan-100.service`는 중지하고 해당 폴더를 rollback용으로 유지한다.
다른 GPU 서비스는 변경하지 않는다. 로컬 SSH tunnel의 `http://127.0.0.1:8453`에서
기존 클라이언트를 사용할 수 있다.

GPU/CPU 자원 값은 warmup을 제외한 측정 구간에서 200ms 간격으로 수집했다.
GPU busy는 NVML sampled busy time이며 SM occupancy가 아니다. GPU 전체 VRAM에는
다른 프로세스가 포함되므로 서버 PID의 VRAM을 별도로 보고한다. CPU 100%는 논리 코어 1개다.

Upstream 계약: [ONNX export](https://rfdetr.roboflow.com/latest/exports/onnx/),
[TensorRT export](https://rfdetr.roboflow.com/latest/exports/tensorrt/),
[BoT-SORT MIT source](https://github.com/NirAharon/BoT-SORT).
