# Mac에서 MobileFaceSwap 224 테스트

`models/best.pt`는 그대로 탐지·segmentation에 사용하고, 얼굴 합성 renderer만
MobileFaceSwap 224로 선택한다. 기존 추적, metadata, 프리셋 선택, whitelist,
번호판 blur, 합성 실패 시 blur 처리를 유지한다. protobuf 변경은 없다.

## 실행

Apple Silicon Mac에서 저장소 루트를 기준으로 실행한다.

```bash
.venv/bin/python -m pip install -r requirements-mobilefaceswap.txt
./scripts/run_mobilefaceswap.sh
```

브라우저에서 <http://127.0.0.1:8001>을 연다. **추론할 이미지 선택 → 추론 실행**으로
먼저 확인한 뒤 동영상 또는 **카메라 시작**을 사용한다. 화이트리스트에 등록한 얼굴은
합성을 건너뛴다. 중지는 실행한 터미널에서 `Ctrl+C`; 두 서버를 함께 종료한다.
이미 50051/8001 포트가 사용 중이면 기존 서버를 먼저 종료해야 한다.

실행 구성:

| 단계 | 실행 장치 / 파일 |
| --- | --- |
| 얼굴·번호판 탐지 및 segmentation | PyTorch MPS / `models/best.pt` |
| 얼굴 속성 분류 | PyTorch MPS / `models/face_metadata_extracter.pt` |
| 얼굴 정렬 | ONNX Runtime CoreML CPU/GPU / `landmarks/scrfd_10g_bnkps.onnx` |
| source identity·특징 추출 | Paddle CPU / `arcface.pdparams` |
| 224×224 얼굴 합성 및 마스크 | PyTorch MPS / `MobileFaceSwap_224.pdparams`에서 생성한 프리셋별 가중치 |

이 launcher의 **매 프레임 얼굴 합성은 MPS**에서 실행한다. 처음 사용하는 프리셋의
identity·조건부 가중치 생성은 원본 Paddle CPU로 수행하고, 그 결과를 PyTorch UNet에
strict 로딩해 GPU에 캐시한다. Conv/BatchNorm fusion과 channels-last 메모리 배치를 적용했다.
원본 `.pdparams`는 수정하지 않으며 추가 모델 다운로드는 하지 않는다. Paddle 3.3.1 / Python 3.13 /
macOS arm64에서 확인했다. `landmarks/glintr100.onnx`는 수정하지 않으며 이 경로에서는
사용하지 않는다. MobileFaceSwap에 필요한 identity feature map은 `arcface.pdparams`에서 얻는다.

프리셋은 기존 `config/face_presets.json`과 해당 이미지들을 사용한다. 모델 파일은
`models/mobilefaceswap/` 아래에 보관하며 Git에서 제외한다.

브라우저 launcher는 `face_swap` 모드를 사용한다. 느린 최초 모델 로딩을 위해
60초 응답 제한, 한 번에 한 프레임, 최대 10 FPS 입력 설정을 전달한다.
10 FPS는 입력 상한이며 실제 처리 속도를 보장하는 수치가 아니다.

## 개별 실행과 비교

서버만 시작하려면:

```bash
OMP_NUM_THREADS=1 .venv/bin/python ai_processor_server.py \
  --backend pytorch --model models/best.pt --device mps \
  --face-metadata-device mps \
  --face-swap-backend mobilefaceswap --face-swap-provider mps
```

다른 모델 폴더는 `--mobilefaceswap-dir /path/to/mobilefaceswap`으로 지정한다.
기존 InSwapper로 돌아가려면 서버를 재시작하고 `--face-swap-backend inswapper`를
사용한다. 일반 서버의 기본값은 여전히 InSwapper이며 기본 요청 모드는 blur이다.
MobileFaceSwap의 `--face-swap-provider cpu`는 원본 Paddle/ONNX CPU 경로를 선택한다.
`mps`는 PyTorch MPS + CoreML, `cuda`는 PyTorch CUDA + ONNX CUDA를 선택한다.
CUDA는 호스트에 `onnxruntime-gpu`가 필요하며 이번 Mac에서는 실행 검증하지 않았다.

이미 실행 중인 서버에 OpenCV 프리뷰 클라이언트로 접속할 수도 있다:

```bash
.venv/bin/python scripts/face_preset_client.py --camera 0 --mode face_swap
.venv/bin/python scripts/face_preset_client.py --video /path/to/video.mp4 --mode face_swap
.venv/bin/python scripts/face_preset_client.py --image /path/to/image.jpg \
  --mode face_swap --frames 35 --headless --output-dir reports/mobilefaceswap/grpc
```

## 구현 및 검증

[공식 구현](https://github.com/Seanseattle/MobileFaceSwap/tree/29552474c2f621c27818ff7fc32447e1a60a96fb)의
`model.py`, `arcface.py`를 변경 없이 `service/_vendor/mobilefaceswap/`에 포함했다.
224px full-affine 정렬 → source를 112px로 resize → RGB/ImageNet 정규화,
target RGB 0..1 입력, 학습된 mask와 paste-back 처리를 원본과 맞췄다.
최종 적용 영역만 기존 파이프라인의 YOLO segmentation polygon으로 추가 제한한다.
source-conditioned network는 프리셋별 최대 16개를 캐시하며 다른 프리셋의 가중치를
덮어쓰지 않는다. 누락·불일치 checkpoint, 얼굴 정렬 실패, 잘못된 출력은 기존 blur로 돌아간다.

카메라에서 크게 잡힌 얼굴의 ROI를 640px로 확대하면 SCRFD 검출이 실패할 수 있어,
target ROI는 320px로 조정한 뒤 기존 static 640px canvas에 넣는다. 실패하면 원본 전체
프레임을 640/320px 두 크기로 재검출하며, 추적 bbox와의 IoU 기준(0.25)은 유지한다.
여러 얼굴의 정렬·합성 입력은 항상 원본 프레임을 사용하고, 전체 프레임 검출 결과는
해당 프레임 안에서만 공유한다. 이전 프레임의 랜드마크를 재사용하지 않는다.

2026-10-19 카메라 blur 전환 수정 검증:

- 합성 프리셋 사진의 크기·위치·기울기를 연속 변경한 4:3/16:9 입력을 실제 WebSocket으로
  각 60프레임 전송: **120/120 `swapped`**. 같은 JPEG·반환 bbox에 기존 ROI 640px 정렬을
  적용하면 67/120에서 랜드마크 연결 실패. 이는 실제 웹캠 촬영 검증이 아닌 재현 입력이다.
- 얼굴 2개를 배치한 입력 12프레임 처리와 가림/복귀 25프레임 처리 확인. 가림 5프레임은
  `blur_fallback`, 가림을 제거한 후 합성 복귀. 총 157프레임 처리 후 health `ok` 확인.
- 측정한 전체 서버 처리 시간은 첫 프레임 제외 중앙값 109.21ms, p95 163.97ms.
  얼굴 수와 프리셋 초기화가 다른 조건을 섞은 값이며 브라우저 표시 FPS와 다르다.
- 재현 스크립트·입출력 JPEG·프레임별 상태는 로컬 `reports/mobilefaceswap/camera-fix/`에 저장.
  원본 모델 파일과 `best.pt` SHA-256은 유지했다.

2026-10-18 이 Mac의 실제 확인 결과:

- 공식 소스와 같은 source/target 입력으로 정렬·입력 tensor·합성 출력·mask·paste-back
  비교: 최대 차이 모두 0. 재현용 `scripts/verify_mobilefaceswap.py` 포함.
- 실제 HTTP 이미지 요청: `swapped`, 최초 모델 로딩 포함 약 3.70초.
- 640×640 합성 프리셋 이미지를 gRPC로 35프레임 반복: 35/35 `swapped`, identity 유지.
  첫 프레임 제외 `server_total_ms` 중앙값 168.8ms, p95 225.1ms. 네트워크·브라우저는 제외.
- `best.pt` SHA-256 전후 동일:
  `1566707c2c1c43f09a9a8929835ab16a16f9cfdc9f36b6d4d13d35caa5777f2e`.
- 실제 웹캠 움직임, 다양한 얼굴·가림·각도, 장시간 안정성, InSwapper와 동일 조건 성능
  비교는 아직 측정하지 않았다. 위 결과는 정지 이미지 반복 테스트다.

### Apple M4 10-core GPU 최적화 추가 검증

- 합성 모델 단독 FP32: Paddle CPU 약 26~31ms → fused/channels-last MPS 중앙값
  2.88ms, p95 3.02ms (warmup 3회, 측정 15회, CPU 출력 복사로 GPU 완료 동기화).
- 같은 정렬 crop과 같은 source 가중치에서 MPS와 Paddle의 합성 출력 최대 절대 오차
  `4.06e-6`, 평균 `7.57e-8` (0..1 범위), mask 최대 오차 `2.66e-6`.
- SCRFD 640px 단독: CPU 중앙값 39.82ms → CoreML 10.44ms. ONNX profile에서
  CoreML 실행을 확인했고 9개 출력 tensor 최대 오차는 `8.59e-6` 이하였다.
- 이 단독 수치는 전체 파이프라인 FPS가 아니다. 근거는 `torch-probe.json`,
  `coreml-probe.json`과 최종 `grpc-mps/metadata.jsonl`에 저장한다.
- 최종 MPS/CoreML 서버에서 640×640 이미지 35프레임 반복: 35/35 `swapped`, identity 유지.
  첫 프레임 제외 전체 서버 중앙값 **101.43ms**, p95 **116.89ms**. 기존 CPU 합성 경로의
  168.78ms 대비 약 **1.66배** 빠르다. 합성·인코딩 구간 중앙값은 **23.56ms**,
  YOLO 추론은 **70.50ms**였다. 두 실행의 무작위 프리셋 identity는 달랐다.
  최초 MPS/CoreML 준비 포함 첫 프레임은 6.72초이며, 위 캐시 후 성능과 구분한다.

### RTX 3090 예상 범위 (실측 아님)

224px 얼굴 1개, batch=1, source 가중치 캐시 완료, 640px 입력, 최신 데스크톱 CPU,
YOLO와 MobileFaceSwap/SCRFD를 CUDA에서 실행하는 조건의 **계획용 추정**이다.

| 범위 | 예상 시간 | 역수 기준 처리량 |
| --- | --- | --- |
| MobileFaceSwap 합성 모델만 | 1~5ms/얼굴 | 200~1,000회/초 |
| 탐지·추적·정렬·합성·JPEG 전체 서버 | 20~50ms/프레임 | 20~50 FPS |

이는 M4 실측 단계별 병목과 작은 batch의 GPU 실행 오버헤드를 고려한 넓은 추정 구간이며
신뢰구간이나 벤치마크 결과가 아니다. 신규 identity의 CPU 준비, metadata 갱신,
다중 얼굴, whitelist 인식, FHD 합성, 네트워크 지연은 별도로 추가된다.
현재 Mac의 CPU provider를 그대로 쓰면 3090을 장착해도 swap이 CUDA로 이동하지 않는다.

논문의 [224px IDN은 0.50M parameters / 0.33G FLOPs](https://arxiv.org/html/2201.03808)이지만,
[NVIDIA 성능 가이드](https://docs.nvidia.com/deeplearning/performance/dl-performance-gpu-background/index.html)처럼
메모리 이동·작은 연산·launch 비용이 지배할 수 있어 TFLOPS 비례 계산을 FPS로 쓰지 않았다.
3090의 실제 수치는 해당 호스트에서 `--face-swap-provider cuda`로 같은 입력을 측정해야 확정된다.

검증 artifacts는 Git에서 제외한 `reports/mobilefaceswap/`에 저장한다.

```bash
git clone https://github.com/Seanseattle/MobileFaceSwap.git /tmp/MobileFaceSwap-reference
git -C /tmp/MobileFaceSwap-reference checkout 29552474c2f621c27818ff7fc32447e1a60a96fb
OMP_NUM_THREADS=1 .venv/bin/python scripts/verify_mobilefaceswap.py \
  --upstream /tmp/MobileFaceSwap-reference \
  --source config/face_presets/images/g0_a0_gl0_e1_id0.png \
  --target config/face_presets/images/g0_a1_gl0_e1_id0.png --torch-provider mps
```
