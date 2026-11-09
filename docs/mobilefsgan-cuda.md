# MobileFSGAN K22741 CUDA 실행

> 현재 탐지 backend는 RF-DETR-Seg Small입니다. 새 설치·엔진·성능은 [RF-DETR 전환 가이드](rfdetr-migration.md)를 따르세요. 아래 과거 측정의 YOLO 수치는 당시 모델 기준입니다.

`mobilefsgan_k22741_inference_20261108.tar.gz`의 BN 보정 K22741 G와 자체
MobileFaceNet C_04000을 기존 gRPC 얼굴 합성 경로에 연결한다.
탐지는 RF-DETR-Seg Small, 추적은 BoT-SORT를 사용한다. `models/inno_fs.pth`는 사용하지 않는다.
모델 재학습이나 BN 재보정은 수행하지 않는다.

## 설치와 실행

[팀 Google Drive 배포 폴더](https://drive.google.com/drive/folders/1B4ZWbUPJp-t6mzJpSYqhkOxTRM1eH0i1?usp=sharing)의
모델·프리셋 4개 파일을 [다운로드 가이드](face-swap-model-download.md)에 따라 설치한다.
이 번들은 [MobileFSGAN 다운로드](https://drive.google.com/file/d/1DQic3xT-CpkvLvvvc_fHFyWJDzqesciN/view?usp=drivesdk)에서 받을 수 있다.

Linux NVIDIA CUDA 환경에서 저장소 런타임 의존성을 먼저 설치하고
`requirements-mobilefsgan-cuda.txt`를 추가한다. 브라우저 WebRTC를 쓰려면
`requirements-webrtc.txt`도 필요하다. 가중치·엔진·프리셋 이미지는 Git에 포함하지 않는다.

```bash
mkdir -p models/face_swap/mobilefsgan
tar -xzf /path/to/mobilefsgan_k22741_inference_20261108.tar.gz \
  --strip-components=1 -C models/face_swap/mobilefsgan
.venv/bin/python models/face_swap/mobilefsgan/verify_files.py
.venv/bin/python -m pip install -r requirements-mobilefsgan-cuda.txt
```

원본 엔진은 TensorRT **11.4.0.106**, RTX 3090, compute capability 8.6용이다.
시작 시 버전·GPU·capability·원본 ONNX/YuNet·엔진 SHA-256을 검사한다.
다른 장비나 TensorRT에서는 제공된 엔진을 억지로 로드하지 않고 재빌드한다.
RF-DETR 엔진도 같은 TensorRT 버전으로 빌드해야 한다.

```bash
.venv/bin/python scripts/export_tensorrt.py \
  --checkpoint models/rfdetr_seg_s.pth --output models/rfdetr_seg_s_b1.engine

# 원본을 보존한 별도 출력 디렉터리. 기존 디렉터리 덮어쓰기 금지.
.venv/bin/python scripts/build_mobilefsgan_engines.py \
  --precision mixed_fp16 --with-yunet --output .runtime/mobilefsgan-fast
.venv/bin/python scripts/benchmark_mobilefsgan.py \
  --engine-dir .runtime/mobilefsgan-fast --onnx-reference --quality-samples 64 \
  --output reports/mobilefsgan-quality
.venv/bin/python scripts/verify_mobilefsgan_landmarks.py \
  --engine-dir .runtime/mobilefsgan-fast --output reports/mobilefsgan-landmarks.json

# 실제 gRPC 서버와 기존 사진·영상·카메라 테스트 웹 클라이언트
MOBILEFSGAN_ENGINE_DIR="$PWD/.runtime/mobilefsgan-fast" \
  WEB_HOST=127.0.0.1 bash scripts/run_mobilefsgan_cuda.sh
```

기본 gRPC는 `127.0.0.1:50071`, 웹은 `:8453`이다. 준비된 mixed 엔진을
명시하지 않으면 묶음의 원본 FP32 엔진을 사용한다. Mac에서 서버를 조작하거나
화면을 볼 수 있지만 추론과 성능 측정은 GPU 서버에서 실행한다.

```bash
ssh -N -L 8453:127.0.0.1:8453 -L 50071:127.0.0.1:50071 gorani@gpu
# 브라우저: http://localhost:8453
```

카메라는 localhost 또는 HTTPS에서 사용한다. `SSL_CERTFILE`, `SSL_KEYFILE`로
기존 launcher의 HTTPS 설정을 사용할 수 있다. 사진·영상 테스트도 기존 웹 UI를 사용하며,
CLI 프리뷰는 다음과 같다.

`/`는 WebRTC 카메라·영상 클라이언트, `/jpeg`는 사진·화이트리스트 및 JPEG 비교
클라이언트다. SSH 포워딩은 HTTP에 적용되며 WebRTC 미디어는 브라우저에서 GPU 서버의
ICE 주소까지 직접 연결되어야 한다. 같은 VPN/LAN 밖에서는 TURN 등 별도 구성이 필요하다.
`/jpeg`는 HTTP 포워딩만으로도 사용할 수 있다.

```bash
.venv/bin/python scripts/face_preset_client.py --target 127.0.0.1:50071 \
  --video /path/to/test.mp4 --mode face_swap --long-edge 1920 --frames 300 \
  --output-dir reports/mobilefsgan-preview
```

## 처리와 실패 계약

- Source는 묶음과 동일한 YuNet 5점 similarity 정렬, 256px RGB `[-1,1]` 입력을 사용한다. C 내부의 112px 변환을 중복 적용하지 않는다. 512D 정규화 embedding은 프리셋 해시별 최대 640개를 캐시한다.
- 프레임 전체 YuNet 결과를 얼굴 사이에 공유하고, 매칭 실패 시 해당 ROI를 재검출한다. `--with-yunet` 엔진은 Target 정렬만 TensorRT FP32로 실행하고 Source는 원본 CPU YuNet을 유지한다. Target crop은 합성된 출력이 아닌 원본에서 읽는다.
- 다른 Source 인물도 동일한 G의 별도 `source_z` 행으로 함께 합성한다. 최대 8개씩 처리하고, 각 batch shape별 TensorRT context·주소·CUDA Graph를 고정한다.
- Context는 한 stream에서 renderer lock 아래 순차 실행한다. 공유 activation workspace를 실제 최대 batch에 맞춰 할당하고 pinned 입력·출력 버퍼를 재사용한다. 그래프 capture는 기존 CUDA capture lock과 조정한다.
- 역 affine과 타원 alpha는 얼굴 polygon의 bounding ROI에만 적용한다. polygon 밖 픽셀은 유지하고 여러 얼굴은 추적 순서로 합성한다. 모든 출력 검증과 blending 계산이 성공한 뒤에만 출력 ROI를 수정한다.
- Source·정렬·엔진·출력 오류는 해당 얼굴의 블러로 처리한다. whitelist, held face, 번호판, 프리셋 선택과 세션별 identity 계약은 기존 경로를 유지한다. 묶음의 사진 데모처럼 실패 시 원본 얼굴을 반환하지 않는다.
- `MOBILEFSGAN_MAX_BATCH=1 MOBILEFSGAN_CUDA_GRAPHS=0`으로 순차 비교가 가능하다. `OPENCV_FOR_THREADS_NUM=1`이 CUDA launcher 기본이며 짧은 ROI 연산의 과도한 CPU 스레드를 제한한다.

혼합 FP16은 C를 FP32로 유지하고 G의 연산 dtype만 변환한다. G의 reduce·sqrt·pow·div
연산은 FP32로 유지하며 I/O는 FP32이다. 원본 그래프·변환 그래프·생성 엔진 해시를 모두
기록한다. 새 엔진은 실제 프리셋으로 수치·화면 검사를 통과한 뒤 사용한다.
성능 수치와 검증 범위는 아래 측정 결과에 기록한다.

## 재현 가능한 측정

```bash
# 같은 crop/source의 원본 FP32 순차 처리와 최적화 엔진 비교
.venv/bin/python scripts/benchmark_mobilefsgan.py \
  --engine-dir .runtime/mobilefsgan-fast --onnx-reference --quality-samples 64 \
  --repeats 80 --output reports/renderer-mixed

# 서버 준비 완료 후 실행. CPU 이미지 생성/JPEG 인코딩·디스플레이는 측정에서 제외.
.venv/bin/python scripts/benchmark_face_capacity.py --target 127.0.0.1:50071 \
  --cases 1x1,2x1,4x1,8x1,1x4,4x2 --width 1920 --height 1080 \
  --seconds 10 --warmup 3 --transport raw --window 1 --output reports/grpc-fhd
```

`--transport jpeg`는 JPEG 입출력, `--window 2`는 최대 2프레임 in-flight 조건이다.
raw는 YUV420p이다. 결과는 세션별 FPS·서버 p50/p95·왕복 p50/p95·실제 검출/합성률·
fallback 사유를 기록한다. 응답 순서·빈 출력·출력 해상도도 검사한다.
왕복 지연은 클라이언트의 한 monotonic clock으로 전송 직전부터 응답까지 측정한다.
GPU loopback 측정은 외부 회선·브라우저 표시·물리 카메라 지연을 의미하지 않는다.

## 2026-11-08 RTX 3090 측정

Linux x86_64, RTX 3090 24GB, driver 595.99.02, TensorRT 11.4.0.106,
PyTorch 2.14.0+cu126, Python 3.14.4, OpenCV 5.0.0에서 검증했다.
기존 GPU 서비스는 실행 중이나 별도 부하를 주지 않았고, 이 작업의 성능 시험은 순차 실행했다.

움직이는 프리셋 얼굴 grid를 각 해상도에 맞춰 확대했다. 각 조건 예열 3초 후 10초,
raw YUV420p gRPC loopback, window 1, 전송 속도 제한 없이 측정한 처리 용량이다.
테스트 입력은 4종 프리셋의 반복이며 실제 군중·카메라 품질 평가가 아니다.

| 입력 | 얼굴 | FPS | 서버 p50 / p95 (ms) | gRPC RTT p50 / p95 (ms) |
|---|---:|---:|---:|---:|
| 640×480 | 1 | 87.3 | 10.24 / 11.09 | 11.26 / 12.22 |
| 640×480 | 2 | 67.7 | 13.47 / 15.00 | 14.54 / 16.48 |
| 640×480 | 4 | 48.1 | 19.23 / 22.03 | 20.43 / 23.23 |
| 640×480 | 8 | 29.5 | 32.02 / 36.05 | 33.34 / 37.43 |
| 1280×720 | 1 | 70.0 | 12.07 / 13.27 | 14.03 / 15.35 |
| 1280×720 | 2 | 56.0 | 15.53 / 17.97 | 17.56 / 20.11 |
| 1280×720 | 4 | 39.3 | 22.83 / 26.13 | 24.93 / 28.57 |
| 1280×720 | 8 | 24.8 | 37.30 / 42.43 | 39.55 / 45.00 |
| 1920×1080 | 1 | 51.8 | 14.99 / 16.63 | 19.00 / 20.70 |
| 1920×1080 | 2 | 42.8 | 19.00 / 22.33 | 22.90 / 26.10 |
| 1920×1080 | 4 | 30.9 | 27.74 / 31.53 | 31.73 / 36.20 |
| 1920×1080 | 8 | 21.1 | 42.50 / 49.02 | 46.47 / 53.78 |

전체 8,979개 측정 응답에서 요청한 수의 얼굴이 검출·합성됐고 fallback은 0건이었다.
이 성공률은 위 고정 입력에서의 결과이며 미검출 얼굴이나 실제 환경의 익명화 성공률을 뜻하지 않는다.

| 입력 | 얼굴 × 세션 | 세션별 FPS |
|---|---:|---:|
| 640×480 | 1 × 4 | 32.3 / 32.3 / 32.2 / 32.2 |
| 640×480 | 4 × 2 | 30.8 / 30.8 |
| 1920×1080 | 1 × 4 | 22.9 / 22.9 / 22.9 / 22.9 |
| 1920×1080 | 4 × 2 | 23.2 / 23.2 |

480p 원본 FP32 순차 합성·CPU YuNet 설정은 1/2/4/8얼굴에서 각각
**40.4 / 33.0 / 22.7 / 13.8 FPS**, 최종 설정은 **87.3 / 67.7 / 48.1 / 29.5 FPS**였다.
두 설정은 같은 YOLO 엔진·입력·전송 조건을 사용했다. 최종 설정에는 mixed FP16,
얼굴 batch, CUDA Graph, GPU YuNet, OpenCV thread 1 제한을 함께 적용했다.
원본 비교 설정의 OpenCV thread 수는 기본 24였다. 기존 MobileFaceSwap 모델과의 비교가 아니다.

같은 정렬 crop만 별도로 번갈아 측정한 1/2/4/8얼굴 합성 p50은
**5.74 / 11.72 / 23.87 / 47.99ms → 2.73 / 4.61 / 8.30 / 16.13ms**였다.
Source embedding은 예열됐으며 H2D·G·D2H만 포함한다. 이 값의 역수를 전체 FPS로 쓰지 않는다.
FP32 YuNet은 36개 입력에서 CPU p50 **8.05ms → TensorRT 1.05ms**였다.

C/G 공유 workspace는 각각 **3,612,672 / 492,316,672 bytes**이며,
이는 모델·입출력·YOLO·AdaFace를 포함한 프로세스 전체 VRAM이 아니다.
서버 준비 뒤 nvidia-smi 전체 사용량은 약 6.7GiB였고 기존 서비스의 약 4.4GiB가 포함됐다.

### 출력 및 회귀 검증

- 원본 CPU ONNX 대비 실제 얼굴 crop: RGB 0..1 MAE 0.000240, p99 0.001008, C cosine 약 1.0.
- 표정·안경·±8도 회전을 포함한 64개 crop: 최대 sample MAE 0.000289, 최대 sample p99 0.001716, 최대 단일 pixel 차이 0.007621. 각 sample MAE ≤0.0005, p99 ≤0.005 기준 통과.
- 480p/720p/FHD × 1/2/4/8얼굴 × 3각도, 36개 입력에서 CPU/TensorRT YuNet 검출 수 일치. 최대 landmark 차이 0.000244px, 최대 정렬 crop MAE 0.000001995.
- 원본 FP32/혼합 FP16 비교 이미지와 실제 FHD 합성 결과 육안 확인. 원본 모델의 이마·머리카락·안경 부근 합성 흔적은 남으며 자연스러움 개선으로 해석하지 않음.
- Linux 전체 Python 478 passed, 502 subtests passed, 2 skipped(기존 MobileFaceSwap SCRFD 미설치 및 NVENC opt-in). NVENC 활성화 별도 실행은 14 passed, 3 subtests passed. Ruff 전체, 변경 Python format, launcher shell syntax, JavaScript 계약 2개 통과.
- GPU 엔진 설치 뒤 `--device cpu`의 auto backend가 TensorRT를 잘못 선택하던 기존 오류 수정. VOD SIGTERM→resume 및 전체 회귀 재검증 통과.

원시 JSON·이미지·로그는 서버 `/home/gorani/apps/innolive-mobilefsgan-100/reports`와
로컬 `reports/mobilefsgan-100`에 보관했다. Mac 성능은 측정하지 않았다.

### 브라우저 확인

GPU 서버의 headless Chrome에서 640×480·4얼굴·30 FPS 가상 카메라를 입력하고,
5초 예열 후 40초간 WebRTC H.264/NVENC 경로를 측정했다.

| 항목 | 결과 |
|---|---:|
| 고유 출력 프레임 / FPS | 1,163 / 29.08 |
| 프리뷰→합성 화면 p50 / p95 | 33.30 / 33.60ms |
| 서버 처리 p95 / 최대 | 23.34 / 28.62ms |
| 100ms 초과 출력 간격 / freeze / packet loss | 0 / 0 / 0 |

영상 상단 16-bit frame marker를 브라우저 프리뷰와 수신 화면에서 읽어 같은 프레임을
비교했다. 화면 표시는 requestAnimationFrame 단위로 관측하므로 지연 값은 그 주기로
양자화된다. 이 수치는 실제 카메라 센서, 외부 네트워크 또는 Mac 브라우저 성능을
측정한 결과가 아니다. 결과 화면에서 합성 4명·blur 0명도 확인했다.

```bash
# camera4.y4m: reports에 보관된 marker 포함 가상 카메라 입력
.venv/bin/python scripts/benchmark_webrtc_browser.py \
  --url fast=http://127.0.0.1:8453 --fake-video reports/camera4.y4m \
  --browser /usr/bin/google-chrome --seconds 40 --warmup 5 \
  --output reports/browser-fast
```

검증 서버는 `/home/gorani/apps/innolive-mobilefsgan-100`의 독립 실행 환경이며,
기존 GPU 서버 프로세스는 교체하지 않았다. 현재 user systemd의 임시 unit으로 실행한다.
재부팅 시 자동 시작되는 서비스는 아니다.

```bash
ssh gorani@gpu 'systemctl --user status innolive-mobilefsgan-100.service'
ssh gorani@gpu 'journalctl --user -u innolive-mobilefsgan-100.service -n 60 --no-pager'
# 검증 서버 종료가 필요한 경우
ssh gorani@gpu 'systemctl --user stop innolive-mobilefsgan-100.service'
```

## 참고

- [기존 InSwapper와 동일 조건 성능 비교 및 PPT용 요약](face-swap-performance-comparison.md)
- [GPU·CPU 자원 사용량과 추가 최적화 점검](mobilefsgan-resource-audit.md)
- 전달 묶음의 `docs/INFERENCE_ARCHITECTURE.md`, `model_manifest.json`, `licenses/`에 원본 모델·정렬·전처리·출처 기록 보존
- [NVIDIA TensorRT CUDA Graph 및 context/workspace 규칙](https://docs.nvidia.com/deeplearning/tensorrt/11.4.1/performance/optimization.html)
- [TensorRT 11의 명시적 tensor precision](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/precision-control.html)
