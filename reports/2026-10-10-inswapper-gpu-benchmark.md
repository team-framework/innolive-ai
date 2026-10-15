# InSwapper GPU 성능 측정 — 2026-10-10

RTX 3090에서 네트워크를 제외해 측정했다. InSwapper 자체의 평균 forward는 ONNX CUDA
16.71 ms, TensorRT FP32 14.05 ms, TensorRT mixed 5.42 ms였다. 현재 메인 기능의
얼굴 1개 처리량은 640×360 34.88 FPS, 720p 17.48 FPS, 1080p 12.82 FPS다.
별도 TensorRT 실험 경로는 720p 얼굴 1개에서 54.88 FPS를 기록했다.

메인의 InSwapper backend는 아직 ONNX CPU/CUDA이며 TensorRT를 선택하는 설정은 없다.
이번에 TensorRT로 실행한 메인 구성은 **YOLO TensorRT + metadata CUDA + InSwapper ONNX CUDA**다.
InSwapper TensorRT mixed와 YuNet TensorRT·ROI 합성을 쓰는 경로는
`experiments/trt_swap_client`에 별도로 존재한다. 실험 경로의 FPS를 메인 기능 성능으로 볼 수 없다.

## 환경과 코드

| 항목 | 측정 조건 |
| --- | --- |
| GPU / CPU | NVIDIA GeForce RTX 3090 / AMD Ryzen 9 7900, logical CPU 24개 |
| 서버 checkout | `/home/gorani/apps/new-inno-live-server-python` |
| branch / commit | `feat/face-metadata-swap/#37` / `c3910ec295d44fa4536e926e9f9dafb5ed67dad7` |
| 업데이트 | `git pull --ff-only`로 `4270fd2` → `c3910ec` |
| Python 환경 | `.venv-face-swap-lab/bin/python`, Python 3.14.4 |
| 라이브러리 | torch 2.13.0+cu130, CUDA build 13.0, TensorRT 11.1.0.106, ORT 1.24.4 |
| 영상 라이브러리 | Ultralytics 8.4.104, OpenCV 5.0.0, grpcio 1.81.1 |
| CPU thread 설정 | OpenCV 1 / torch 24 |
| 메인 detector | 새로 export한 static B1 FP16 640 engine, `B1-640-Q90-W5` manifest 검증 |
| 실험 detector | 기존 dynamic B1..B4 FP16 640 engine, 동일 YOLO checkpoint |

NVIDIA kernel driver 595.91.07과 NVML library 595.99의 불일치 때문에 `nvidia-smi`가
실패했다. CUDA tensor 연산, TensorRT engine 실행, 실제 모델 추론은 성공했다.
GPU utilization은 수집하지 못했으며 기존 다른 서버와 Jupyter 프로세스를 유지했다.
드라이버 변경이나 재부팅은 하지 않았다.

## 1. 순수 모델 forward

입력 tensor와 source latent를 먼저 준비하고 30회 warmup 후 200회 측정했다.
TensorRT·PyTorch는 해당 execution stream의 CUDA event, ONNX CUDA는 GPU synchronize를
포함한 wall time을 사용했다. 네트워크, JPEG, tracking, 얼굴 alignment, paste-back은 제외한다.
YOLO는 1×3×640×640, metadata는 B×3×224×224, InSwapper는 1×3×128×128이다.

| 모델과 실행 경로 | 평균 ms | p95 ms |
| --- | ---: | ---: |
| YOLO / TensorRT FP16 / B1 640 | 1.053 | 1.073 |
| metadata / CUDA FP32 / B1 224 | 1.961 | 2.038 |
| metadata / CUDA FP32 / B2 224 | 1.977 | 2.037 |
| metadata / CUDA FP32 / B4 224 | 1.958 | 1.988 |
| InSwapper / ONNX CUDA / host I/O | 16.709 | 16.967 |
| InSwapper / ONNX CUDA / GPU I/O binding | 16.706 | 16.802 |
| InSwapper / TensorRT FP32 / GPU forward | 14.053 | 14.195 |
| InSwapper / TensorRT mixed / GPU forward | 5.423 | 5.503 |
| InSwapper / TensorRT FP32 / host I/O | 14.133 | 14.425 |
| InSwapper / TensorRT mixed / host I/O | 5.529 | 5.723 |

mixed engine은 19개 Conv를 FP16으로 변환한 모델이며 I/O binding은 FP32다.
모든 layer를 FP16으로 실행한 engine이라는 뜻은 아니다. InSwapper GPU forward 기준으로
mixed는 ONNX CUDA 대비 약 3.08배 빠르다. 이 비율은 전체 영상 처리량 향상 비율이 아니다.

같은 aligned 얼굴과 latent 1쌍의 raw 출력 `(1, 3, 128, 128)`을 비교했다. 모두 finite였다.

| 출력 비교 | MAE (0..1 raw) | 최대 절대 오차 | MAE × 255 |
| --- | ---: | ---: | ---: |
| ONNX GPU I/O ↔ host I/O | 0.00000000 | 0.00000000 | 0.00000 |
| TensorRT FP32 ↔ ONNX CUDA | 0.00009859 | 0.00134325 | 0.02514 |
| TensorRT mixed ↔ ONNX CUDA | 0.00135656 | 0.01648998 | 0.34592 |

이 비교는 입력 1쌍에 대한 수치 확인이다. 다양한 얼굴·표정·움직임의 시각적 품질이나
identity 보존 검증은 포함하지 않는다. 원본 측정은 [model-components.json](2026-10-10-inswapper-gpu/model-components.json)에 있다.

## 2. 현재 메인 기능의 네트워크 제외 처리량

실제 `AiProcessorServicer._process_request`를 직접 호출했다. timed 구간에서는 socket/RPC를
사용하지 않았으며 원격 SSH와 파일 전송은 실행 밖에서 수행했다. JPEG Q90 입력을 미리 만들고
각 session에서 30프레임 warmup, 120프레임 측정했다. 입력 해상도와 관계없이 YOLO 입력은 640이다.

이 처리량에는 JPEG decode/encode, YOLO pre/forward/post, BoT-SORT, metadata 분류,
프리셋 선택, CPU YuNet, InSwapper ONNX CUDA, CPU full-frame warp/blend, 응답 생성이 포함된다.
`metadata_device=cuda:0`, `swap_provider=cuda`를 명시했고 실제 InSwapper·ArcFace session에서
CUDAExecutionProvider 활성화를 확인했다. 메인 설정 기본값은 둘 다 CPU다.
`anonymization_mode=1`, `output_mode=2`로 모든 프레임을 처리했다.

| 입력 | 얼굴 | session | 전체 FPS | session당 FPS | 평균 지연 ms | p95 ms | codec·queue 제외 계산 평균 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 640×360 | 1 | 1 | 34.88 | 34.88 | 28.65 | 29.74 | 27.10 |
| 1280×720 | 1 | 1 | 17.48 | 17.48 | 57.17 | 59.77 | 53.63 |
| 1920×1080 | 1 | 1 | 12.82 | 12.82 | 77.97 | 80.82 | 71.24 |
| 1280×720 | 2 | 1 | 11.65 | 11.65 | 85.78 | 89.95 | 81.45 |
| 1280×720 | 4 | 1 | 7.97 | 7.97 | 125.41 | 130.91 | 121.31 |
| 1280×720 | 1 | 4 | 23.72 | 5.93 | 168.08 | 177.01 | 48.89 |
| 1280×720 | 1 | 8 | 23.76 | 2.97 | 335.45 | 347.22 | 48.90 |

전체 FPS는 모든 session의 완료 프레임 수 / 측정 wall time이다. session당 FPS는 이를
session 수로 나눈 값이며 지연은 각 요청이 직접 호출에서 완료되기까지 걸린 시간이다.
마지막 열은 `runtime_total_ms + composition_direct_ms - jpeg_encode_ms`로 계산했다.
YOLO·tracking·객체 안정화 및 합성 작업을 포함하고 JPEG decode/encode와 worker 입장 대기,
상위 응답 생성은 제외한다. 중첩된 작업 시간을 합친 값이므로 역수를 실제 FPS로 해석하지 않는다.

총 2,040프레임, 2,520개의 얼굴 합성을 처리했고 전부 `swapped`였다. 오류와 blur fallback은 0개다.
metadata는 트랙별 30프레임마다 갱신했고 source embedding cache를 사용했다.
고정 합성 인물 1장의 tiled 입력이며 얼굴 1·2·4개를 실제 detector/metadata로 확인했다.
whitelist는 비어 있고 AdaFace inference 호출은 0회였다.

720p·얼굴 1개·session 1의 평균 단계 시간:

| 단계 | 평균 ms | 범위 |
| --- | ---: | --- |
| YOLO inference | 3.064 | pre/post 포함 |
| BoT-SORT tracking | 3.436 | main runtime |
| 얼굴 합성 renderer | 41.476 | YuNet·InSwapper·alignment·blend 포함 |
| InSwapper ONNX CUDA run | 17.285 | renderer 안에 포함 |
| YuNet CPU detect | 13.511 | renderer 안에 포함 |
| metadata predict | 0.158 | 갱신 없는 프레임도 포함한 프레임당 평균 |
| JPEG encode | 1.190 | 합성 결과 출력 |

단계 시간이 서로 포함되므로 합해서 전체 지연으로 계산하면 안 된다. renderer에서
InSwapper와 YuNet 등을 뺀 약 10.7 ms는 alignment·full-frame warp/blend 등을 포함한 잔여 시간이다.
이 값은 독립적으로 계측한 paste-back 시간은 아니다. 프로파일상 메인에서 YOLO보다
YuNet과 InSwapper 및 CPU 합성이 더 큰 비용을 차지한다.

모델 로드 약 1.95초와 초기 source embedding 계산은 warmup에 포함해 위 FPS에서 제외했다.
추가 실제 gRPC loopback face_swap smoke에서 `swapped`와 JPEG 응답을 확인했다.
기존 blur 모드의 loopback B1 gate 120프레임도 전 항목 통과했다
(result FPS 36.21, server_total 평균 27.50 / p95 28.65 ms).
이 gate의 RTT와 전송 FPS는 모델 성능 표에 합치지 않았다.

원본 프레임별 timing·identity·fallback 기록은
[main-feature.json](2026-10-10-inswapper-gpu/main-feature.json)에 있다.

## 3. 별도 TensorRT 실험 경로

기존 `experiments.trt_swap_client.bench_lab`을 그대로 실행했다.
1280×720 NumPy frame, InSwapper TensorRT mixed, YuNet TensorRT ROI alignment,
CPU ROI paste, YOLO dynamic B1..B4, batch wait 3 ms, max queue 16 조건이다.
각 얼굴 수 조건의 warmup은 30프레임이며 측정은 session당 120프레임이다.
네트워크·JPEG·WebRTC·WS·HLS 출력은 측정 경로에 포함하지 않는다.

| 얼굴 | session | 전체 FPS | session당 FPS | p50 ms | p95 ms | 얼굴 합성/초 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 54.88 | 54.88 | 18.17 | 18.65 | 54.88 |
| 1 | 4 | 90.38 | 22.59 | 44.03 | 47.02 | 90.38 |
| 2 | 1 | 33.54 | 33.54 | 29.79 | 30.52 | 67.08 |
| 2 | 4 | 50.10 | 12.53 | 79.87 | 81.54 | 100.19 |
| 4 | 1 | 26.92 | 26.92 | 37.22 | 38.11 | 107.66 |
| 4 | 4 | 32.21 | 8.05 | 124.06 | 126.25 | 128.82 |

총 1,800프레임, 4,200개 얼굴 합성, drop 0개였다. 모든 조건에서 요청한 얼굴 수가 합성됐다.
이 경로는 한 source identity cache를 사용하며 메인의 metadata·프리셋 정책을 수행하지 않는다.
YuNet/ROI·CPU 병렬 합성·detector batching·codec 조건도 메인과 다르므로
두 경로의 FPS 차이를 TensorRT 단독 효과로 해석할 수 없다.

원본은 [tensorrt-lab.json](2026-10-10-inswapper-gpu/tensorrt-lab.json)에 있다.

## 측정 범위와 배포 상태

- 고정 synthetic portrait 반복 측정이며 실시간 카메라, 움직임, 다양한 사람·표정·조명은 검증하지 않았다.
- whitelist/AdaFace 비용과 실서비스 경쟁 부하, 긴 시간의 발열·메모리·지연 안정성은 별도 측정이 필요하다.
- 서버 코드는 pull했지만 측정용 서버는 종료했다. 기존 서비스는 그대로 유지했다.
- remote catalog 560개 이미지가 없어서 필요한 preset 5개를 임시 경로로 복사했다.
  ArcFace도 기본 경로 대신 기존 `~/.insightface/models/buffalo_l/w600k_r50.onnx`를 사용했다.
  전체 preset 설치와 상시 메인 TensorRT face_swap 배포가 완료된 상태는 아니다.
- 이번 작업은 애플리케이션 코드 수정·commit·push·PR 생성 없이 측정과 문서 기록만 수행했다.

## 근거 파일과 재현

같은 폴더의 `benchmark_current_feature.py`, `benchmark_model_components.py`는 실제 실행한
계측 helper다. app 구현은 수정하지 않고 인스턴스의 timing wrapper만 추가했다.
helper의 `ROOT`는 위 서버 checkout, components의 `SCRATCH`는 아래 임시 경로에 고정되어 있다.
다른 경로로 재현할 때 해당 상수를 바꿔야 한다. 입력 SHA-256, 실제 engine binding,
engine manifest와 버전은 JSON에 기록했다. engine과 얼굴 PNG 원본은 보고서에 넣지 않았다.

| artifact | SHA-256 |
| --- | --- |
| YOLO checkpoint | `307e9b5895654d25d264903451abc4acad9ee30f5e2f2bf64af164a9f46bc115` |
| 새 YOLO B1 engine | `7f0bb9eaa0b2f1d92c44855d20263adf4787d29a6abd865720fde8b28f726778` |
| InSwapper 원본 ONNX | `e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af` |
| InSwapper TensorRT FP32 | `7e2ea6378c808b5f7c1faada207927970538386c5c6310e1e1284c1787d3e40c` |
| InSwapper TensorRT mixed | `460046cf698437d5dae5eed8b72d33ff703e57c5b0667e03a0adc88ff4143af9` |

`provenance.json`에는 YOLO B4·YuNet engine hash/manifest와 실행 전 서비스·Git 상태를,
`best_b1.engine.json`에는 export 도구가 생성한 B1 manifest를 보존했다.
`required-presets.txt`는 필요한 5개 이미지와 catalog의 상대 경로다. 이 파일 목록을 기준으로
로컬 `config/`에서 임시 경로로 복사한 뒤, 임시 경로에 `best.pt` symlink도 만들어야 한다.
실행 전 부족한 aiortc 의존성을 `extra-deps`에만 `pip --target`으로 설치했다.

실제 실행 명령 (작업 디렉터리는 서버 checkout; 재실행 시 scratch를 먼저 준비):

```bash
.venv-face-swap-lab/bin/python -m scripts.export_tensorrt \
  --checkpoint models/best.pt --output /tmp/innolive-bench-20261010-IbBR9b/best_b1.engine \
  --device 0 --workspace 2

PYTHONPATH=/tmp/innolive-bench-20261010-IbBR9b/extra-deps \
.venv-face-swap-lab/bin/python /tmp/innolive-bench-20261010-IbBR9b/benchmark_current_feature.py \
  --scratch /tmp/innolive-bench-20261010-IbBR9b \
  --source /tmp/innolive-bench-20261010-IbBR9b/face_presets/images/g0_a1_gl0_e0_id0.png \
  --output /tmp/innolive-bench-20261010-IbBR9b/main-feature.json --warmup 30 --frames 120

PYTHONPATH=/tmp/innolive-bench-20261010-IbBR9b/extra-deps \
.venv-face-swap-lab/bin/python /tmp/innolive-bench-20261010-IbBR9b/benchmark_model_components.py

PYTHONPATH=/tmp/innolive-bench-20261010-IbBR9b/extra-deps \
.venv-face-swap-lab/bin/python -m experiments.trt_swap_client.bench_lab \
  --detector models/best_swap_b4.engine --swapper models/face_swap/inswapper_128.onnx \
  --swapper-engine models/face_swap/inswapper_128_trt11_mixed.engine \
  --source /tmp/innolive-bench-20261010-IbBR9b/face_presets/images/g0_a1_gl0_e0_id0.png \
  --target-aligner yunet_roi --faces 1 2 4 --sessions 1 4 --warmup 30 --frames-per-session 120 \
  --hls-dir /tmp/innolive-bench-20261010-IbBR9b/hls \
  --swapper-trt-cache /tmp/innolive-bench-20261010-IbBR9b/trt-cache \
  --output /tmp/innolive-bench-20261010-IbBR9b/tensorrt-lab.json
```

## 정리 결과

측정용 프로세스를 종료하고 임시 경로를 삭제했다. 확인 결과는 `cleanup.json`에 기록했다.
서버의 기존 모델·engine·가상환경 및 사전에 있던 untracked 파일은 보존하고,
계측 helper·JSON·로그·문서만 양쪽 저장소의 reports에 남긴다. Tailscale은 사용자가
활성화한 연결 상태를 유지한다. Ultralytics import 중 기존 settings version을 기본값으로
reset한다는 라이브러리 경고가 출력됐으며 설정 파일의 이전 내용을 확보하지 못해 복원하지 않았다.
