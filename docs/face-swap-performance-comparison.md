# 얼굴 합성 성능 비교: InSwapper vs MobileFSGAN

측정일: **2026-11-09 KST** · 장비: **RTX 3090 24GB** · 대상: InnoLive의 기존 InSwapper CUDA 경로와 최적화된 MobileFSGAN K22741 경로

## PPT용 요약

> 기존 InSwapper CUDA 처리 경로 대비, 얼굴 1개 합성 지연 **약 84% 감소**(16.7 → 2.7ms), FHD 4얼굴 처리량 **약 5.6배 향상**(5.2 → 28.9 FPS).

※ 합성 지연은 정렬된 얼굴 입력의 p50이며 탐지·정렬·화면 합성을 제외한다. FPS는 이들을 포함한 단일 세션 gRPC loopback 전체 처리량이다.

## 구현 비교

| 항목 | 기존 InSwapper | MobileFSGAN K22741 |
|---|---|---|
| 합성 모델 실행 | ONNX Runtime CUDA, FP32 입출력 | TensorRT mixed FP16, FP32 입출력 |
| 얼굴 합성 native 해상도 | 128 × 128 | 256 × 256 |
| 다얼굴 합성 | 얼굴별 B1 순차 실행 | 최대 8얼굴 배치 + CUDA Graph |
| Source identity encoder | ArcFace w600k_r50 | 자체 MobileFaceNet C_04000 |
| Target landmark | CPU YuNet, 얼굴 ROI별 검출 | TensorRT YuNet, 프레임 결과 공유 및 ROI 재검출 |
| 원본 영상에 합성 | CPU 전체 프레임 warp/blend | CPU 얼굴 polygon ROI warp/blend |
| Source embedding | 최대 64개 캐시 | 최대 640개 캐시 |

합성 얼굴의 가로·세로 출력 크기는 2배이며, 전체 영상 입출력 해상도는 두 경로에서 같다. 출력 픽셀 수 증가를 자연스러움·정체성 품질 향상 배수로 해석하지 않는다.

## 합성 단계 지연

같은 source/target 인물 조합을 모델 고유 방식으로 정렬했다. Source embedding은 캐시하고, 정렬된 BGR crop의 전처리·GPU 전송·모델 실행·CPU 결과 반환까지 측정했다.
얼굴 검출·alignment 계산·paste-back·영상 변환·네트워크는 제외했다. 20회 예열 후 두 경로를 번갈아 80회씩 측정했다.

| 한 번에 처리한 얼굴 | InSwapper p50 / p95 | MobileFSGAN p50 / p95 | p50 기준 속도 향상 |
|---|---:|---:|---:|
| 1개 | 16.71 / 17.09ms | 2.71 / 2.82ms | 6.16배 |
| 4개 | 68.31 / 70.09ms | 8.17 / 8.36ms | 8.36배 |
| 8개 | 137.21 / 142.45ms | 15.83 / 16.29ms | 8.67배 |

**얼굴 1개는 약 16.7ms → 2.7ms, 6.2배**다. 4·8얼굴 행은 모든 얼굴을 처리한 합계 시간이며 얼굴당 지연이 아니다. 이 시간의 역수를 전체 영상 FPS로 사용하지 않는다.

## 전체 영상 파이프라인 FPS

YOLO 탐지·BoT-SORT 추적·metadata·프리셋 선택·얼굴 정렬·합성·YUV 변환·gRPC 왕복을 포함한다. 동일한 30프레임 움직이는 프리셋 grid를 반복하고, 예열 5초 뒤 15초간 측정했다. 한 세션, window 1, raw YUV420p, 속도 제한 없음 조건이다.

| 영상 해상도 | 얼굴 수 | InSwapper | MobileFSGAN | 처리량 향상 |
|---|---:|---:|---:|---:|
| 640 × 480 | 1 | 37.2 FPS | 87.1 FPS | 2.34배 |
| 640 × 480 | 4 | 11.5 FPS | 48.1 FPS | 4.17배 |
| 640 × 480 | 8 | 5.9 FPS | 29.5 FPS | 4.98배 |
| 1920 × 1080 | 1 | 17.9 FPS | 52.3 FPS | 2.93배 |
| 1920 × 1080 | 4 | 5.2 FPS | 28.9 FPS | 5.55배 |
| 1920 × 1080 | 8 | 2.8 FPS | 19.9 FPS | 7.12배 |

## FHD 전체 지연

서버 지연은 서버 내부 처리 시간, RTT는 클라이언트 전송 직전부터 응답 수신까지의 시간이다. 각 값은 **p50 / p95**다.

| 얼굴 수 | InSwapper 서버 | MobileFSGAN 서버 | InSwapper RTT | MobileFSGAN RTT |
|---|---:|---:|---:|---:|
| 1 | 51.37 / 53.90ms | 14.80 / 16.68ms | 55.54 / 58.21ms | 18.69 / 20.75ms |
| 4 | 185.35 / 188.53ms | 27.64 / 31.53ms | 192.25 / 195.54ms | 33.33 / 37.62ms |
| 8 | 350.45 / 354.78ms | 42.43 / 49.07ms | 359.23 / 364.82ms | 48.43 / 56.17ms |

12개 전체 파이프라인 조건에서 **5,195프레임, 15,521개 얼굴 합성**을 확인했고, 모든 측정 응답에서 요청한 얼굴 수가 검출·합성됐다. 오류·blur fallback은 0건이었다.

## 비교 조건과 해석 범위

- 두 경로 모두 GPU를 사용했다. 같은 GPU·YOLO B1 TensorRT 엔진·metadata 모델·프리셋 카탈로그·영상 입력·전송 형식·CPU thread 설정을 사용하고 순차적으로 부하를 주었다. 입력 JPEG 기록의 SHA-256도 backend 간 일치했다.
- 전체 파이프라인의 세션별 source identity 선택은 기존 정책을 유지했다. 합성 단계 비교에서는 동일 source/target 조합을 고정했다.
- **모델 교체와 실행 엔진·정렬·배치·합성 파이프라인 최적화를 합친 효과**다. 두 모델을 같은 TensorRT 조건으로 최적화한 아키텍처 단독 비교는 아니다.
- 2026-10-10의 별도 InSwapper TensorRT mixed 실험에는 host I/O 평균 5.53ms 기록이 있다. 당시 TensorRT 버전·입력·측정 조건이 달라 이번 표와 개선 배수에는 섞지 않았다.
- 이전의 480p 4얼굴 22.7→48.1 FPS는 MobileFSGAN 자체 최적화 전후 값이다. InSwapper 비교 기준으로 사용하지 않았다.
- 고정 프리셋의 단기 loopback 측정이다. 실제 카메라·외부 회선·장시간 부하·다양한 군중·화이트리스트 인식 비용은 비교하지 않았다. Mac 성능은 측정하지 않았다.
- 두 모델의 시각적 자연스러움·identity 보존·익명화 품질 우열은 이 속도 측정으로 판단하지 않는다.

## 환경과 재현

- Linux x86_64, NVIDIA driver 595.99.02, 전력 한도 300W
- Python 3.14.4, PyTorch 2.14.0+cu126, OpenCV 5.0.0
- MobileFSGAN: TensorRT 11.4.0.106, mixed FP16 G + FP32 C/YuNet
- InSwapper: ONNX Runtime GPU 1.24.4, CUDAExecutionProvider 활성화 확인
- OpenCV / OMP / OpenBLAS thread 1. 기존 InSwapper adapter의 ORT intra-op thread 2 유지
- 기본 환경의 ORT 1.29 CUDA 13 의존성 문제로 CPU fallback이 발생한 초기 실행은 측정에서 제외. 1.24.4를 비교 전용 경로에 격리 설치한 뒤 GPU를 확인하고 측정
- 기존 서비스는 유지했고 임시 InSwapper 비교 서버는 측정 후 종료

합성 단계 재현:

```bash
# 비교 전용 ORT 설치. 기존 실행 환경의 패키지는 변경하지 않음.
.venv/bin/python -m pip install --target .runtime/inswapper-ort-cu12 \
  --no-deps onnxruntime-gpu==1.24.4

PYTHONPATH="$PWD/.runtime/inswapper-ort-cu12" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OPENCV_FOR_THREADS_NUM=1 \
  .venv/bin/python scripts/benchmark_swap_backends.py \
  --inswapper /path/to/inswapper_128.onnx \
  --arcface /path/to/w600k_r50.onnx \
  --engine-dir .runtime/mobilefsgan-fast \
  --warmup 20 --repeats 80 --output reports/swap-model-comparison
```

전체 파이프라인은 각 backend 서버에 같은 명령을 실행한다:

```bash
.venv/bin/python scripts/benchmark_face_capacity.py \
  --target 127.0.0.1:50072 --cases 1x1,4x1,8x1 \
  --width 1920 --height 1080 --transport raw --window 1 \
  --warmup 5 --seconds 15 --output reports/swap-pipeline-comparison
```

이번 비교의 포트는 InSwapper `50072`, MobileFSGAN `50071`이었다.

근거 파일은 로컬 `reports/inswapper-comparison-20261109/` 및 GPU 서버의
`/home/gorani/apps/innolive-mobilefsgan-100/reports/inswapper-comparison-20261109/`에 보관했다.
`models-final/result.json`이 최종 합성 단계 측정이며, `grpc-*`에 전체 파이프라인 JSON과 입출력 이미지, `provenance.json`에 환경·모델·코드 해시가 있다.

관련 문서: [MobileFSGAN 실행 및 측정](mobilefsgan-cuda.md), [GPU·CPU 자원 분석](mobilefsgan-resource-audit.md), [ONNX Runtime CUDA 호환 조건](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
