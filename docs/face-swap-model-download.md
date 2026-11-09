# 얼굴 합성 모델 다운로드와 설치

> 현재 탐지 backend는 RF-DETR-Seg Small입니다. 새 설치·엔진·성능은 [RF-DETR 전환 가이드](rfdetr-migration.md)를 따르세요. 아래 과거 측정의 YOLO 수치는 당시 모델 기준입니다.

현재 GPU 실행 구성은 **RF-DETR-Seg Small 탐지 + MobileFSGAN K22741 합성**이다.
일반 CLI의 합성 기본값은 InSwapper이며, MobileFSGAN은 전용 launcher 또는
`--face-swap-backend mobilefsgan`으로 선택한다. MobileFaceSwap 224도 선택할 수 있다.
아래 작업은 **AI 서버의 저장소 루트**에서 실행한다. Go 서버는 [gRPC 요청 필드](go-face-swap-grpc.md)만 설정한다.
서버는 모델을 자동 다운로드하지 않는다.

## 팀 모델·프리셋 다운로드

**[Google Drive 배포 폴더 열기](https://drive.google.com/drive/folders/1B4ZWbUPJp-t6mzJpSYqhkOxTRM1eH0i1?usp=sharing)**

2026-11-09 기준 폴더에 아래 4개 파일이 있다. 파일별 공유 권한은 링크가 있는 사용자에게
읽기를 허용한다. 파일을 다운로드해 저장소의 `.runtime/downloads/`에 모은다.

| 다운로드 | 용도 | 크기 | 설치 위치 |
| --- | --- | ---: | --- |
| [checkpoint_best_ema.pth](https://drive.google.com/file/d/1qEBO8kzcsolx1wgBewsX0Wskt_gGrajO/view?usp=drivesdk) | RF-DETR-Seg Small 얼굴·번호판 탐지/분할 | 130.2 MiB | `models/rfdetr_seg_s.pth`로 이름 변경 |
| [mobilefsgan_k22741_inference_20261108.tar.gz](https://drive.google.com/file/d/1DQic3xT-CpkvLvvvc_fHFyWJDzqesciN/view?usp=drivesdk) | MobileFSGAN G·C, YuNet, manifest·검증 파일 | 30.8 MiB | `models/face_swap/mobilefsgan/`에 압축 해제 |
| [face_presets.zip](https://drive.google.com/file/d/1ebHHPNoxo1qUbSdAlltjV9OvpdadPOHT/view?usp=drivesdk) | 합성용 가상 얼굴 560장과 데이터 manifest | 1,041.0 MiB | `config/face_presets/` |
| [face_presets.json](https://drive.google.com/file/d/1QGIsh7y3KIvrub8DeskaTM0vu1kMic7-/view?usp=drivesdk) | 위 이미지의 속성·identity·해시 카탈로그 | 184.0 KiB | `config/face_presets.json` |

RF-DETR와 MobileFSGAN 가중치·프리셋 이미지는 Git에 포함하지 않는다.
`face_presets.json`은 저장소에도 있지만, 이미지를 교체할 때는 같은 배포본의 카탈로그를 사용한다.
MobileFSGAN 압축파일은 원본 ONNX와 엔진을 함께 제공한다. 최적화 엔진은 배포 대상 GPU에서
[CUDA 실행 가이드](mobilefsgan-cuda.md)의 명령으로 빌드한다.

검증에 사용한 로컬 배포본의 SHA-256은 다음과 같다. 다운로드한 파일이 모두 `OK`인지
확인한 뒤 설치한다. macOS에서는 `sha256sum -c` 대신 `shasum -a 256 -c`를 사용할 수 있다.

```bash
sha256sum -c <<'EOF'
a5da6cb2a53ff001215149bb941678a27b02acd1dba6521a419fea30defdb2cf  .runtime/downloads/checkpoint_best_ema.pth
31070bc1ac87d6545fd945e8a9ad9e0ecd533a8f8f40da5c5be59b8f8d8ec84a  .runtime/downloads/mobilefsgan_k22741_inference_20261108.tar.gz
cc37e6d96de73c9c3bc892edfa69c13a2e601805b0a4eeb8c96d5b02572e6547  .runtime/downloads/face_presets.zip
799e042fc40421ec70ff04f32e49bb55368203d0f2805cfeef8b538e0222166d  .runtime/downloads/face_presets.json
EOF
```

아래는 신규 설치 기준이다. 기존 설치가 있다면 해당 모델·프리셋을 별도로 보관한 뒤 적용한다.
ZIP 안에 이미 `face_presets/` 폴더가 있으므로 `config/`에 푼다.

```bash
mkdir -p models/face_swap/mobilefsgan config
cp .runtime/downloads/checkpoint_best_ema.pth models/rfdetr_seg_s.pth
tar -xzf .runtime/downloads/mobilefsgan_k22741_inference_20261108.tar.gz \
  --strip-components=1 -C models/face_swap/mobilefsgan
python3 models/face_swap/mobilefsgan/verify_files.py
unzip -q .runtime/downloads/face_presets.zip 'face_presets/*' -d config
cp .runtime/downloads/face_presets.json config/face_presets.json
git lfs pull --include='models/face_metadata_extracter.pt'
```

프리셋 ZIP의 로컬 배포본에서 카탈로그가 가리키는 560장의 존재와 SHA-256 일치를 확인했다.
추가 whitelist용 AdaFace·YuNet은 [공식 모델 다운로드 안내](../models/README.md)를 따른다.
의존성 설치·TensorRT 빌드·실행은 [MobileFSGAN CUDA 가이드](mobilefsgan-cuda.md)를 따른다.

## 준비할 파일

| 저장 위치 | 용도 | 확보 방법 |
| --- | --- | --- |
| `models/rfdetr_seg_s.pth` | 얼굴·번호판 탐지 | 위 Drive의 `checkpoint_best_ema.pth`를 다운로드하고 이름 변경 |
| `models/face_metadata_extracter.pt` | 성별·나이·안경·표정 분석 | Git LFS에 포함, 최신 `main`에서 `git lfs pull`로 확보 |
| `models/face_swap/mobilefsgan/` | MobileFSGAN 얼굴 합성 | 위 Drive의 MobileFSGAN 번들을 압축 해제 |
| `models/face_swap/inswapper_128.onnx` | InSwapper 얼굴 합성 | 아래 공식 다운로드 명령 |
| `models/face_swap/w600k_r50.onnx` | InSwapper source identity 추출 | 공식 `buffalo_l.zip`에서 추출 |
| `models/mobilefaceswap/MobileFaceSwap_224.pdparams` | MobileFaceSwap 얼굴 합성 | [공식 번들 다운로드·설치](../models/README.md#mobilefaceswap-224) |
| `models/mobilefaceswap/arcface.pdparams` | MobileFaceSwap source identity·특징 추출 | 같은 공식 번들에서 추출 |
| `models/mobilefaceswap/landmarks/scrfd_10g_bnkps.onnx` | MobileFaceSwap 얼굴 정렬 | 같은 공식 번들에서 추출 |
| `models/face_detection_yunet_2023mar.onnx` | 얼굴 위치와 landmark 정렬 | 아래 OpenCV 다운로드 명령 |
| `config/face_presets.json`과 `config/face_presets/images/*.png` | 합성할 가짜 얼굴 목록과 이미지 | 위 Drive의 `face_presets.zip`과 `face_presets.json` 설치 |

기존 whitelist용 AdaFace 모델은 [models/README.md](../models/README.md)의 준비 방법을 따른다.

합성 모델은 선택한 backend에 필요한 파일만 설치한다. 탐지·metadata·프리셋과
whitelist용 AdaFace·YuNet은 공통 준비 항목이다.

## InSwapper 외부 모델 다운로드

InSwapper와 호환되는 ArcFace는 **`buffalo_l`의 `w600k_r50.onnx`**다.
공식 [InSwapper 안내](https://github.com/deepinsight/insightface/blob/master/examples/in_swapper/README.md)와
[model-zoo release](https://github.com/deepinsight/insightface/releases/tag/model-zoo)의 파일을 사용한다.
YuNet은 [OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)의 2023mar 모델을 사용한다.

```bash
mkdir -p models/face_swap
face_model_tmp=$(mktemp -d)

curl -fL --retry 3 \
  https://github.com/deepinsight/insightface/releases/download/model-zoo/inswapper_128.onnx \
  -o models/face_swap/inswapper_128.onnx

curl -fL --retry 3 \
  https://github.com/deepinsight/insightface/releases/download/model-zoo/buffalo_l.zip \
  -o "$face_model_tmp/buffalo_l.zip"
unzip -p "$face_model_tmp/buffalo_l.zip" w600k_r50.onnx \
  > models/face_swap/w600k_r50.onnx

curl -fL --retry 3 \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
  -o models/face_detection_yunet_2023mar.onnx

rm "$face_model_tmp/buffalo_l.zip"
rmdir "$face_model_tmp"
```

다운로드 후 아래 명령에서 모든 파일이 `OK`인지 확인한다. Linux에서는 `shasum -a 256 -c` 대신 `sha256sum -c`도 사용할 수 있다.
hash는 현재 연동 모델 기준이며 InSwapper 값은 공식 release의 SHA-256과도 일치한다.

```bash
shasum -a 256 -c <<'EOF'
e4a3f08c753cb72d04e10aa0f7dbe3deebbf39567d4ead6dce08e98aa49e16af  models/face_swap/inswapper_128.onnx
4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43  models/face_swap/w600k_r50.onnx
8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4  models/face_detection_yunet_2023mar.onnx
be9b42f67dbab87c6a6a73161995ddb43b73e64c6203748f3ab3222edef4448a  models/face_metadata_extracter.pt
EOF
```

## MobileFaceSwap 모델 다운로드

[models/README.md의 MobileFaceSwap 224 설치 명령](../models/README.md#mobilefaceswap-224)으로
공식 `checkpoints.tar`를 다운로드하고 필요한 세 파일을 설치·검증한다.
InSwapper용 ONNX 두 개는 필요하지 않으며, 탐지는 공통 RF-DETR 모델을 사용한다.
MPS 의존성·실행은 [Mac 가이드](mobilefaceswap-mac.md), CUDA 의존성·캐시 준비·실행은
[CUDA 가이드](mobilefaceswap-cuda.md)를 따른다.

## 프리셋 이미지와 Python 의존성 설치

위 Drive의 ZIP과 JSON을 설치했다면 이 절의 importer는 실행하지 않아도 된다.
원본 데이터셋을 import할 때는 `dataset_manifest.csv`와 `images/`가 들어 있는
**현재 카탈로그와 같은 배포본**을 사용한다.
현재 카탈로그는 `natural_faces_v2` 560장을 참조하며, 이전 `synthetic_faces` 이미지와 해시가 다르다.
프리셋 PNG는 Git에 포함되지 않으므로 팀 배포본을 별도로 받아야 한다.

```bash
# InSwapper CPU 실행용 기본 설치 (MobileFaceSwap은 위 환경별 가이드 사용)
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-face-swap.txt

# 받은 배포본의 실제 경로로 변경. 기존 이미지 세트는 먼저 별도 보존.
.venv/bin/python scripts/import_face_presets.py --source /path/to/natural_faces_v2
```

importer가 560장의 이미지 hash와 조합을 검증하고 `config/face_presets/`에 복사한 뒤 catalog를 갱신한다.
원본 배포본은 수정하지 않는다. 모델이나 프리셋을 교체한 뒤에는 AI 서버를 재시작한다.

## 서버 실행과 확인

```bash
# 기본 InSwapper를 CPU에서 동작 확인
.venv/bin/python ai_processor_server.py --backend pytorch --device cpu
```

NVIDIA에서는 CPU용 `onnxruntime` 대신 호스트 CUDA/cuDNN과 호환되는 `onnxruntime-gpu`를 설치한다.
버전은 [ONNX Runtime CUDA 호환표](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)를 따른다.
서버 시작 옵션에 `--face-swap-provider cuda --face-metadata-device cuda:0`를 추가한다.
RF-DETR backend 설정은 [전환 가이드](rfdetr-migration.md)를 따른다.

별도 터미널에서 테스트 이미지를 보내 결과 영상과 metadata를 확인한다.

```bash
.venv/bin/python scripts/face_preset_client.py \
  --target 127.0.0.1:50051 --image /path/to/face.jpg \
  --mode face_swap --frames 3 --headless --output-dir face_preset_client_output
```

얼굴별 상태가 `swapped`이면 합성 성공이다. `blur_fallback`이면 `fallback_reason`을 확인한다.
모델·프리셋 누락이나 합성 실패 시 해당 얼굴은 blur 처리된다. 상세 설정은
[서버 가이드](face-preset-anonymization.md)를 참고한다.

InsightFace 공개 모델은 비상업 연구용 조건으로 배포되며, 상업적 사용에는 별도 모델 라이선스가 필요하다.
조건은 [공식 모델 라이선스 안내](https://github.com/deepinsight/insightface#license)를 확인한다.
