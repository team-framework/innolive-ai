# 얼굴 합성 모델 다운로드와 설치

[PR #38](https://github.com/team-framework/innolive-ai/pull/38)이 병합된 `main`의 gRPC 얼굴 합성 기능 기준이다.
아래 작업은 **AI 서버의 저장소 루트**에서 실행한다. Go 서버는 [gRPC 요청 필드](go-face-swap-grpc.md)만 설정한다.
서버는 모델을 자동 다운로드하지 않는다.

## 준비할 파일

| 저장 위치 | 용도 | 확보 방법 |
| --- | --- | --- |
| `models/best.pt` | 얼굴·번호판 탐지 | Git에 포함, 최신 `main` checkout으로 확보 |
| `models/face_metadata_extracter.pt` | 성별·나이·안경·표정 분석 | Git에 포함, 최신 `main` checkout으로 확보 |
| `models/face_swap/inswapper_128.onnx` | 얼굴 합성 | 아래 공식 다운로드 명령 |
| `models/face_swap/w600k_r50.onnx` | 합성할 얼굴의 identity 추출 | 공식 `buffalo_l.zip`에서 추출 |
| `models/face_detection_yunet_2023mar.onnx` | 얼굴 위치와 landmark 정렬 | 아래 OpenCV 다운로드 명령 |
| `config/face_presets.json`과 `config/face_presets/images/*.png` | 합성할 가짜 얼굴 목록과 이미지 | catalog는 Git에 포함, 이미지는 팀의 `synthetic_faces` 배포본 설치 |

기존 whitelist용 AdaFace 모델은 [models/README.md](../models/README.md)의 준비 방법을 따른다.

## 외부 모델 다운로드

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

## 프리셋 이미지와 Python 의존성 설치

팀에서 `dataset_manifest.csv`와 `images/`가 들어 있는 **현재 카탈로그와 같은 배포본**을 받는다.
현재 카탈로그는 `natural_faces_v2` 560장을 참조하며, 이전 `synthetic_faces` 이미지와 해시가 다르다.
프리셋 PNG는 Git에 포함되지 않으므로 팀 배포본을 별도로 받아야 한다.

```bash
# CPU 실행용 기본 설치
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-face-swap.txt

# 받은 배포본의 실제 경로로 변경. 기존 이미지 세트는 먼저 별도 보존.
.venv/bin/python scripts/import_face_presets.py --source /path/to/natural_faces_v2
```

importer가 560장의 이미지 hash와 조합을 검증하고 `config/face_presets/`에 복사한 뒤 catalog를 갱신한다.
원본 배포본은 수정하지 않는다. 모델이나 프리셋을 교체한 뒤에는 AI 서버를 재시작한다.

## 서버 실행과 확인

```bash
# CPU에서 동작 확인
.venv/bin/python ai_processor_server.py --backend pytorch --device cpu
```

NVIDIA에서는 CPU용 `onnxruntime` 대신 호스트 CUDA/cuDNN과 호환되는 `onnxruntime-gpu`를 설치한다.
버전은 [ONNX Runtime CUDA 호환표](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)를 따른다.
서버 시작 옵션에 `--face-swap-provider cuda --face-metadata-device cuda:0`를 추가한다.
YOLO backend 설정은 기존 배포 설정을 따른다.

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
