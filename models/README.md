# Runtime model files

`best.pt` is the serving YOLO segmentation checkpoint. The runtime uses `imgsz=640`
and expects class `0` as `face` and class `1` as `number_plate`; both classes are
included in the protected mosaic path, while only `face` enters AdaFace recognition.

The TensorRT, AdaFace, and YuNet binaries are deployment artifacts and are not
tracked by Git. Build the TensorRT engine on the target Linux x86_64 NVIDIA host
from `best.pt`; do not copy an engine built on another GPU or TensorRT stack.
The export command creates both `best_b1.engine` and its local validation manifest
`best_b1.engine.json` next to the checkpoint. The server expects these defaults:

| File | Source | SHA-256 used for smoke testing |
| --- | --- | --- |
| `adaface_vit_base_kprpe_webface12m.ckpt` | [Official CVLFace AdaFace ViT-Base KP-RPE WebFace12M checkpoint](https://huggingface.co/minchul/cvlface_adaface_vit_base_kprpe_webface12m) | `04b4bee1de7cefa9e97900f8449fca906d8afbab2029bd39cc5049d33e927ed9` |
| `adaface_ir18_casia.ckpt` | [Official AdaFace IR-18 CASIA checkpoint](https://drive.google.com/file/d/1BURBDplf2bXpmwOL1WVzqtaVmQl9NpPe/view) | `2be3042e2266b745824ba95dad4d09a0e4d9f67bd08fb0b79f5b7a605c48fd30` |
| `face_detection_yunet_2023mar.onnx` | [OpenCV Zoo YuNet model](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |

Download YuNet with:

```bash
curl -L --fail \
  -o models/face_detection_yunet_2023mar.onnx \
  https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
```

The default recognizer is pinned to CVLFace revision
`daefd5012d369588bd214fbaf4cc6b1d286e7066`. Download its backbone-only state
dictionary with:

```bash
curl -L --fail \
  -o models/adaface_vit_base_kprpe_webface12m.ckpt \
  https://huggingface.co/minchul/cvlface_adaface_vit_base_kprpe_webface12m/resolve/daefd5012d369588bd214fbaf4cc6b1d286e7066/pretrained_model/model.pt
shasum -a 256 models/adaface_vit_base_kprpe_webface12m.ckpt
```

Alternative architecture and artifact paths can be supplied with
`--adaface-architecture`, `--adaface-weights`, and `--adaface-detector`.

The legacy IR-18 CASIA checkpoint remains supported through
`--adaface-architecture ir18`; it is the smallest backbone and training-set
combination in the official model table. An official
[IR-18 WebFace4M checkpoint](https://drive.google.com/file/d/1J17_QW1Oq00EhSWObISnhWEYr2NNrg2y/view)
is architecture-compatible and can be evaluated through `--adaface-weights`.
Recalibrate the cosine threshold on deployment data before changing the
production default; do not lower it solely to improve apparent recall.

CVLFace source code is MIT-licensed. The publisher does not declare an SPDX
license for the WebFace12M weights and asks users to follow the training data
license. Review those rights separately before commercial deployment.

The server stores only normalized embeddings in memory. Enrollment images and
aligned face crops are not written to this directory.

## Experimental face metadata and swap

`face_metadata_extracter.pt` is the supplied MobileNetV3-Large attribute checkpoint
(SHA-256 `be9b42f67dbab87c6a6a73161995ddb43b73e64c6203748f3ab3222edef4448a`).
The serving adapter validates its gender/age/glasses/exp class order and reconstructs
the backbone and four heads. It is loaded only for explicitly requested experimental
metadata/swap modes. The training run snapshot verifies Hardswish heads and RGB ImageNet
normalization. Face boxes receive a 15% margin on each side, then aspect-preserving resize
and centered replicated-edge padding. A CPU FP32 comparison of 48 synthetic face inputs
matched the training implementation's input tensors and all four logits exactly.

InSwapper and ArcFace binaries, plus synthetic preset images, are deployment artifacts.
See [the preset pipeline guide](../docs/face-preset-anonymization.md) for artifact paths,
optional ONNX dependencies, the manifest, and the actual gRPC test client.
For official download commands and preset installation, see
[the model download guide](../docs/face-swap-model-download.md).

## MobileFaceSwap 224

[공식 사전학습 모델 다운로드: checkpoints.tar](https://drive.google.com/file/d/1ZIzGLDB15GRAZAbkfNR0hNWdgQpxeA_r/view)
([원본 프로젝트 안내](https://github.com/Seanseattle/MobileFaceSwap#readme)).
약 543 MiB의 TAR 파일이며, 내부 `checkpoints/`를 이 서버의
`models/mobilefaceswap/`에 맞춰 설치한다. `models/best.pt`와 metadata 모델은 그대로 사용한다.

| 저장 위치 (`models/` 기준) | 용도 | 다운로드 파일에서 확인한 SHA-256 |
| --- | --- | --- |
| `mobilefaceswap/MobileFaceSwap_224.pdparams` | 224px 얼굴 합성 | `ee51a18684d2862363deefde14220d7e7eb33b78b5b1840e1a76566a7f3508fb` |
| `mobilefaceswap/arcface.pdparams` | 합성할 얼굴의 identity·특징 추출 | `adc5d94075fc2f8ad93ba1040310b97f604843d03c82ecf3a4bdb86af658ca3b` |
| `mobilefaceswap/landmarks/scrfd_10g_bnkps.onnx` | 얼굴 검출·landmark 정렬 | `5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91` |

다음 명령은 **저장소 루트**에서 실행한다. 프로젝트의 `.venv`를 사용하며, 없으면
먼저 `python3 -m venv .venv`로 생성한다. `gdown`은 Google Drive 다운로드에만 필요하다.

```bash
.venv/bin/python -m pip install gdown
mkdir -p .runtime/downloads models/mobilefaceswap
.venv/bin/python -m gdown 1ZIzGLDB15GRAZAbkfNR0hNWdgQpxeA_r \
  --no-cookies --continue -O .runtime/downloads/checkpoints.tar
```

Drive 다운로드 제한이 발생하면 위 링크에서 직접 받아
`.runtime/downloads/checkpoints.tar`에 저장한 뒤 아래부터 실행한다.
전체 압축을 풀지 않고 런타임에 필요한 세 파일만 설치한다.
번들에 포함된 `landmarks/glintr100.onnx`는 현재 파이프라인에서 사용하지 않는다.

```bash
tar -xf .runtime/downloads/checkpoints.tar \
  -C models/mobilefaceswap --strip-components=1 \
  checkpoints/MobileFaceSwap_224.pdparams \
  checkpoints/arcface.pdparams \
  checkpoints/landmarks/scrfd_10g_bnkps.onnx

shasum -a 256 -c <<'EOF'
ee51a18684d2862363deefde14220d7e7eb33b78b5b1840e1a76566a7f3508fb  models/mobilefaceswap/MobileFaceSwap_224.pdparams
adc5d94075fc2f8ad93ba1040310b97f604843d03c82ecf3a4bdb86af658ca3b  models/mobilefaceswap/arcface.pdparams
5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91  models/mobilefaceswap/landmarks/scrfd_10g_bnkps.onnx
EOF
```

세 파일이 모두 `OK`여야 한다. Linux에서는 `sha256sum -c`도 사용할 수 있다.
이 번들에는 탐지·metadata 모델과 프리셋 이미지가 없으므로
공통 파일은 [모델·프리셋 준비 가이드](../docs/face-swap-model-download.md)를 따른다.
InSwapper용 `inswapper_128.onnx`와 `w600k_r50.onnx`는 MobileFaceSwap에서 사용하지 않는다.

의존성 설치와 실행은 환경별 가이드를 따른다.

- Apple Silicon MPS: [Mac 실행 가이드](../docs/mobilefaceswap-mac.md), `./scripts/run_mobilefaceswap.sh`
- NVIDIA CUDA: [CUDA 배포 가이드](../docs/mobilefaceswap-cuda.md), 캐시 사전 생성 후 `./scripts/run_mobilefaceswap_cuda.sh`

메인 서버에서는 `--face-swap-backend mobilefaceswap`으로 선택한다.
일반 서버의 기본 backend는 InSwapper이며 Go gRPC 요청·응답 계약은 유지된다.
모델과 생성 캐시는 Git에 추가하지 않는다.
