# Face Swap Lab

`face_swap_lab.py`는 production gRPC server와 분리된 macOS webcam prototype입니다.
왼쪽에는 webcam target, 오른쪽에는 처리 결과를 보여 주며 같은 frame을 1·4·16개의
독립 synthetic session이 각각 처리한 것처럼 반복해 compute load 변화를 표시합니다.
이는 **network/WebRTC capacity benchmark가 아니라 model compute contention 실험**입니다.

## 준비

source 기본 경로는 `/Users/gwondaehyeong/Documents/input.png`입니다.

```bash
python3 -m venv .venv-face-swap-lab
.venv-face-swap-lab/bin/pip install -r requirements-face-swap-lab.txt
```

생성형 face-swap model 없이 webcam·mask·부하 UI를 먼저 확인하려면 아래 command로 실행하고
기본값인 `Landmark mask mapping (YuNet + 106-point ONNX)`를 선택합니다. 이 모드는
매 frame YuNet으로 target face를 탐지하고 source의 얼굴 질감을 106개 landmark에 정렬한 뒤
feathered mask로 합성합니다. 표정·옆모습·가림을 생성하지는
않지만, 생성형 model보다 훨씬 가볍고 mask 경계와 frame-rate의 기준선으로 적합합니다.

아래 두 model asset이 있어야 합니다. 이 repository에는 YuNet이 포함되어 있고,
`2d106det.onnx`는 InsightFace `buffalo_l` bundle에서 별도로 준비합니다.

```text
models/face_detection_yunet_2023mar.onnx
~/.insightface/models/buffalo_l/2d106det.onnx
```

```bash
.venv-face-swap-lab/bin/python experiments/face_swap_lab.py
```

## InSwapper 128 실험

`InSwapper 128`을 쓰려면 사용 권한이 있는 model artifact를 아래 경로에 준비합니다.
가중치는 Git에 넣지 않습니다.

```text
models/face_swap/inswapper_128.onnx
~/.insightface/models/buffalo_l/
```

`buffalo_l`은 source embedding과 target face detection/landmark에 사용됩니다. 앱은
ONNX Runtime의 `CoreMLExecutionProvider`를 먼저 선택하고, 지원되지 않는 연산만 CPU로
fallback합니다. UI provider 표시가 `CoreML preferred`인지 확인하세요.

InsightFace model은 별도 라이선스가 있을 수 있으므로 자동 다운로드하지 않습니다.
source image와 webcam 대상에는 모두 사용 권한을 확보해야 합니다.

## 실험 순서

1. session `1`, `Landmark mask mapping`으로 camera·source·mask 품질·저장 버튼을 확인합니다.
2. session `1`, `InSwapper 128`으로 기본 결과와 frame p50/p95를 저장합니다.
3. 같은 장면에서 session `4`, `16`을 순서대로 선택합니다.
4. 큰 얼굴, 측면 얼굴, 안경/손/머리카락 가림 장면을 `Save pair`로 저장해 비교합니다.

`Ellipse mask mapping fallback (OpenCV)`은 YuNet 또는 landmark asset이 없을 때만 사용합니다.
`face_swap_lab_output/`은 Git에서 제외됩니다. 128px model은 큰 1080p 얼굴의 최종 품질
후보가 아니라 속도와 합성 경계의 기준선입니다.
