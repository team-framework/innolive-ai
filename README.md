# InnoLive AI gRPC Server

본 레포지토리는 InnoLive 프로젝트의 gRPC 기반 AI 서버입니다 (Python).
양방향 gRPC 프로토콜을 통해 클라이언트의 영상을 실시간으로 비식별화합니다.

현재 비식별화 파이프라인은 YOLO 세그멘테이션(`models/best.pt`,
`0: head`)과 BoT-SORT를 사용합니다. 각 스트림은 독립된 트래커를 가지며,
일시적인 검출 누락에는 예측 위치로 이전 마스크를 이동해 유지하므로 마스크가
프레임 사이에서 깜빡이지 않습니다.

주요 기술: Python, gRPC, Ultralytics YOLO, TensorRT, BoT-SORT, OpenCV

## 설치

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 -m grpc_tools.protoc --proto_path=./protos \
  --python_out=__generated__ --pyi_out=__generated__ --grpc_python_out=__generated__ \
  ./protos/ai_processor.proto
```

NVIDIA 서버에서는 해당 CUDA 버전에 맞는 PyTorch를 먼저 설치하는 것을
권장합니다. TensorRT 엔진은 GPU 아키텍처와 TensorRT 버전에 종속되므로 다른
장비에서 만든 엔진을 복사하지 말고 실제 배포 장비에서 생성해야 합니다.

## TensorRT 엔진 생성

고정 입력 크기, batch 1, 통합 NMS를 사용하는 FP16 엔진이 기본값입니다.
실시간 단일 프레임 처리에서 동적 shape보다 빠르고, 별도 calibration 없이
정확도를 안정적으로 유지합니다.

```bash
python3 scripts/export_tensorrt.py --device 0 --imgsz 640
```

생성된 `models/best.engine`이 `best.pt`보다 최신이면 서버가 자동으로 우선
사용합니다. INT8은 대표성이 있는 head 학습/검증 이미지가 있을 때만 생성하세요.

```bash
python3 scripts/export_tensorrt.py \
  --device 0 --int8 --data /path/to/data.yaml
```

## 실행

프로덕션 기본 모드는 head 마스크 비식별화입니다.

```bash
python3 ai_processor_server.py --host 0.0.0.0 --device 0
```

주요 설정은 CLI 또는 환경 변수로 조정할 수 있습니다.

| CLI | 환경 변수 | 기본값 | 설명 |
| --- | --- | --- | --- |
| `--model` | `AI_MODEL_PATH` | `models/best.pt` | `.engine` 또는 `.pt` 모델 |
| `--imgsz` | `AI_IMAGE_SIZE` | `640` | TensorRT 생성값과 동일해야 함 |
| `--confidence` | `AI_CONFIDENCE` | `0.10` | BoT-SORT 저신뢰도 복구 검출 포함 |
| `--mask-hold-frames` | `AI_MASK_HOLD_FRAMES` | `8` | 누락 시 마스크 유지 프레임 수 |
| `--jpeg-quality` | `AI_JPEG_QUALITY` | `90` | 출력 JPEG 품질 |
| `--no-tensorrt` | `AI_PREFER_TENSORRT=0` | 엔진 우선 | PyTorch 모델 강제 사용 |

BoT-SORT의 association, camera motion compensation, track buffer 설정은
`config/botsort.yaml`에서 조정합니다. `track_buffer`는 마스크 유지 프레임보다
크게 유지해야 누락 구간에서 칼만 예측 위치를 계속 받을 수 있습니다.

## gRPC 코드 생성

```bash
python3 -m grpc_tools.protoc --proto_path=./protos  \
 --python_out=__generated__ --pyi_out=__generated__ --grpc_python_out=__generated__ \
 ./protos/ai_processor.proto
```

호스트, 포트와 worker 수는 CLI 또는 `GRPC_HOST`, `GRPC_PORT`,
`GRPC_MAX_WORKERS` 환경 변수로 조정할 수 있습니다.
`--processing-mode passthrough`(또는 `GRPC_PROCESSING_MODE=passthrough`)를
사용하면 영상 변환 비용을 제외한 gRPC 통신 성능을 분리해 측정할 수 있습니다.

## 성능 테스트

전용 gRPC 부하 도구인 ghz를 사용하는 재현 가능한 성능 테스트 환경은
[`benchmarks/README.md`](benchmarks/README.md)를 참고하세요. 테스트별 상세
수치와 환경 정보는 Git에서 제외된 `benchmark-results/` 아래에 기록됩니다.
