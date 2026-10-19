# MobileFaceSwap 224 CUDA 서버

기존 `best.pt`·추적·metadata·화이트리스트·blur 계약을 유지하고 합성 renderer만
`--face-swap-backend mobilefaceswap`으로 선택한다. 기본 renderer는 InSwapper,
기본 요청 모드는 blur이며 protobuf 필드는 추가하지 않는다.

## 설치 및 모델 준비

Linux x86_64, Python 3.11 또는 3.12, CUDA 13을 지원하는 NVIDIA 드라이버를 사용한다.
별도 가상환경에서 설치하고 CPU `onnxruntime`과 `onnxruntime-gpu`를 함께 설치하지 않는다.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-mobilefaceswap-cuda.txt
# best.pt의 TensorRT export와 실행을 위한 기존 의존성
.venv/bin/python -m pip install -r requirements-export.txt
```

Torch 2.13 CUDA 13 / cuDNN 9와 ORT GPU 1.30을 조합한다.
[ORT의 CUDA 호환 표](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)를
참고하며, CUDA renderer 로딩 시 `preload_dlls()`로 해당 환경의 CUDA/cuDNN을 준비한다.

모델·프리셋은 GitHub에 추가하지 않고 서버에 별도로 설치한다.

- Git LFS의 `models/best.pt`, `models/face_metadata_extracter.pt` 실제 파일
- 기존 AdaFace checkpoint와 YuNet: [모델 가이드](face-swap-model-download.md)
- `models/mobilefaceswap/MobileFaceSwap_224.pdparams`
- `models/mobilefaceswap/arcface.pdparams`
- `models/mobilefaceswap/landmarks/scrfd_10g_bnkps.onnx`
- `config/face_presets.json`이 참조하는 560개 프리셋 이미지와 일치하는 SHA-256

## RTX 3090 실행

GPU별 TensorRT engine은 대상 서버에서 생성하며 다른 호스트에서 복사하지 않는다.
원본 `best.pt`는 수정하지 않고 engine과 manifest를 별도로 만든다.

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/export_tensorrt.py \
  --checkpoint models/best.pt --output models/best_b1.engine --device 0
./scripts/run_mobilefaceswap_cuda.sh
```

이 launcher는 YOLO TensorRT FP16, metadata CUDA, SCRFD CUDA, MobileFaceSwap CUDA FP16을
선택한다. 합성에는 Conv/BatchNorm fusion과 channels-last를 적용하고 프리셋별 조건부
가중치를 최대 16개 캐시한다. 최초 identity 계산은 Paddle CPU에서 수행한다.
`MOBILEFACESWAP_PRECISION=fp32`로 합성 정밀도만 되돌릴 수 있다. 일반 CLI의 기본값은
FP32이며 `--mobilefaceswap-precision fp16`은 MobileFaceSwap CUDA에서만 허용한다.

기본 gRPC는 `127.0.0.1:50061`, 웹은 `0.0.0.0:8443`이다. `GRPC_PORT`, `WEB_HOST`,
`WEB_PORT`, `TARGET_FPS` 환경변수로 변경한다. 웹 입력 상한은 30 FPS이며 실측 FPS와 다르다.
CUDA launcher의 기본 웹 화면은 [WebRTC H.264 전송](webrtc-serving.md)을 사용한다.
JPEG 비교 화면은 `/jpeg`이며, 이 경로의 `CLIENT_WINDOW=2`는 원격 왕복 대기를 겹친다.
1~5 사이에서 조정할 수 있으며 값이 커지면 대기 중인 프레임도 늘어난다.
`WEBRTC_ENABLED=0`이면 기존 JPEG 화면을 기본으로 사용한다. 기본 blur 5 / Mac 합성 1
설정은 유지한다.
`AI_BACKEND=pytorch`로 engine 없이 비교할 수 있고, 이때도 합성과 SCRFD는 CUDA를 사용한다.

## HTTPS 및 지속 실행

원격 브라우저의 카메라는 HTTPS 또는 localhost가 필요하다. TLS 파일을 지정하지 않으면
launcher는 HTTP를 제공하므로, 카메라 용도로는 다음처럼 인증서를 지정한다.

```bash
mkdir -p .runtime/tls
tailscale cert --cert-file .runtime/tls/server.crt \
  --key-file .runtime/tls/server.key YOUR-HOST.YOUR-TAILNET.ts.net
```

인증서 발급 권한이 있는 Tailscale 호스트에서 실행한다. 인증서와 서버별 설정은
Git 제외 경로인 `.runtime/`에 둔다. `.runtime/server.env` 예시:

```dotenv
GRPC_PORT=50061
WEB_HOST=0.0.0.0
WEB_PORT=8443
TARGET_FPS=30
CLIENT_WINDOW=2
WEBRTC_ENABLED=1
WEBRTC_ENCODER=auto
WEBRTC_MAX_PEERS=2
AI_BACKEND=tensorrt
MOBILEFACESWAP_PRECISION=fp16
OMP_NUM_THREADS=1
TLS_DOMAIN=YOUR-HOST.YOUR-TAILNET.ts.net
SSL_CERTFILE=/home/YOUR-USER/apps/innolive-mobilefaceswap/.runtime/tls/server.crt
SSL_KEYFILE=/home/YOUR-USER/apps/innolive-mobilefaceswap/.runtime/tls/server.key
```

배포 경로는 `~/apps/innolive-mobilefaceswap`을 기준으로 한다. 다른 경로이면 unit의
`WorkingDirectory`, `EnvironmentFile`, `ExecStart`도 맞춘다.

```bash
chmod 600 .runtime/server.env .runtime/tls/server.key
mkdir -p ~/.config/systemd/user
cp deploy/mobilefaceswap*.service deploy/mobilefaceswap-cert.timer ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now mobilefaceswap.service mobilefaceswap-cert.timer
systemctl --user status mobilefaceswap.service
journalctl --user -u mobilefaceswap.service -n 60
```

로그아웃 후에도 실행하려면 해당 사용자에 대한 systemd linger가 활성화돼 있어야 한다.
인증서 timer는 매일 갱신을 확인하고 인증서 내용이 바뀔 때만 서비스를 재시작한다.
서버 업데이트는 서비스 중지 → `git pull --ff-only` → 의존성·모델 확인 → 서비스 시작 순서로
수행한다. 중지는 `systemctl --user stop mobilefaceswap.service`이다.

## 검증

- `.venv/bin/python -m unittest discover -s tests`
- `node --test tests/test_web.mjs`
- `scripts/verify_mobilefaceswap.py --upstream /path/to/official-checkout --source SOURCE.png
  --target TARGET.jpg --torch-provider cuda --torch-precision fp16`
- HTTPS `/healthz`가 `ok`, gRPC가 `SERVING`인지 확인
- 실제 HTTP 이미지·WebSocket 연속 입력에서 `swapped` 및 가림 후 복귀 확인

FP16 검증은 같은 source/target을 원본 Paddle과 비교한다. 0..1 출력에서 최대 오차 0.01,
평균 오차 0.0005를 상한으로 두고, FP32의 최대 오차 상한은 0.0001이다.
합성 모델 단독 시간, 전체 서버 시간, 네트워크 왕복 시간은 구분해서 기록한다.

### RTX 3090 JPEG 경로 확인 결과 (2026-10-19)

Python 3.11.15, Torch 2.13.0+cu130, ORT GPU 1.30.0, Paddle 3.3.1,
TensorRT 11.1.0.106, RTX 3090 24GB에서 확인했다.

| 범위 | 결과 |
| --- | --- |
| YOLO 탐지, 동일 움직임 입력 60프레임, 처음 5개 제외 | PyTorch 중앙값 7.66ms → TensorRT FP16 3.75ms |
| 주요 얼굴 bbox 일치 | 평균 IoU 0.9915, 최소 0.9754 |
| MobileFaceSwap 단독, 동일 source/target 1쌍, warmup 3 + 측정 15회 | CUDA FP32 중앙값 0.987ms / FP16 0.896ms |
| FP16 대 원본 Paddle, 0..1 출력 | 합성 최대 오차 0.001705, 평균 0.000154 / mask 최대 0.002207 |
| HTTPS/WebSocket 157프레임, 첫 프레임 제외 전체 서버 시간 | 중앙값 17.135ms, p95 34.573ms |
| 원격 왕복을 포함한 640×640 정지 입력 40프레임 | 요청 window 1: 3.76 FPS / window 2: 7.19 FPS |

60프레임 비교의 첫 프레임에서 PyTorch가 동일 얼굴을 2개로 중복 검출했고 TensorRT는
1개로 검출했다. 주요 얼굴 누락은 없었으며 완전히 동일한 검출 출력을 보장하는 결과는 아니다.

157프레임은 합성 프리셋 사진의 크기·위치·기울기를 바꾼 4:3/16:9 각 60프레임,
얼굴 2개 배치 12프레임, 가림/복귀 25프레임으로 구성했다. 움직임 입력 120/120에서
`swapped`, 가림 5프레임에서 `blur_fallback`, 가림 제거 후 합성 복귀를 확인했다.
얼굴 2개 배치에서는 탐지기가 프레임당 3개 face 후보를 반환했다.

마지막 행의 원격 FPS는 해당 Tailscale 연결에서 Mac 클라이언트가 측정한 값이다.
서버 처리 시간의 역수를 브라우저 FPS로 해석하지 않는다. 실제 웹캠 촬영 및 장시간·동시
사용자 부하는 별도 검증 대상이다. GPU 서버의 `reports/parity-cuda-*`,
`reports/detector-compare.json`과 클라이언트의 `reports/mobilefaceswap/gpu-websocket/`에
원시 결과를 보관하며 Git에는 포함하지 않는다.
