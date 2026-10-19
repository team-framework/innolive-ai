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

이 launcher는 YOLO TensorRT FP16, metadata CUDA, SCRFD CUDA(선택 시 TensorRT batch), MobileFaceSwap CUDA FP16을
선택한다. 합성에는 Conv/BatchNorm fusion과 channels-last를 적용하고 프리셋별 조건부
가중치를 최대 16개 캐시한다. 최초 identity 계산은 Paddle CPU에서 수행한다.
`MOBILEFACESWAP_PRECISION=fp32`로 합성 정밀도만 되돌릴 수 있다. 일반 CLI의 기본값은
FP32이며 `--mobilefaceswap-precision fp16`은 MobileFaceSwap CUDA에서만 허용한다.

기본 gRPC는 `127.0.0.1:50061`, 웹은 `0.0.0.0:8443`이다. `GRPC_PORT`, `WEB_HOST`,
`WEB_PORT`, `TARGET_FPS` 환경변수로 변경한다. 웹 입력 상한은 30 FPS이며 실측 FPS와 다르다.
CUDA launcher의 기본 웹 화면은 [WebRTC H.264 전송](webrtc-serving.md)을 사용한다.
새 프리셋의 첫 준비 비용을 줄이려면 실행 전
`.venv/bin/python scripts/prepare_mobilefaceswap_cache.py --workers 4`로 캐시를 만든다.
`MOBILEFACESWAP_CACHE_DIR` 기본값은 `.runtime/mobilefaceswap-cache`이며 생성된 가중치는
Git에서 제외한다. NVENC 드라이버 버전 불일치의 서비스 한정 복구는 WebRTC 가이드를 따른다.
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

## 다중 얼굴과 CUDA 배치

CUDA launcher는 `MOBILEFACESWAP_CUDA_GRAPHS=1`로 인물별 생성 모델을 재사용한다.
`0`으로 끄면 일반 Torch 실행으로 돌아간다. 일반 CLI와 MPS/CPU 경로는 기본적으로
CUDA Graph를 사용하지 않는다. 인물별로 입력·출력·가중치·graph를 분리하고 캐시는
최대 16개로 제한한다. 새 graph를 만드는 동안만 YOLO의 device synchronization을
잠시 대기시키며, 이후 replay와 다른 세션의 검출은 기존처럼 겹쳐 실행한다.

랜드마크 검출을 묶으려면 **배포할 GPU에서** SCRFD engine을 생성한다.
TensorRT 10/11과 `onnxconverter-common`이 필요하다. FP16은 명시적으로 선택한다.
서비스 전용 NVIDIA 라이브러리를 사용하는 환경에서는 빌드 시에도 같은
`LD_LIBRARY_PATH`를 설정한다.

```bash
.venv/bin/python scripts/build_scrfd_engine.py \
  --onnx models/mobilefaceswap/landmarks/scrfd_10g_bnkps.onnx \
  --output .runtime/scrfd-b8-fp16.engine --max-batch 8 --precision fp16
export MOBILEFACESWAP_SCRFD_ENGINE="$PWD/.runtime/scrfd-b8-fp16.engine"
./scripts/run_mobilefaceswap_cuda.sh
```

engine과 함께 생성되는 `.engine.json`도 보존한다. 원본 ONNX·engine의 SHA-256,
TensorRT 버전, GPU 이름을 시작 시 확인하며 불일치하면 renderer를 사용할 수 없다.
`MOBILEFACESWAP_SCRFD_ENGINE`을 해제하면 기존 ONNX Runtime으로 돌아간다.
`best.pt`와 원본 MobileFaceSwap 가중치는 바뀌지 않으며 engine은 Git에 넣지 않는다.

한 프레임의 독립적인 얼굴 ROI를 최대 8개씩 검출한다. 원래의 640 canvas,
320 ROI scale, 정규화, threshold, NMS와 bbox 대응 기준을 유지한다. uint8 픽셀만
전송하고 RGB 변환·정규화·padding은 GPU에서 처리한다. 각 얼굴의 좌표와 인물은
분리하고, 검출 실패 시 같은 프레임 원본에서 재시도한 뒤 기존 blur 정책을 적용한다.
생성 모델은 인물마다 가중치가 달라 서로 다른 인물을 하나의 일반 batch로 합치지 않는다.
세션을 모으기 위해 추가로 기다리는 batching 지연은 도입하지 않는다.

합성의 affine warp와 blending은 해당 얼굴 polygon의 bounds에서만 계산한다.
blur는 이미 주변 ROI를 묶어 처리하며, 실패·held 얼굴과 번호판의 보호 처리는 합성
마지막에 적용한다. 이 순서와 whitelist 처리 방식은 유지한다.

서버 내부 처리량과 실제 WebRTC 수신 FPS는 별도로 측정한다.

```bash
# 서버 내부: raw YUV, gRPC window=1, 무제한 입력, 얼굴별 동일 tile 크기(1..8개)
OMP_NUM_THREADS=1 .venv/bin/python scripts/benchmark_face_capacity.py \
  --target 127.0.0.1:50061 --manifest config/face_presets.json \
  --cases 1x1,2x1,4x1,8x1,1x4,2x3,1x8,1x16 --output reports/capacity
# blur만 비교하려면 --mode blur 추가
# Chrome: --sessions 4를 추가하면 같은 URL에 독립 WebRTC 세션 네 개를 동시 측정
# scripts/benchmark_webrtc_browser.py의 --fake-video에는 frame ID가 있는 Y4M 파일 필요
```

이 부하 시험은 합성한 얼굴 수와 검출 수를 함께 기록한다. 얼굴을 놓치거나 blur로
바뀐 프레임의 처리량을 성공적인 얼굴 합성 용량으로 해석하면 안 된다. 16개 얼굴
조건은 120px tile이며 1..8개 조건의 160px tile과 크기가 다르다.

### 다중 얼굴·세션 실측 (2026-10-19)

3090에서 변경 전 `f000ed1`과 최적화 코드 `b80bc54`를 비교했다. raw gRPC 표는
640×480, window 1, 무제한 입력, 각 조건 예열 3초 후 8초 측정한 서버 내부 처리량이다.
서로 다른 합성 인물 사진 4개를 반복 배치하고 ±3도 움직임을 줬다. 1~8개 얼굴은
같은 160px tile을 사용했다. 모델 최초 준비 시간과 네트워크·NVENC 비용은 이 표에서 제외했다.

| 얼굴 / 세션 | 변경 전 세션당 FPS | 변경 후 세션당 FPS |
| --- | ---: | ---: |
| 1 / 1 | 61.1 | 100.4 |
| 2 / 1 | 38.3 | 76.1 |
| 4 / 1 | 21.6 | 52.0 |
| 8 / 1 | 11.8 | 30.0 |
| 1 / 4 | 20.5 | 35.8~35.9 |
| 2 / 4 | 11.3 | 24.8~24.9 |
| 1 / 8 | 10.1 | 17.8 |

추가 시험에서 얼굴 2개 × 3세션은 33.3~33.4 FPS, 얼굴 4개 × 2세션은
33.6~33.8 FPS, 얼굴 8개 × 2세션은 19.6~19.8 FPS였다. 얼굴 1개 × 16세션은
8.6~8.8 FPS까지 하락했다. 1~8개 얼굴의 모든 측정 응답에서 요구한 얼굴 수가
그대로 합성됐으며, 16세션까지 프레임 처리 오류가 없었다. 이는 짧은 부하 시험의
처리 한계이며 브라우저에서 30 FPS를 보장하는 동시 접속 수가 아니다.

16개 얼굴을 120px tile로 줄인 조건은 16.6 FPS였지만 정확히 16개를 합성한 응답은
66.2%였다. 중복 검출(17~19개)과 일부 blur fallback이 있어 안정적인 합성 범위에서
제외한다. 일반 카메라의 가림·측면·조명 변화 및 다른 해상도는 이 실측으로 보장하지 않는다.

같은 입력 24프레임(1/2/4/8개 얼굴, 각 6개 각도)과 동일 인물 가중치를 비교한 결과,
얼굴 polygon 내부 평균 픽셀 차이는 0.080~0.156/255, PSNR은 56.15~59.10dB였다.
polygon 밖 픽셀은 동일했다. CUDA Graph의 생성 RGB/mask 출력은 4개 인물 모두
기존 FP16 모델과 bitwise 일치했다. GPU 픽셀 전처리도 기존 OpenCV blob 입력과
동일한 SCRFD 출력을 냈다. 프리셋의 조건부 가중치를 새로 만들 때는 기존 FP32 ONNX
검출을 사용하므로 target 검출 가속이 source cache의 의미를 바꾸지 않는다.

blur는 기존 CPU ROI 공유 방식 그대로 측정했다. 1개/8개 얼굴의 blur+YUV 출력
중앙값은 2.16/6.23ms였다. 이번 변경은 더 큰 비용을 차지하던 반복 랜드마크 검출과
얼굴 합성·붙여넣기에 집중했으며, blur 강도·경계·whitelist·번호판 보호는 바꾸지 않았다.

같은 Mac Chrome 가상 카메라와 Tailscale 회선, NVENC, 30 FPS 입력, 각 20초 측정:

| 얼굴 | 브라우저 수신 FPS 전→후 | 표시 지연 p50 전→후 | 표시 지연 p95 전→후 |
| --- | ---: | ---: | ---: |
| 4 | 19.54 → 25.52 | 249.9 → 150.8ms | 266.9 → 183.5ms |
| 8 | 11.27 → 21.85 | 349.9 → 266.8ms | 367.7 → 300.2ms |

네 조건 모두 브라우저 freeze와 RTP 수신 손실은 0이었다. 100ms 초과 표시 간격은
4개 얼굴 18→7회, 8개 얼굴 80→17회로 줄었지만 완전히 사라지지는 않았다.
표시 지연은 로컬 미리보기 표시부터 반환 영상 표시까지이며 카메라 센서부터의 지연은 아니다.
8개 얼굴에서 서버만 약 30 FPS여도 전송·인코딩·디코딩을 포함하면 약 22 FPS인 차이가 남는다.


다중 Chrome 세션은 동일 Mac에서 각각 독립 카메라 track으로 시험했다. 얼굴 4개씩
2세션의 30초 측정은 세션당 21.6~21.8 FPS, 표시 지연 중앙값 234~267ms였고,
각 세션에서 freeze 3회가 기록됐다. 서버 내부 약 34 FPS보다 실제 수신 FPS가 낮아
이 조건에서 부드러운 30 FPS를 보장할 수 없다. 운영 접속 상한과 얼굴 수에 따른
처리 예산을 별개로 관리해야 하며, 많은 얼굴·세션의 성능은 회선에서 재검증해야 한다.

얼굴 1개씩 4세션의 30초 재검증은 세션당 23.4~24.0 FPS, 표시 지연 중앙값
169~184ms, freeze 0회였다. 세션별 RTP 손실 0~12개와 100ms 초과 표시 간격
10~17회가 남았다. 최종 운영 설정은 최대 4 WebRTC 세션을 허용하며, 여러 얼굴이
있는 세션까지 모두 30 FPS로 유지된다는 의미는 아니다. launcher의 공통 기본값은
2이며 `WEBRTC_MAX_PEERS=4`로 해당 배포에서만 늘린다.
