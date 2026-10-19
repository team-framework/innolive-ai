# WebRTC 얼굴 합성 서빙

브라우저 영상은 H.264 RTP로 보내고 합성된 영상을 별도 RTP 트랙으로 받는다.
매 프레임 JPEG 업로드·응답 대기 대신 연속 영상으로 전달한다. 기존 gRPC 서버·모델·
세션·화이트리스트·실패 시 blur 처리는 유지한다.

```text
카메라 → 브라우저 H.264 → WebRTC/DTLS-SRTP → YUV420P
      → localhost ProcessVideo → 기존 탐지·추적·합성 → YUV420P
      → H.264 → WebRTC/DTLS-SRTP → 브라우저 영상
```

gRPC의 기존 `pix_fmt=yuv420p`, `width`, `height` 필드를 사용한다. 필드 추가는 없으며,
이 opt-in 경로의 응답도 packed YUV420P다. JPEG 요청은 계속 JPEG 응답을 받는다.
`grpc_client.VideoFrame`의 마지막 선택 인자와 `VideoResult.processed_data`로 raw 경로를
사용할 수 있다. 프레임 순서·크기·오류 응답은 JPEG 경로와 동일하게 검사한다.

## 실행

```bash
.venv/bin/python -m pip install -r requirements-webrtc.txt
.venv/bin/python server.py --host 0.0.0.0 --port 8443 \
  --grpc-target 127.0.0.1:50061 --anonymization-mode face_swap \
  --target-fps 30 --webrtc --webrtc-encoder auto \
  --ssl-certfile .runtime/tls/server.crt --ssl-keyfile .runtime/tls/server.key
```

`scripts/run_mobilefaceswap_cuda.sh`는 WebRTC를 기본 활성화한다. `WEBRTC_ENABLED=0`이면
기존 JPEG 웹 화면을 기본으로 제공한다. 일반 `server.py`는 `--webrtc`가 없으면 기존
동작을 유지한다. WebRTC 활성화 상태에서도 `/jpeg`, `/ws`, 이미지·세션·화이트리스트
API를 그대로 사용할 수 있다. `/`의 새 화면에서 카메라 또는 동영상을 선택한다.
화이트리스트는 `/jpeg`에서 같은 세션에 등록하며 등록된 얼굴은 합성에서 제외된다.

## 지연과 인코더 설정

- 수신 프레임은 최대 1개만 대기하고 새 프레임으로 교체
- 내부 gRPC는 window 1, 프레임 순서와 추적 상태 유지
- 영상은 긴 변 640px 이하, 최대 30 FPS, 브라우저 송신 상한 1.5Mbps
- H.264는 B-frame·lookahead 없이 인코딩, RTCP REMB에 따른 비트레이트 조절 유지
- 최초 모델·프리셋 준비는 기존 lazy cache를 사용하므로 첫 결과는 정상 상태보다 느릴 수 있음
- 브라우저는 결과가 2초간 멈추거나 연결이 실패하면 결과를 숨기고 연결 종료
- 브라우저 수신 jitter buffer 목표 0ms 요청, 실제 지연은 브라우저가 회선 상태에 맞춰 결정
- 서버는 유휴 연결 20초 후 해제, 기본 동시 연결 상한 2개
- RTP payload는 1150바이트로 제한해 IPv6·SRTP 헤더를 포함해 Tailscale MTU 1280에 수용
- H.264 marker가 있고 패킷이 연속이면 다음 프레임 도착을 기다리지 않고 디코딩
- 패킷 누락 시 기존 재정렬·NACK 처리를 유지하며 브라우저와 서버 양쪽 RTX 협상
- NVENC의 비트레이트 변경은 기존 세션에서 처리해 CUDA encoder 재생성 중단 방지
- 카메라의 불규칙한 타임스탬프는 90kHz 정밀도로 유지해 인접 프레임 병합 방지

`WEBRTC_ENCODER=auto`는 실제 NVENC 첫 프레임 인코딩을 검사한 뒤 선택한다. GPU 모델명이나
FFmpeg 인코더 목록만으로 활성화하지 않는다. NVENC가 초기화되지 않으면 2개 스레드의
`libx264`, `veryfast`, `zerolatency`를 사용한다. `nvenc`는 사용 불가 시 시작 실패,
`x264`는 CPU 인코더를 명시한다. 현재 인코더는 `/api/webrtc/config` 및 `/healthz`에서
확인할 수 있다. `WEBRTC_MAX_PEERS`로 연결 상한을 바꾸되 다중 사용자 용량은 별도 측정한다.

aiortc 1.15.0의 H.264 encoder factory를 사용하므로 버전을 고정한다. PyAV의 codec 설정만
교체하고 aiortc의 패킷 분할, keyframe 요청, RTCP 혼잡 피드백은 유지한다.
브라우저는 Constrained Baseline(`42e0`) 프로파일을 우선한다. 시험 Mac의 Chrome
VideoToolbox에서 Baseline(`4200`) 송신이 첫 프레임 뒤 멈추는 현상이 있었고, Constrained
Baseline으로 연속 송신·중지·재시작을 검증했다.
최종 Chrome 통계에서 송신 인코더는 OpenH264, 수신 디코더는 VideoToolbox로 확인했다.
설정 근거: [aiortc API](https://aiortc.readthedocs.io/en/latest/api.html),
[NVIDIA 저지연 인코딩 지침](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/ffmpeg-with-nvidia-gpu/index.html#low-latency-transcoding).

## 네트워크

HTTPS 포트는 페이지·SDP 연결 설정용이고 실제 영상은 ICE가 선택한 UDP 경로로 전달한다.
현재 배포는 같은 Tailscale 네트워크의 주소에 직접 연결하며 별도 STUN/TURN을 요구하지
않는다. 웹 포트 하나만 인터넷에 포워딩하는 구성과는 다르다.

다른 네트워크에서 사용할 때는 `WEBRTC_ICE_SERVERS_JSON` 환경변수에 브라우저와 서버가
함께 사용할 STUN/TURN 목록을 지정한다. 인증 정보는 `.runtime/server.env` 같은 Git 제외
경로에 둔다. 예시 형식은 `[{"urls":"turn:HOST:3478","username":"USER","credential":"TOKEN"}]`이다.
이 데모의 기존 세션 API와 동일하게 신뢰된 네트워크를 전제로 하며, 인터넷 공개용 인증이나
TURN 계정 관리는 추가하지 않는다.

## 재현 및 검증

```bash
.venv/bin/python -m unittest discover -s tests
node --test tests/test_web.mjs tests/test_webrtc.mjs
.venv/bin/python scripts/benchmark_webrtc.py \
  --url https://YOUR-HOST:8443 --source /path/to/synthetic-portrait.png \
  --seconds 20 --warmup 5 --output reports/webrtc/one-face
# 동일 조건에서 두 얼굴 입력
# ... --faces 2 --output reports/webrtc/two-faces
```

비교 입력은 640×480, 30 FPS의 위치·기울기 변화가 있는 합성 인물이다. JPEG는 window 2,
WebRTC는 H.264를 사용한다. 양쪽 모두 한 클라이언트 시계에서 프레임 생성부터 반환 영상
디코딩까지의 지연을 계산한다. H.264를 통과하는 화면 상단 frame ID로 프레임을 대응한다.
서버 처리 시간은 별도 기록하며 `1 / 서버 처리 시간`을 수신 FPS로 보고하지 않는다.

`payload_bytes`는 WebSocket application payload, `transport_bytes`는 WebRTC transport
통계이므로 서로 동일한 계층의 패킷 캡처는 아니다. 고정된 실제 회선 대역폭이나 장시간
품질을 보장하는 결과로 해석하지 않는다. 원시 결과와 이미지는 Git에서 제외한다.

### RTX 3090 / Mac 클라이언트 비교 (2026-10-19)

aiortc 클라이언트, 위 스크립트의 움직임 입력, 각 경로 warmup 5초 후 1명 20초 / 2명 15초
측정 결과다. 두 얼굴 입력은 빈 배경 면적과 압축률도 달라지므로 두 행의 FPS 차이를
얼굴 수만의 효과로 해석하지 않는다.
서버 시간과 합성 상태 집계는 최초 5개 응답을 제외한 전체 응답 기준이다.

| 입력 | JPEG 수신 FPS | WebRTC 수신 FPS | JPEG 왕복 중앙값 | WebRTC 왕복 중앙값 | WebRTC 서버 중앙값 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1명 | 13.04 | 26.33 | 134.6ms | 149.0ms | 17.34ms |
| 2명 | 14.45 | 26.24 | 121.1ms | 170.7ms | 27.38ms |

이 비교에서 WebRTC의 수신 FPS는 증가했지만 왕복 지연은 감소하지 않았다. 기존 정지
640×640 JPEG 시험의 7.19 FPS와 입력 조건이 다르므로 그 값과 개선 배수를 계산하지 않는다.
같은 Mac의 실제 Chrome 페이지에서도 합성 인물 Y4M 가상 카메라로 약 26 FPS 재생,
중지·재시작·연결 해제·390px 화면 폭을 검사했다. 실제 물리 카메라·장시간·동시 사용자
성능은 미측정이다.

Chrome 페이지끼리 동일한 640×480, 30 FPS Y4M 입력을 다시 비교한 결과는 아래와 같다.
각 경로 22초 중 최초 5초를 제외했으며 화면의 frame ID를 대조했다. 이 지연은 같은 페이지의
원본 미리보기 표시부터 결과 표시까지이며, 위 aiortc의 프레임 생성→디코딩 구간과 다르다.

| 경로 | 화면 수신 FPS | 표시 지연 중앙값 | 표시 지연 p95 |
| --- | ---: | ---: | ---: |
| JPEG / WebSocket | 11.35 | 182.7ms | 313.7ms |
| H.264 / WebRTC | 24.83 | 250.1ms | 267.3ms |

이 브라우저 비교에서도 프레임 수는 약 2.2배 증가했지만 지연 중앙값은 증가했다.
WebRTC의 연결 RTT나 서버 처리 시간을 화면 반영 지연으로 해석하면 안 된다.

RTX 3090의 CUDA 합성은 계속 FP16으로 실행한다. 해당 서버는 로드된 NVIDIA kernel
595.91.07과 userspace 595.99.02가 달랐고, NVENC 실기동 검사가 실패했다. 시스템 드라이버나
다른 서비스를 변경하지 않았으며 위 결과의 반환 영상은 `libx264`로 인코딩했다.
아래 후속 개선에서는 서비스 전용 라이브러리로 NVENC를 복구했다.

## NVENC 복구와 프리셋 전환 지연 개선

위 초기 결과 이후, 실행 중인 커널 `595.91.07`과 같은 버전의 NVIDIA 공식 사용자
라이브러리를 `.runtime/nvidia/595.91.07`에 준비해 실제 NVENC 인코딩을 검증했다.
`libnvidia-encode`뿐 아니라 내부에서 함께 로드하는 `libnvcuvid`와 `libcuda`도 일치해야 한다.
시스템 패키지·커널 모듈·다른 프로세스는 변경하지 않는다. 시스템 전체의 버전 불일치를
해소하려면 별도 점검 후 드라이버 업데이트를 완료해야 하며, 이번 복구 범위는 이 서비스다.

```bash
# Linux x86_64, dpkg-deb 필요. 공식 Packages 인덱스의 SHA-256·크기 확인 후 추출
.venv/bin/python scripts/prepare_nvidia_userspace.py
# 기존 디렉터리는 덮어쓰지 않음. 실행기는 현재 커널과 같은 버전 디렉터리만 선택
./scripts/run_mobilefaceswap_cuda.sh
```

실행기는 현재 `/sys/module/nvidia/version`과 일치하는 경로만 `LD_LIBRARY_PATH` 앞에
추가한다. 추후 새 모듈로 재부팅하면 예전 라이브러리를 계속 강제하지 않는다.
`/api/webrtc/config`의 `encoder=h264_nvenc` 및 실제 영상 송수신으로 사용 여부를 확인한다.
근거: [NVIDIA 공식 패키지](https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/),
[FFmpeg NVENC 비트레이트 재설정](https://github.com/FFmpeg/FFmpeg/blob/master/libavcodec/nvenc.c).

약 780ms의 서버 정지는 새로운 프리셋의 Paddle source conditioning에서 관찰됐다.
다음 명령은 카탈로그의 모든 프리셋을 미리 계산해 CPU FP32 fused weights로 저장한다.

```bash
OMP_NUM_THREADS=1 .venv/bin/python scripts/prepare_mobilefaceswap_cache.py --workers 4
```

CUDA 실행기의 기본 캐시 경로는 `.runtime/mobilefaceswap-cache`이며
`MOBILEFACESWAP_CACHE_DIR`로 변경한다. 모델 두 개·SCRFD 파일·소스 이미지의 SHA-256과
캐시 형식 버전으로 유효성을 확인한다. `torch.load(weights_only=True)`·strict tensor 검사·
유한값 검사 후 읽고, 손상되거나 없으면 기존 경로에서 다시 준비한다. CPU/MPS의 일반
실행에는 환경변수를 지정한 경우만 적용한다. GPU 메모리 캐시는 기존처럼 16개로 제한한다.
전처리·포팅·fusion 구조를 바꾸면 `CACHE_FORMAT`도 갱신해야 한다.

배포 서버에서 560개 프리셋을 준비·재검증했다. 6개 프리셋의 최초 디스크 읽기와 CUDA 전송은
8.9~10.5ms였다. CPU SCRFD로 준비한 캐시와 기존 CUDA SCRFD 경로를 같은 입력으로 비교한
FP16 합성 최대 차이는 0..1 범위에서 0.0004883, mask 최대 차이는 0.0001107이었다.
이는 픽셀 단위 완전 일치 주장이 아니며, 생성된 가중치·캐시·드라이버 파일은 Git에서 제외한다.

메타데이터 모델도 CUDA에서 새 batch 크기를 처음 처리할 때 약 900ms, 이후 약 3ms가
측정됐다. `FACE_METADATA_WARMUP_MAX_BATCH=8`을 CUDA 실행기의 기본값으로 두어 모델을
처음 준비할 때 1~8명 입력을 예열한다. 일반 실행 기본값은 0이며 기존 동작을 유지한다.
최초 준비 시간은 늘어나지만 재생 중 얼굴 수가 바뀔 때 발생하는 첫 실행 비용을 줄인다.
8명을 넘는 입력이나 장시간·다중 사용자 성능은 별도 검증 대상이다.

실제 Chrome 비교는 Playwright와 설치된 Chrome, 상단 256×16px에 16-bit frame ID가 있는
640×480·30 FPS Y4M 가상 카메라를 사용한다. `scripts/benchmark_webrtc.py`의
`MovingPortrait`와 동일한 입력이다. 물리 카메라는 열지 않는다.

```bash
python scripts/benchmark_webrtc_browser.py \
  --url before=https://BASELINE-HOST:8443 --url after=https://TEST-HOST:8443 \
  --fake-video /path/to/marked-camera.y4m --browser /path/to/chrome \
  --seconds 30 --warmup 5 --output reports/webrtc/browser-comparison
```

첫 합성 결과를 받은 뒤 5초 예열하고 양쪽 모두 같은 30초를 측정한다. 프레임 간격의
p95, 100ms 초과 간격, 표시 지연과 실제 `RTCInboundRtpStreamStats.freezeCount` 변화량을
함께 기록한다. 연결 RTT를 화면 지연으로 대체하지 않으며 측정 끝에 합성이 멈췄으면 실패한다.

### NVENC 및 전송 개선 전후 (2026-10-19)

같은 Mac의 Chrome 155, 같은 Tailscale 경로, 위 가상 카메라로 각 30초를 측정했다.
기존 서버는 `5f5e363`의 x264, 개선 서버는 NVENC·RTP 수정·프리셋 캐시·메타데이터 예열을
모두 적용했다. 여러 변경을 함께 비교한 결과이므로 NVENC만의 개선 효과는 아니다.

| 측정 | 기존 | 개선 |
| --- | ---: | ---: |
| 화면 수신 FPS | 25.76 | 27.06 |
| 표시 지연 중앙값 | 232.7ms | 199.9ms |
| 표시 지연 p95 | 250.1ms | 217.0ms |
| 화면 프레임 간격 p95 | 83.4ms | 66.7ms |
| 100ms 초과 화면 간격 | 20회 | 6회 |
| 브라우저 freeze 증가 | 0회 | 0회 |
| 서버 처리 p95 | 28.34ms | 27.18ms |

지연 중앙값은 약 14%, 100ms 초과 간격은 70% 줄었다. NVENC는 실제 H.264 세션과
`nvidia-smi encodersessions`에서 약 0.34~0.38ms 하드웨어 인코딩 지연을 확인했다.
이 값은 화면 표시 지연이 아니다. 50ms 재생 버퍼 상한을 요청하는 실험은 지연이 줄어도
끊김이 증가해 적용하지 않았다. 실제 카메라·다른 회선·동시 사용자에서 같은 결과를
보장하지 않으며, 이 측정에서도 모든 100ms 간격이 제거된 것은 아니다.

같은 개선 설정의 별도 120초 연속 시험은 26.63 FPS, 표시 지연 중앙값 199.0ms·p95
216.8ms, 브라우저 freeze 0회, 서버 처리 최대 40.52ms였다. 100ms 초과 화면 간격은
31회, RTP 수신 손실은 4개로 기록됐으며 종료 시점까지 합성 출력을 유지했다.
