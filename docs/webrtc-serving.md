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
다른 서비스를 변경하지 않았으며 위 결과의 반환 영상은 `libx264`로 인코딩했다. 향후
드라이버가 일치하면 `auto`의 실제 인코딩 검사 통과 여부로 NVENC를 선택한다.
