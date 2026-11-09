# MobileFSGAN GPU·CPU 자원 및 추가 최적화 점검

2026-11-08, `gorani@gpu` RTX 3090에서 PR #101의 mixed FP16·batch·CUDA Graph·GPU
YuNet 설정을 확인했다. 추가 개선 여지는 **CPU 전후처리와 단계 사이 대기**에 있으며,
동시 스트림에서는 **300W 전력 제한**도 관측됐다. VRAM 부족·열 제한·NVENC 포화는
관측되지 않았다. 전력 한도·클록·기존 서비스는 변경하지 않았다.

## 측정 조건

- 기존 4종 프리셋 grid, FHD raw YUV420p gRPC loopback, window 1, 속도 제한 없음
- 각 조건 예열 5초 + 측정 20초, NVML/psutil 약 0.24초 간격의 실제 샘플 82개
- NVML 호출을 별도 스레드로 실행해 클라이언트 event loop 정지 방지
- GPU 사용률은 GPU 전체에서 커널이 동작한 시간 비율이며 SM occupancy/연산 효율이 아님
- CPU 100%는 논리 CPU 1개 사용량. 서버에는 논리 CPU 24개가 노출됨
- 측정 중 모든 응답에서 요청한 수의 얼굴 검출·합성, fallback 0
- 초기 1얼굴/4얼굴 단일 세션에서 시스템 CPU 사용률 순간 상승 관측(p95 87.9%/95.3%).
  원인은 분리하지 않았으며 이전 10초 성능표와 FPS를 직접 비교하지 않음

## 실제 부하의 자원 사용량

사용률·전력·온도·CPU는 측정 구간 평균이다. CPU는 AI 서버 프로세스 기준이다.

| FHD 얼굴 × 세션 | 세션별 FPS | GPU busy | DRAM busy | 전력 | 온도 | CPU | 전력 제한 관측 샘플 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 × 1 | 44.1 | 28.0% | 9.6% | 163W | 53.9°C | 98.6% | 0% |
| 4 × 1 | 30.0 | 32.6% | 19.7% | 198W | 56.8°C | 110.8% | 0% |
| 8 × 1 | 21.1 | 36.7% | 26.1% | 222W | 60.3°C | 108.6% | 15.9% |
| 1 × 4 | 22.7–22.8 | 51.6% | 20.0% | 287W | 68.3°C | 226.3% | 100% |
| 4 × 2 | 23.2 | 48.0% | 31.2% | 289W | 69.8°C | 205.6% | 100% |
| 4 × 4 | 11.6 | 47.9% | 31.2% | 293W | 71.4°C | 206.1% | 100% |

- 새 AI 서버 VRAM: **1,994MiB**, 측정한 모든 부하에서 동일
- 기존 GPU 프로세스 VRAM: **4,140 + 276MiB**, 이 서버와 구분
- GPU 전체 사용량: 약 **6.7–6.9GiB / 24GiB**. 나머지는 시스템·그래픽 할당도 포함
- AI 서버 RSS 최대 약 **3.01GiB**, 시스템 available RAM 약 **39GiB 이상**
- GPU 최고 온도 **72°C**, thermal slowdown 플래그 전 구간 0
- 설정된 power limit **300W**, 장치 default **350W**, max **365W**. 한도는 그대로 유지
- 전력·클록 이벤트의 관측 창이 다르므로 평균 전력이 300W 미만이어도 순간 power cap 발생 가능
- PCIe 최대 **Gen4 ×16**. 측정한 평균 전송량은 방향별 대략 **0.1–0.5GiB/s**

4얼굴 2세션의 합산 **46.3 FPS**에서 4세션의 **46.4 FPS**로 처리량이 거의 늘지 않았다.
첫 세션 RTT p50은 **42.1 → 84.1ms**로 늘었다. 세션 수 증가만으로 처리 용량을
늘릴 수 있는 상태는 아니다.

## 단계별 병목

같은 모델을 별도 직접 파이프라인에서 실행하고 NVTX 구간을 측정했다.
FHD 4얼굴, 30프레임 예열 후 100프레임 평균이며 gRPC·브라우저는 제외했다.
상위 구간은 하위 구간을 포함하므로 표 전체를 더하면 안 된다.

| 구간 | 평균 시간/프레임 |
|---|---:|
| 탐지·추적·객체 구성 전체 | 8.84ms |
| 그중 YOLO predict | 4.96ms |
| 그중 객체 구성 | 0.84ms |
| 얼굴 처리 전체 | 18.01ms |
| 그중 G 추론 + H2D/D2H | 8.25ms |
| 그중 CPU paste-back | 4.16ms |
| 그중 YuNet | 1.45ms |
| 그중 affine crop | 1.03ms |
| 그중 tensor 전처리 | 0.65ms |
| 그중 YUV 변환 | 0.61ms |
| 그중 alignment 계산 | 0.35ms |
| metadata 재평가의 프레임당 평균 비용 | 0.23ms |
| 직접 파이프라인 전체 | 27.12ms |

PyTorch/CUPTI로 별도 30프레임 CUDA timeline도 수집했다. GPU kernel 합계는
**292.86ms**, GPU 메모리 복사는 **15.00ms**였다. 겹치는 이벤트를 합친 GPU 활동은
**307.61ms / 837.77ms = 36.7%**, 프레임당 약 **10.25ms**였다.
복사 자체는 프레임당 약 **0.50ms**로, 현재 가장 큰 단독 비용은 아니다.
이 trace는 픽셀 동일 CPU 후보에서 수집했으며 profiler 오버헤드·직접 실행 조건이
포함된다. 실서버 FPS와 직접 환산하지 않는다. SM occupancy·실제 DRAM bandwidth
카운터는 측정하지 않았다. 설치된 Nsight Systems는 `.qdstrm` 수집만 가능하고 host
importer가 없어 kernel 집계에는 PyTorch/CUPTI trace를 사용했다.

## 확인한 개선 후보

### 요청 대기열 확대: 낮은 지연에는 불리

| FHD 얼굴 | window | FPS | RTT p50 / p95 |
|---|---:|---:|---:|
| 4 | 1 | 30.00 | 31.77 / 43.00ms |
| 4 | 2 | 32.65 | 59.92 / 65.87ms |
| 8 | 1 | 21.05 | 46.44 / 53.91ms |
| 8 | 2 | 21.90 | 89.37 / 102.02ms |

서버가 같은 스트림의 프레임을 순서대로 처리하므로 window 확대만으로 다음 프레임
추론과 현재 프레임 합성이 중첩되지 않는다. 단일 스트림 RTT 우선이면 window 1이 적합하다.
서버나 기존 클라이언트 기본값을 이 실험만으로 변경하지 않았다.

### CPU 임시 배열 제거: 미세한 이득, 런타임 반영하지 않음

전처리·paste-back의 NumPy 임시 배열을 줄인 후보를 같은 입력에서 번갈아 100회씩
측정했다. 480p/720p/FHD × 1/4/8얼굴의 9조건에서 RGB 및 전처리 tensor가 완전히
일치했다. 하지만 FHD 4얼굴의 해당 CPU 구간은 **5.52 → 5.44ms**, 8얼굴은
**10.93 → 10.69ms**에 그쳤다. 전체 FPS 개선을 입증할 수준의 큰 차이가 아니므로
검증 서버와 PR의 합성 런타임은 기존 구현을 유지했다. 후보와 결과는 reports에 보관했다.

### 다음 작업의 우선순위

1. **CPU paste-back·crop의 GPU 이동 또는 CPU/GPU 단계 중첩.** 현재 CPU paste-back만
   약 4.2ms이다. GPU 이동 시 왕복 복사·mask 계약·픽셀 차이까지 포함해 검증해야 하며,
   이 시간이 전부 절감된다고 보장하지 않는다.
2. **다음 프레임 탐지와 현재 프레임 합성을 중첩하는 bounded pipeline.** 단일 요청의
   대기열 길이만 늘리는 것과 구분해야 한다. tracker 순서, whitelist 승인 시점,
   세션 identity, 실패 시 블러 및 최신 프레임 우선 정책을 유지하는 설계가 필요하다.
3. **전용 GPU 운영 시 전력 한도 영향 A/B 측정.** 현재 동시 부하에서 cap이 걸리지만,
   350W로 올릴 때의 이득은 측정하지 않았다. 공유 GPU의 설정 변경 없이 이번 점검 종료.

face composition executor는 1개이며 renderer도 lock으로 직렬화한다. worker 수만
늘리면 같은 lock 대기 또는 공유 context 충돌을 만들 수 있다. 병렬화하려면 context·버퍼
소유권과 메모리 상한까지 함께 바꿔야 한다. VRAM 여유만으로 병렬 처리 이득을 보장하지 않는다.

## WebRTC 및 NVENC

GPU 서버 headless Chrome, 640×480 4얼굴 가상 카메라, 2세션, 5초 예열 후 40초.
별도 monitor가 가운데 25초를 관측했다.

- RTP 수신 디코딩 FPS: **29.08 / 28.88**, 송신 인코딩 FPS: **29.30 / 29.25**
- 프리뷰→출력 관측 지연: p50 **66.7ms**, p95 **66.9 / 67.0ms**
- 두 세션 모두 decoded frame drop·freeze·packet loss 0
- GPU 평균 busy **58.3%**, DRAM busy **38.1%**, 전력 **288W**, power cap 샘플 100%
- NVENC 평균 **1.9%**, 최대 **3%**; NVDEC **0%**(이 경로의 수신 디코딩은 소프트웨어)
- AI 서버 CPU **171.8%**, 웹 서버 CPU **16.7%**, GPU 전체 VRAM **7.28GiB**

첫 탭의 requestAnimationFrame 관측은 source marker 일부를 놓쳐 marker 기준 FPS가
19.68로 낮게 나왔지만 RTP는 29.08 FPS를 디코딩했다. 따라서 benchmark에
`rtp_decoded_fps`, `source_encoded_fps`를 추가했고 기존 `fps`는 marker 관측 값으로
보존했다. 여러 탭의 성능 평가는 RTP 수신률과 실제 presentation 관측을 구분한다.
지연 값은 관측된 동일 marker의 차이이며 센서 지연·외부망·실제 모니터 표시 지연이 아니다.

## 재현과 결과 위치

```bash
# GPU 서버의 독립 CUDA 환경. PIDs는 실행 시 확인.
.venv/bin/python -m pip install psutil nvidia-ml-py
.venv/bin/python scripts/profile_face_resources.py \
  --server-pid <AI_PID> --cases 1x1,4x1,8x1,1x4,4x2,4x4 \
  --seconds 20 --warmup 5 --output reports/resources-fhd

# 별도로 동작 중인 브라우저 부하 관측
.venv/bin/python scripts/profile_face_resources.py \
  --observe --server-pid <AI_PID> --extra-pid <WEB_PID> \
  --seconds 25 --warmup 15 --output reports/resources-browser-monitor
```

서버 `/home/gorani/apps/innolive-mobilefsgan-100/reports`와 로컬
`reports/mobilefsgan-100`에 raw sample·단계별 JSON·CUDA trace·브라우저 stats를 보관했다.
고정 입력의 단기 확인이며 실카메라·외부 회선·장시간 안정성·Mac 성능은 검증하지 않았다.

- [NVIDIA의 GPU/메모리 utilization 정의](https://nvidia.custhelp.com/app/answers/detail/a_id/3751/kw/driver)
- [TensorRT 동시 실행과 CUDA Graph의 context·버퍼 제약](https://docs.nvidia.com/deeplearning/tensorrt/11.4.1/performance/optimization.html)
