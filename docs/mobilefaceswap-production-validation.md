# MobileFaceSwap 운영 경로 검증 (2026-10-19)

`feat/mobilefaceswap-runtime/#66`의 메인 gRPC 경로와 RTX 3090에서 검사했다.
비교 기준은 기존 배포 `430b6a4`에 최신 main `7bf78f9`의 blur 수정을 병합한 `20aa425`이다.
새 이미지 변경에 따른 영향과 코드 최적화 효과를 구분하기 위해 성능 비교는 같은 기존
프리셋·입력으로 실행하고, 새 프리셋은 별도로 전수 캐시 생성과 합성 검사를 수행했다.

## 보존한 계약과 변경한 처리

- `protos/ai_processor.proto`는 `7bf78f9`와 byte-identical이며 SHA-256은 `a55690094906062a294cc1bb1ecfde971f434b8d18f9115eed22f44b1c14207e`이다.
- 기본 backend는 InSwapper, 기본 요청 모드는 blur이다. MobileFaceSwap는 AI 서버 CLI로 선택하며 Go 요청에 새 필드를 추가하지 않는다.
- metadata의 30프레임 갱신, stream별 identity, whitelist, held face와 번호판 blur, 실패 시 원본 미출력 계약을 유지한다.
- MobileFaceSwap는 얼굴별 전체 프레임 복사를 제거하고 검증을 마친 ROI만 caller 소유 출력에 반영한다. 입력 crop은 항상 원본 프레임에서 읽는다.
- 서로 다른 인물은 조건부 가중치가 달라 합성 forward를 각각 실행한다. CUDA 입출력 전송은 최대 8개씩 묶고 동기화를 줄인다. 같은 CUDA Graph의 출력은 재실행 전에 별도 buffer에 복사한다. 이를 서로 다른 가중치의 단일 forward batch로 해석하면 안 된다.
- CUDA launcher는 준비된 가중치를 요구하고 40개 기본 표정 source를 시작 시 예열한다. GPU source cache는 64개이며 1..128 범위로 설정한다. 캐시 준비를 생략한 배포는 시작을 거부하고, 요청 중 cache 오류는 해당 얼굴의 blur로 처리한다.

## 동일 입력의 서버 내부 성능

RTX 3090, YOLO TensorRT FP16, SCRFD TensorRT 최대 batch 8, MobileFaceSwap FP16/CUDA Graph.
640×480 합성 얼굴 grid, ±3도 회전, 1~8명 모두 160px tile, raw YUV gRPC window 1.
각 조건 예열 4초 후 12초 측정했으며 네트워크·브라우저·NVENC 시간은 포함하지 않는다.
다른 GPU 서비스는 실행 상태였으나 별도 부하 시험을 동시에 실행하지 않았다.

| 얼굴 × 세션 | 이전 세션당 FPS | 변경 후 세션당 FPS |
| --- | ---: | ---: |
| 1 × 1 | 105.50 | 105.58 |
| 2 × 1 | 79.42 | 81.33 |
| 4 × 1 | 53.92 | 55.33 |
| 8 × 1 | 30.92 | 32.42 |
| 1 × 4 | 36.25 | 35.67 |
| 4 × 2 | 34.67~34.75 | 36.83 |

모든 측정 응답에서 요청한 수의 얼굴을 검출하고 모두 합성했다. 4명 × 2세션의
합성·출력 구간 중앙값은 18.06ms에서 14.74ms로 감소했다. 1명 × 4세션은 1.6% 감소해
개선으로 해석하지 않는다. 이 조건의 주된 제한은 한 번에 한 프레임을 처리하는 탐지 lane이다.
8명 × 1세션의 합성·출력 구간 중앙값은 20.67ms에서 19.47ms였다.

합성·paste-back만 따로 비교한 480p 측정은 같은 crop·polygon·가중치를 사용하고
serial/grouped 순서를 번갈아 실행했다. 8명은 11.59ms에서 10.85ms로 감소했다.
검출·전송을 제외한 수치이며 위 전체 서버 처리량과 구분한다.

## 품질과 클라이언트 검사

- 기존 프리셋: 1/2/4/8명 × 6각도 × 480p/FHD, 총 48개 입력에서 serial/grouped 최종 RGB가 bitwise 일치했다. polygon 밖 픽셀도 원본과 동일했다. FHD는 parity 검증이며 브라우저 FHD 처리량 검증이 아니다.
- 새 프리셋: 560개 source 전부 원본 ONNX landmark·Paddle conditioning·저장 후 재로드를 검증했다. 캐시 준비는 4 CPU worker에서 126.9초였다. 24개 다중 얼굴 입력에서도 serial/grouped 출력과 얼굴 밖 픽셀이 일치했다.
- 저장된 실제 합성 이미지에서 얼굴 위치, 눈·입과 피부 경계, 배경 유지 상태를 검토했다. 합성 모델 자체의 부드러워지는 피부 질감은 남으며, 측면·심한 가림·실제 카메라의 장시간 품질을 보장하는 검사는 아니다.
- Go gRPC 클라이언트를 현재와 동일한 proto로 빌드해 3090에 연결했다. blur/face_swap/face_metadata 각각 JPEG 35프레임, frame ID·timestamp echo, 합성 35/35, metadata-only 빈 data, stream 중 모드 변경의 INVALID_ARGUMENT를 확인했다. 새 프리셋 적용 환경에서도 같은 검사를 통과했다. 실제 Go 백엔드 전체 서비스와의 배포 E2E는 포함하지 않는다.
- Mac 전체 Python 테스트 337개(가속기 전용 5개 제외), Linux/3090 337개(NVENC opt-in 1개 제외)를 통과했다. NVENC를 켠 별도 WebRTC 테스트 14개도 모두 통과했다. JavaScript 2개, 협업 정책 29개, 변경 범위 Ruff·format·shell 문법·diff 검사를 통과했다.

## 운영 WebRTC 60초 결과와 남은 제한

실제 운영 HTTPS 8443, NVENC, Mac Chrome 가상 카메라, 640×480 4얼굴·30 FPS 입력,
예열 5초 뒤 60초 측정에서 다음 결과를 얻었다. 다른 부하 시험은 동시에 실행하지 않았다.

| 항목 | 결과 |
| --- | ---: |
| 표시 FPS | 25.75 |
| 미리보기 → 결과 표시 지연 중앙값 / p95 | 166.3 / 194.6ms |
| 서버 처리 p95 / 최대 | 18.8 / 161.56ms |
| Chrome freeze | 2회, 총 0.92초 |
| 100ms 초과 표시 간격 | 23회 |
| RTP 수신 손실 | 5개 |

1623개 frame telemetry 모두 4얼굴 swapped였고 blur fallback은 없었다.
긴 프레임의 탐지 4.84ms·합성 10.48ms만으로 서버 전체 161.56ms를 설명할 수 없었다.
현재 수집한 단계 정보만으로 나머지 지연과 freeze 원인을 확정하지 않았다.
따라서 끊김 없는 30 FPS나 병목 완전 제거를 주장하지 않으며, 이 항목은 PR의 미완료
검증으로 남긴다. 실제 물리 카메라·다른 회선·장시간 다중 사용자도 별도 검증 대상이다.
운영 서비스는 정상 응답했고 검사 후 active peers 0, 자동 재시작 0회를 확인했다.

## 프리셋 배포

`config/face_presets.json`은 `natural_faces_v2` 560장의 새 SHA-256을 참조한다.
이미지와 모델, 생성된 가중치·engine, 인증서·측정 원본은 Git에 포함하지 않는다.
배포 시 새 `dataset_manifest.csv`와 `images/`가 포함된 배포본을 함께 설치해야 한다.
기존 560장과 카탈로그는 Mac의 `~/Downloads/innolive-face-presets-backup-20261019-231541`
아래 보존했고 모든 파일 해시를 대조했다. `best.pt`도 변경하지 않았다.

3090에서도 `~/Downloads/innolive-face-presets-backup-20261019-144532`에 기존 세트를
보존하고 해시를 검증했다. 새 카탈로그 SHA-256은
`799e042fc40421ec70ff04f32e49bb55368203d0f2805cfeef8b538e0222166d`이다.
운영 코드 `a263b49`와 새 560장을 함께 배포하고 40개 기본 source 예열 후 SERVING을 확인했다.
현재 NVIDIA 커널은 `595.99.02`이며 시스템 라이브러리로 CUDA와 NVENC를 사용했다.
이전 커널용 `595.91.07` 서비스 라이브러리를 강제로 지정하면 CUDA error 803이 발생하므로
그 경로를 수동 지정하지 않는다. launcher는 실제 로드된 커널 버전에 맞는 경로만 선택한다.

새 프리셋을 사용한 별도 raw gRPC 검증은 같은 640×480·예열 4초·측정 12초 조건에서
1명 × 1세션 104.25 FPS, 8명 × 1세션 34.08 FPS, 4명 × 2세션 각 37.83~37.92 FPS였다.
모든 측정 응답에서 해당 얼굴 수 전체가 합성됐다. 프리셋과 입력이 바뀐 결과이므로
기존 프리셋의 전후 표와 직접적인 개선율로 비교하지 않는다.

기존 이미지 디렉토리가 다른 세트이면 importer는 덮어쓰기를 거부한다. 기존 세트를 백업한 후
새 배포본을 import하고, 준비된 cache와 카탈로그가 일치하는 상태에서 AI 서버를 재시작한다.
자세한 명령은 [CUDA 운영 가이드](mobilefaceswap-cuda.md)를 따른다.

## 재현 명령과 측정 한계

```bash
# 3090 내부, 다른 부하 시험과 겹치지 않게 실행
.venv/bin/python scripts/benchmark_face_capacity.py \
  --target 127.0.0.1:50062 --cases 1x1,2x1,4x1,8x1,1x4,4x2 \
  --warmup 4 --seconds 12 --output reports/capacity
.venv/bin/python scripts/benchmark_mobilefaceswap_render.py \
  --manifest config/face_presets.json --output reports/render-parity
```

측정 원본은 로컬과 3090의 `reports/mobilefaceswap-production-66` 또는 검수 디렉토리
`reports`에 보관한다. GPU 가중치 캐시·전송 묶음은 stream을 넘어서 동시에 재사용하지 않도록
기존 bounded face executor를 유지한다. 세션을 무제한 병렬 실행하면 이 메모리 소유권과
지연 상한이 깨질 수 있다. 탐지·합성·회선·브라우저 병목이 모두 없어졌다는 보장은 하지 않는다.
