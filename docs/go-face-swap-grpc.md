# Go 서버 얼굴 합성 gRPC 연동

얼굴 합성 기능이 [PR #38](https://github.com/team-framework/innolive-ai/pull/38)로 병합된
`main` 기준이다. 기존 `/AiProcessor/ProcessVideo`에 **얼굴 처리 방식**과 **결과 반환 방식**을 지정하면 된다.
AI 서버도 이 기능이 포함된 버전으로 배포하고 [모델과 프리셋](face-swap-model-download.md)을 준비해야 한다.

AI 서버가 `--face-swap-backend mobilefaceswap`을 선택해도 아래 RPC와 요청·응답 계약은
동일하다. Go 요청에 모델 이름이나 새 enum을 추가하지 않는다. 기본 backend는 InSwapper이며,
모델 선택은 AI 서버 실행 설정으로 한다. CUDA 설치와 프리셋 사전 계산은
[MobileFaceSwap 운영 가이드](mobilefaceswap-cuda.md)를 따른다.

## 어떤 필드에 넣는가

매 프레임 보내는 **`VideoChunk` 메시지의 최상위 필드**에 넣는다.
최신 [ai_processor.proto](../protos/ai_processor.proto)로 기존 Go protobuf 생성 코드를 갱신한다.

| gRPC 요청 필드 | Go 구조체 필드 | 선택하는 내용 |
| --- | --- | --- |
| `anonymization_mode` (field 11) | `VideoChunk.AnonymizationMode` | 얼굴을 blur할지, 다른 얼굴로 합성할지 |
| `output_mode` (field 6) | `VideoChunk.OutputMode` | 처리한 영상도 받을지, 분석 정보만 받을지 |

## 값에 따라 무엇이 달라지는가

**`anonymization_mode`: 얼굴 처리 방식**

| 값 | 의미 | 영상에 적용할 처리 |
| --- | --- | --- |
| `0` 또는 생략 | 기존 blur | 얼굴을 흐리게 처리 |
| `1` | 얼굴 합성 | 서버가 프리셋 얼굴을 선택해 합성, 실패한 얼굴은 blur |
| `2` | 얼굴 속성 확인 | 성별·나이·안경·표정을 분석하고 프리셋을 매칭, 영상은 blur |

**`output_mode`: 결과 반환 방식**

| 값 | 의미 | 응답 `data` |
| --- | --- | --- |
| `0` 또는 생략 | 처리한 영상 반환, `2`와 같은 동작 | 결과 영상 bytes |
| `1` | 분석 정보만 반환, 얼굴 합성 실행 생략 | 비어 있음 |
| `2` | 처리한 영상 반환 | 결과 영상 bytes |

**얼굴이 합성된 영상을 받으려면 `anonymization_mode=1`, `output_mode=2`를 함께 보낸다.**
Go enum의 `MOSAIC_JPEG`는 얼굴 합성 결과를 받을 때도 사용하는 영상 반환 옵션이다.

## Go 요청 예제

기존 프레임 전송 루프에서 다음과 같이 요청을 만든다. `pb`는 기존 protobuf 패키지 alias다.

```go
req := &pb.VideoChunk{
	Data:              jpegBytes, // 기존 JPEG 프레임 bytes
	Timestamp:         timestamp,
	FrameId:           frameID,
	SessionId:         sessionID,
	AnonymizationMode: pb.FaceAnonymizationMode_FACE_ANONYMIZATION_MODE_FACE_SWAP, // 1: 합성
	OutputMode:        pb.VideoOutputMode_VIDEO_OUTPUT_MODE_MOSAIC_JPEG,         // 2: 영상 반환
}
// 기존 stream.Send(req)와 Recv() 흐름을 사용한다.
```

- **매 프레임 같은 모드를 지정한다.** `session_id`, `anonymization_mode`, `output_mode`는 한 RPC 동안 고정이다.
  모드를 바꿀 때는 스트림을 닫고 새 `ProcessVideo`를 호출한다. 중간 변경이나 이후 프레임의 합성 모드 생략은 `INVALID_ARGUMENT`이다.
- 결과 영상은 기존 `resp.GetData()`로 받는다. JPEG 입력은 JPEG로, 기존 raw `yuv420p` 입력은 raw로 반환된다.
  raw 입력의 `pix_fmt`, `width`, `height`도 기존대로 전달한다. deprecated `mosaic_jpeg` 필드는 사용하지 않는다.
- 얼굴별 `faces[].anonymization.status`가 `swapped`이면 합성 성공, `blur_fallback`이면 blur 대체다.
  대체 사유는 `fallback_reason`에 들어간다. whitelist 얼굴과 번호판은 기존 처리 정책을 유지한다.
- `resp.GetErrorCode()`가 있으면 프레임 오류로 처리한다. 보호 처리 실패 시 영상 `data`는 비어 있다.
- 기존 Go 클라이언트는 새 AI 서버에서도 기본 blur로 동작한다.
  구버전 AI 서버는 새 필드를 무시할 수 있으므로, 합성 대상 얼굴의 `anonymization`이 없으면 AI 서버 버전을 확인한다.
