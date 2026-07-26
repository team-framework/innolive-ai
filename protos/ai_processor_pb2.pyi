from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class VideoChunk(_message.Message):
    __slots__ = ("data", "timestamp", "frame_id", "batch_size")
    DATA_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    timestamp: int
    frame_id: int
    batch_size: int
    def __init__(self, data: _Optional[bytes] = ..., timestamp: _Optional[int] = ..., frame_id: _Optional[int] = ..., batch_size: _Optional[int] = ...) -> None: ...

class ProcessedVideoChunk(_message.Message):
    __slots__ = ("data", "timestamp", "status_message", "faces", "width", "height", "frame_id", "processing_ms", "timing", "error_code", "error_message", "stats")
    DATA_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    STATUS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    FACES_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_MS_FIELD_NUMBER: _ClassVar[int]
    TIMING_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    STATS_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    timestamp: int
    status_message: str
    faces: _containers.RepeatedCompositeFieldContainer[FaceMetadata]
    width: int
    height: int
    frame_id: int
    processing_ms: float
    timing: ProcessingTiming
    error_code: str
    error_message: str
    stats: FrameStats
    def __init__(self, data: _Optional[bytes] = ..., timestamp: _Optional[int] = ..., status_message: _Optional[str] = ..., faces: _Optional[_Iterable[_Union[FaceMetadata, _Mapping]]] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., frame_id: _Optional[int] = ..., processing_ms: _Optional[float] = ..., timing: _Optional[_Union[ProcessingTiming, _Mapping]] = ..., error_code: _Optional[str] = ..., error_message: _Optional[str] = ..., stats: _Optional[_Union[FrameStats, _Mapping]] = ...) -> None: ...

class ProcessingTiming(_message.Message):
    __slots__ = ("queue_ms", "decode_ms", "inference_ms", "tracking_ms", "blur_encode_ms", "inference_batch_size", "serialize_ms", "server_total_ms", "runtime_total_ms")
    QUEUE_MS_FIELD_NUMBER: _ClassVar[int]
    DECODE_MS_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_MS_FIELD_NUMBER: _ClassVar[int]
    TRACKING_MS_FIELD_NUMBER: _ClassVar[int]
    BLUR_ENCODE_MS_FIELD_NUMBER: _ClassVar[int]
    INFERENCE_BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    SERIALIZE_MS_FIELD_NUMBER: _ClassVar[int]
    SERVER_TOTAL_MS_FIELD_NUMBER: _ClassVar[int]
    RUNTIME_TOTAL_MS_FIELD_NUMBER: _ClassVar[int]
    queue_ms: float
    decode_ms: float
    inference_ms: float
    tracking_ms: float
    blur_encode_ms: float
    inference_batch_size: int
    serialize_ms: float
    server_total_ms: float
    runtime_total_ms: float
    def __init__(self, queue_ms: _Optional[float] = ..., decode_ms: _Optional[float] = ..., inference_ms: _Optional[float] = ..., tracking_ms: _Optional[float] = ..., blur_encode_ms: _Optional[float] = ..., inference_batch_size: _Optional[int] = ..., serialize_ms: _Optional[float] = ..., server_total_ms: _Optional[float] = ..., runtime_total_ms: _Optional[float] = ...) -> None: ...

class Point(_message.Message):
    __slots__ = ("x", "y")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    def __init__(self, x: _Optional[float] = ..., y: _Optional[float] = ...) -> None: ...

class BoundingBox(_message.Message):
    __slots__ = ("x1", "y1", "x2", "y2")
    X1_FIELD_NUMBER: _ClassVar[int]
    Y1_FIELD_NUMBER: _ClassVar[int]
    X2_FIELD_NUMBER: _ClassVar[int]
    Y2_FIELD_NUMBER: _ClassVar[int]
    x1: float
    y1: float
    x2: float
    y2: float
    def __init__(self, x1: _Optional[float] = ..., y1: _Optional[float] = ..., x2: _Optional[float] = ..., y2: _Optional[float] = ...) -> None: ...

class FaceMetadata(_message.Message):
    __slots__ = ("bbox", "confidence", "polygon", "track_id", "source", "held", "hold_frames", "class_name", "mask_area_px")
    BBOX_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    POLYGON_FIELD_NUMBER: _ClassVar[int]
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    HELD_FIELD_NUMBER: _ClassVar[int]
    HOLD_FRAMES_FIELD_NUMBER: _ClassVar[int]
    CLASS_NAME_FIELD_NUMBER: _ClassVar[int]
    MASK_AREA_PX_FIELD_NUMBER: _ClassVar[int]
    bbox: BoundingBox
    confidence: float
    polygon: _containers.RepeatedCompositeFieldContainer[Point]
    track_id: int
    source: str
    held: bool
    hold_frames: int
    class_name: str
    mask_area_px: float
    def __init__(self, bbox: _Optional[_Union[BoundingBox, _Mapping]] = ..., confidence: _Optional[float] = ..., polygon: _Optional[_Iterable[_Union[Point, _Mapping]]] = ..., track_id: _Optional[int] = ..., source: _Optional[str] = ..., held: _Optional[bool] = ..., hold_frames: _Optional[int] = ..., class_name: _Optional[str] = ..., mask_area_px: _Optional[float] = ...) -> None: ...

class FrameStats(_message.Message):
    __slots__ = ("detections", "raw_detections", "continuation_candidates", "detector_backed_tracks", "low_confidence_continuations", "held_tracks", "tracks", "tracker_frame")
    DETECTIONS_FIELD_NUMBER: _ClassVar[int]
    RAW_DETECTIONS_FIELD_NUMBER: _ClassVar[int]
    CONTINUATION_CANDIDATES_FIELD_NUMBER: _ClassVar[int]
    DETECTOR_BACKED_TRACKS_FIELD_NUMBER: _ClassVar[int]
    LOW_CONFIDENCE_CONTINUATIONS_FIELD_NUMBER: _ClassVar[int]
    HELD_TRACKS_FIELD_NUMBER: _ClassVar[int]
    TRACKS_FIELD_NUMBER: _ClassVar[int]
    TRACKER_FRAME_FIELD_NUMBER: _ClassVar[int]
    detections: int
    raw_detections: int
    continuation_candidates: int
    detector_backed_tracks: int
    low_confidence_continuations: int
    held_tracks: int
    tracks: int
    tracker_frame: int
    def __init__(self, detections: _Optional[int] = ..., raw_detections: _Optional[int] = ..., continuation_candidates: _Optional[int] = ..., detector_backed_tracks: _Optional[int] = ..., low_confidence_continuations: _Optional[int] = ..., held_tracks: _Optional[int] = ..., tracks: _Optional[int] = ..., tracker_frame: _Optional[int] = ...) -> None: ...

class FaceData(_message.Message):
    __slots__ = ("data",)
    DATA_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    def __init__(self, data: _Optional[bytes] = ...) -> None: ...

class WhitelistResponse(_message.Message):
    __slots__ = ("status_message", "timestamp")
    STATUS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    status_message: str
    timestamp: int
    def __init__(self, status_message: _Optional[str] = ..., timestamp: _Optional[int] = ...) -> None: ...
