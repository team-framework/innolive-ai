from google.protobuf import empty_pb2 as _empty_pb2
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class FaceAnonymizationMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    FACE_ANONYMIZATION_MODE_BLUR: _ClassVar[FaceAnonymizationMode]
    FACE_ANONYMIZATION_MODE_FACE_SWAP: _ClassVar[FaceAnonymizationMode]
    FACE_ANONYMIZATION_MODE_FACE_METADATA: _ClassVar[FaceAnonymizationMode]

class VideoOutputMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    VIDEO_OUTPUT_MODE_UNSPECIFIED: _ClassVar[VideoOutputMode]
    VIDEO_OUTPUT_MODE_METADATA_ONLY: _ClassVar[VideoOutputMode]
    VIDEO_OUTPUT_MODE_MOSAIC_JPEG: _ClassVar[VideoOutputMode]

class VodJobState(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    VOD_JOB_STATE_UNSPECIFIED: _ClassVar[VodJobState]
    VOD_JOB_STATE_PROCESSING: _ClassVar[VodJobState]
    VOD_JOB_STATE_PAUSED: _ClassVar[VodJobState]
    VOD_JOB_STATE_DONE: _ClassVar[VodJobState]
    VOD_JOB_STATE_FAILED: _ClassVar[VodJobState]
    VOD_JOB_STATE_CANCELLED: _ClassVar[VodJobState]

class VodQuality(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    VOD_QUALITY_STANDARD: _ClassVar[VodQuality]
    VOD_QUALITY_HIGH: _ClassVar[VodQuality]

class VodPauseReason(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    VOD_PAUSE_REASON_UNSPECIFIED: _ClassVar[VodPauseReason]
    VOD_PAUSE_REASON_REQUESTED: _ClassVar[VodPauseReason]
    VOD_PAUSE_REASON_WORKER_DRAINING: _ClassVar[VodPauseReason]
    VOD_PAUSE_REASON_RUNTIME_RECOVERY: _ClassVar[VodPauseReason]
FACE_ANONYMIZATION_MODE_BLUR: FaceAnonymizationMode
FACE_ANONYMIZATION_MODE_FACE_SWAP: FaceAnonymizationMode
FACE_ANONYMIZATION_MODE_FACE_METADATA: FaceAnonymizationMode
VIDEO_OUTPUT_MODE_UNSPECIFIED: VideoOutputMode
VIDEO_OUTPUT_MODE_METADATA_ONLY: VideoOutputMode
VIDEO_OUTPUT_MODE_MOSAIC_JPEG: VideoOutputMode
VOD_JOB_STATE_UNSPECIFIED: VodJobState
VOD_JOB_STATE_PROCESSING: VodJobState
VOD_JOB_STATE_PAUSED: VodJobState
VOD_JOB_STATE_DONE: VodJobState
VOD_JOB_STATE_FAILED: VodJobState
VOD_JOB_STATE_CANCELLED: VodJobState
VOD_QUALITY_STANDARD: VodQuality
VOD_QUALITY_HIGH: VodQuality
VOD_PAUSE_REASON_UNSPECIFIED: VodPauseReason
VOD_PAUSE_REASON_REQUESTED: VodPauseReason
VOD_PAUSE_REASON_WORKER_DRAINING: VodPauseReason
VOD_PAUSE_REASON_RUNTIME_RECOVERY: VodPauseReason

class VideoChunk(_message.Message):
    __slots__ = ("data", "timestamp", "frame_id", "batch_size", "session_id", "output_mode", "mosaic_config", "width", "height", "pix_fmt", "anonymization_mode")
    DATA_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    BATCH_SIZE_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_MODE_FIELD_NUMBER: _ClassVar[int]
    MOSAIC_CONFIG_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    PIX_FMT_FIELD_NUMBER: _ClassVar[int]
    ANONYMIZATION_MODE_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    timestamp: int
    frame_id: int
    batch_size: int
    session_id: str
    output_mode: VideoOutputMode
    mosaic_config: MosaicConfig
    width: int
    height: int
    pix_fmt: str
    anonymization_mode: FaceAnonymizationMode
    def __init__(self, data: _Optional[bytes] = ..., timestamp: _Optional[int] = ..., frame_id: _Optional[int] = ..., batch_size: _Optional[int] = ..., session_id: _Optional[str] = ..., output_mode: _Optional[_Union[VideoOutputMode, str]] = ..., mosaic_config: _Optional[_Union[MosaicConfig, _Mapping]] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., pix_fmt: _Optional[str] = ..., anonymization_mode: _Optional[_Union[FaceAnonymizationMode, str]] = ...) -> None: ...

class FaceAttributes(_message.Message):
    __slots__ = ("gender", "age", "glasses", "exp", "confidence")
    class ConfidenceEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(self, key: _Optional[str] = ..., value: _Optional[float] = ...) -> None: ...
    GENDER_FIELD_NUMBER: _ClassVar[int]
    AGE_FIELD_NUMBER: _ClassVar[int]
    GLASSES_FIELD_NUMBER: _ClassVar[int]
    EXP_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    gender: str
    age: str
    glasses: str
    exp: str
    confidence: _containers.ScalarMap[str, float]
    def __init__(self, gender: _Optional[str] = ..., age: _Optional[str] = ..., glasses: _Optional[str] = ..., exp: _Optional[str] = ..., confidence: _Optional[_Mapping[str, float]] = ...) -> None: ...

class FaceAnonymization(_message.Message):
    __slots__ = ("attributes", "identity_key", "preset_key", "status", "fallback_reason")
    ATTRIBUTES_FIELD_NUMBER: _ClassVar[int]
    IDENTITY_KEY_FIELD_NUMBER: _ClassVar[int]
    PRESET_KEY_FIELD_NUMBER: _ClassVar[int]
    STATUS_FIELD_NUMBER: _ClassVar[int]
    FALLBACK_REASON_FIELD_NUMBER: _ClassVar[int]
    attributes: FaceAttributes
    identity_key: str
    preset_key: str
    status: str
    fallback_reason: str
    def __init__(self, attributes: _Optional[_Union[FaceAttributes, _Mapping]] = ..., identity_key: _Optional[str] = ..., preset_key: _Optional[str] = ..., status: _Optional[str] = ..., fallback_reason: _Optional[str] = ...) -> None: ...

class MosaicConfig(_message.Message):
    __slots__ = ("blur_radius", "pixel_size", "mask_margin")
    BLUR_RADIUS_FIELD_NUMBER: _ClassVar[int]
    PIXEL_SIZE_FIELD_NUMBER: _ClassVar[int]
    MASK_MARGIN_FIELD_NUMBER: _ClassVar[int]
    blur_radius: float
    pixel_size: int
    mask_margin: int
    def __init__(self, blur_radius: _Optional[float] = ..., pixel_size: _Optional[int] = ..., mask_margin: _Optional[int] = ...) -> None: ...

class ProcessedVideoChunk(_message.Message):
    __slots__ = ("data", "timestamp", "status_message", "faces", "width", "height", "frame_id", "processing_ms", "timing", "error_code", "error_message", "stats", "mosaic_jpeg")
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
    MOSAIC_JPEG_FIELD_NUMBER: _ClassVar[int]
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
    mosaic_jpeg: bytes
    def __init__(self, data: _Optional[bytes] = ..., timestamp: _Optional[int] = ..., status_message: _Optional[str] = ..., faces: _Optional[_Iterable[_Union[FaceMetadata, _Mapping]]] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., frame_id: _Optional[int] = ..., processing_ms: _Optional[float] = ..., timing: _Optional[_Union[ProcessingTiming, _Mapping]] = ..., error_code: _Optional[str] = ..., error_message: _Optional[str] = ..., stats: _Optional[_Union[FrameStats, _Mapping]] = ..., mosaic_jpeg: _Optional[bytes] = ...) -> None: ...

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
    __slots__ = ("bbox", "confidence", "polygon", "track_id", "source", "held", "hold_frames", "class_name", "mask_area_px", "whitelisted", "anonymization")
    BBOX_FIELD_NUMBER: _ClassVar[int]
    CONFIDENCE_FIELD_NUMBER: _ClassVar[int]
    POLYGON_FIELD_NUMBER: _ClassVar[int]
    TRACK_ID_FIELD_NUMBER: _ClassVar[int]
    SOURCE_FIELD_NUMBER: _ClassVar[int]
    HELD_FIELD_NUMBER: _ClassVar[int]
    HOLD_FRAMES_FIELD_NUMBER: _ClassVar[int]
    CLASS_NAME_FIELD_NUMBER: _ClassVar[int]
    MASK_AREA_PX_FIELD_NUMBER: _ClassVar[int]
    WHITELISTED_FIELD_NUMBER: _ClassVar[int]
    ANONYMIZATION_FIELD_NUMBER: _ClassVar[int]
    bbox: BoundingBox
    confidence: float
    polygon: _containers.RepeatedCompositeFieldContainer[Point]
    track_id: int
    source: str
    held: bool
    hold_frames: int
    class_name: str
    mask_area_px: float
    whitelisted: bool
    anonymization: FaceAnonymization
    def __init__(self, bbox: _Optional[_Union[BoundingBox, _Mapping]] = ..., confidence: _Optional[float] = ..., polygon: _Optional[_Iterable[_Union[Point, _Mapping]]] = ..., track_id: _Optional[int] = ..., source: _Optional[str] = ..., held: _Optional[bool] = ..., hold_frames: _Optional[int] = ..., class_name: _Optional[str] = ..., mask_area_px: _Optional[float] = ..., whitelisted: _Optional[bool] = ..., anonymization: _Optional[_Union[FaceAnonymization, _Mapping]] = ...) -> None: ...

class FrameStats(_message.Message):
    __slots__ = ("detections", "raw_detections", "continuation_candidates", "detector_backed_tracks", "low_confidence_continuations", "held_tracks", "tracks", "tracker_frame", "adaface_calls", "adaface_queue_overflow", "whitelisted_tracks")
    DETECTIONS_FIELD_NUMBER: _ClassVar[int]
    RAW_DETECTIONS_FIELD_NUMBER: _ClassVar[int]
    CONTINUATION_CANDIDATES_FIELD_NUMBER: _ClassVar[int]
    DETECTOR_BACKED_TRACKS_FIELD_NUMBER: _ClassVar[int]
    LOW_CONFIDENCE_CONTINUATIONS_FIELD_NUMBER: _ClassVar[int]
    HELD_TRACKS_FIELD_NUMBER: _ClassVar[int]
    TRACKS_FIELD_NUMBER: _ClassVar[int]
    TRACKER_FRAME_FIELD_NUMBER: _ClassVar[int]
    ADAFACE_CALLS_FIELD_NUMBER: _ClassVar[int]
    ADAFACE_QUEUE_OVERFLOW_FIELD_NUMBER: _ClassVar[int]
    WHITELISTED_TRACKS_FIELD_NUMBER: _ClassVar[int]
    detections: int
    raw_detections: int
    continuation_candidates: int
    detector_backed_tracks: int
    low_confidence_continuations: int
    held_tracks: int
    tracks: int
    tracker_frame: int
    adaface_calls: int
    adaface_queue_overflow: int
    whitelisted_tracks: int
    def __init__(self, detections: _Optional[int] = ..., raw_detections: _Optional[int] = ..., continuation_candidates: _Optional[int] = ..., detector_backed_tracks: _Optional[int] = ..., low_confidence_continuations: _Optional[int] = ..., held_tracks: _Optional[int] = ..., tracks: _Optional[int] = ..., tracker_frame: _Optional[int] = ..., adaface_calls: _Optional[int] = ..., adaface_queue_overflow: _Optional[int] = ..., whitelisted_tracks: _Optional[int] = ...) -> None: ...

class FaceData(_message.Message):
    __slots__ = ("data", "session_id")
    DATA_FIELD_NUMBER: _ClassVar[int]
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    data: bytes
    session_id: str
    def __init__(self, data: _Optional[bytes] = ..., session_id: _Optional[str] = ...) -> None: ...

class WhitelistResponse(_message.Message):
    __slots__ = ("status_message", "timestamp", "entry_id", "entry_count", "whitelist_version")
    STATUS_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    TIMESTAMP_FIELD_NUMBER: _ClassVar[int]
    ENTRY_ID_FIELD_NUMBER: _ClassVar[int]
    ENTRY_COUNT_FIELD_NUMBER: _ClassVar[int]
    WHITELIST_VERSION_FIELD_NUMBER: _ClassVar[int]
    status_message: str
    timestamp: int
    entry_id: str
    entry_count: int
    whitelist_version: int
    def __init__(self, status_message: _Optional[str] = ..., timestamp: _Optional[int] = ..., entry_id: _Optional[str] = ..., entry_count: _Optional[int] = ..., whitelist_version: _Optional[int] = ...) -> None: ...

class GetWhitelistStatusRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class GetWhitelistStatusResponse(_message.Message):
    __slots__ = ("session_id", "entry_count", "whitelist_version", "entry_ids")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    ENTRY_COUNT_FIELD_NUMBER: _ClassVar[int]
    WHITELIST_VERSION_FIELD_NUMBER: _ClassVar[int]
    ENTRY_IDS_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    entry_count: int
    whitelist_version: int
    entry_ids: _containers.RepeatedScalarFieldContainer[str]
    def __init__(self, session_id: _Optional[str] = ..., entry_count: _Optional[int] = ..., whitelist_version: _Optional[int] = ..., entry_ids: _Optional[_Iterable[str]] = ...) -> None: ...

class DeleteWhitelistRequest(_message.Message):
    __slots__ = ("session_id", "entry_id")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    ENTRY_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    entry_id: str
    def __init__(self, session_id: _Optional[str] = ..., entry_id: _Optional[str] = ...) -> None: ...

class WhitelistEmbedding(_message.Message):
    __slots__ = ("entry_id", "values")
    ENTRY_ID_FIELD_NUMBER: _ClassVar[int]
    VALUES_FIELD_NUMBER: _ClassVar[int]
    entry_id: str
    values: _containers.RepeatedScalarFieldContainer[float]
    def __init__(self, entry_id: _Optional[str] = ..., values: _Optional[_Iterable[float]] = ...) -> None: ...

class ExportWhitelistRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class ExportWhitelistResponse(_message.Message):
    __slots__ = ("session_id", "embedding_model", "entries", "whitelist_version")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_MODEL_FIELD_NUMBER: _ClassVar[int]
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    WHITELIST_VERSION_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    embedding_model: str
    entries: _containers.RepeatedCompositeFieldContainer[WhitelistEmbedding]
    whitelist_version: int
    def __init__(self, session_id: _Optional[str] = ..., embedding_model: _Optional[str] = ..., entries: _Optional[_Iterable[_Union[WhitelistEmbedding, _Mapping]]] = ..., whitelist_version: _Optional[int] = ...) -> None: ...

class ImportWhitelistRequest(_message.Message):
    __slots__ = ("session_id", "embedding_model", "entries")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    EMBEDDING_MODEL_FIELD_NUMBER: _ClassVar[int]
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    embedding_model: str
    entries: _containers.RepeatedCompositeFieldContainer[WhitelistEmbedding]
    def __init__(self, session_id: _Optional[str] = ..., embedding_model: _Optional[str] = ..., entries: _Optional[_Iterable[_Union[WhitelistEmbedding, _Mapping]]] = ...) -> None: ...

class CreateSessionRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListSessionsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class DeleteSessionRequest(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class SessionInfo(_message.Message):
    __slots__ = ("session_id", "entry_count", "whitelist_version", "created_at_unix_ms", "active_stream_count")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    ENTRY_COUNT_FIELD_NUMBER: _ClassVar[int]
    WHITELIST_VERSION_FIELD_NUMBER: _ClassVar[int]
    CREATED_AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_STREAM_COUNT_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    entry_count: int
    whitelist_version: int
    created_at_unix_ms: int
    active_stream_count: int
    def __init__(self, session_id: _Optional[str] = ..., entry_count: _Optional[int] = ..., whitelist_version: _Optional[int] = ..., created_at_unix_ms: _Optional[int] = ..., active_stream_count: _Optional[int] = ...) -> None: ...

class ListSessionsResponse(_message.Message):
    __slots__ = ("sessions",)
    SESSIONS_FIELD_NUMBER: _ClassVar[int]
    sessions: _containers.RepeatedCompositeFieldContainer[SessionInfo]
    def __init__(self, sessions: _Optional[_Iterable[_Union[SessionInfo, _Mapping]]] = ...) -> None: ...

class VodWhitelist(_message.Message):
    __slots__ = ("embedding_model", "entries")
    EMBEDDING_MODEL_FIELD_NUMBER: _ClassVar[int]
    ENTRIES_FIELD_NUMBER: _ClassVar[int]
    embedding_model: str
    entries: _containers.RepeatedCompositeFieldContainer[WhitelistEmbedding]
    def __init__(self, embedding_model: _Optional[str] = ..., entries: _Optional[_Iterable[_Union[WhitelistEmbedding, _Mapping]]] = ...) -> None: ...

class VodOutputOptions(_message.Message):
    __slots__ = ("video_encoder", "quality", "force_audio_aac")
    VIDEO_ENCODER_FIELD_NUMBER: _ClassVar[int]
    QUALITY_FIELD_NUMBER: _ClassVar[int]
    FORCE_AUDIO_AAC_FIELD_NUMBER: _ClassVar[int]
    video_encoder: str
    quality: int
    force_audio_aac: bool
    def __init__(self, video_encoder: _Optional[str] = ..., quality: _Optional[int] = ..., force_audio_aac: _Optional[bool] = ...) -> None: ...

class StartVodJobRequest(_message.Message):
    __slots__ = ("job_id", "input_path", "output_path", "work_dir", "whitelist", "whitelist_session_id", "quality", "mosaic_config", "anonymization_mode", "output", "resume", "checkpoint_token", "overwrite_output", "segment_frames")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    INPUT_PATH_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_PATH_FIELD_NUMBER: _ClassVar[int]
    WORK_DIR_FIELD_NUMBER: _ClassVar[int]
    WHITELIST_FIELD_NUMBER: _ClassVar[int]
    WHITELIST_SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    QUALITY_FIELD_NUMBER: _ClassVar[int]
    MOSAIC_CONFIG_FIELD_NUMBER: _ClassVar[int]
    ANONYMIZATION_MODE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_FIELD_NUMBER: _ClassVar[int]
    RESUME_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_TOKEN_FIELD_NUMBER: _ClassVar[int]
    OVERWRITE_OUTPUT_FIELD_NUMBER: _ClassVar[int]
    SEGMENT_FRAMES_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    input_path: str
    output_path: str
    work_dir: str
    whitelist: VodWhitelist
    whitelist_session_id: str
    quality: VodQuality
    mosaic_config: MosaicConfig
    anonymization_mode: FaceAnonymizationMode
    output: VodOutputOptions
    resume: bool
    checkpoint_token: str
    overwrite_output: bool
    segment_frames: int
    def __init__(self, job_id: _Optional[str] = ..., input_path: _Optional[str] = ..., output_path: _Optional[str] = ..., work_dir: _Optional[str] = ..., whitelist: _Optional[_Union[VodWhitelist, _Mapping]] = ..., whitelist_session_id: _Optional[str] = ..., quality: _Optional[_Union[VodQuality, str]] = ..., mosaic_config: _Optional[_Union[MosaicConfig, _Mapping]] = ..., anonymization_mode: _Optional[_Union[FaceAnonymizationMode, str]] = ..., output: _Optional[_Union[VodOutputOptions, _Mapping]] = ..., resume: _Optional[bool] = ..., checkpoint_token: _Optional[str] = ..., overwrite_output: _Optional[bool] = ..., segment_frames: _Optional[int] = ...) -> None: ...

class VodJobRef(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class PauseVodJobRequest(_message.Message):
    __slots__ = ("job_id",)
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    def __init__(self, job_id: _Optional[str] = ...) -> None: ...

class CancelVodJobRequest(_message.Message):
    __slots__ = ("job_id", "purge_work_dir")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    PURGE_WORK_DIR_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    purge_work_dir: bool
    def __init__(self, job_id: _Optional[str] = ..., purge_work_dir: _Optional[bool] = ...) -> None: ...

class ListVodJobsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ListVodJobsResponse(_message.Message):
    __slots__ = ("jobs",)
    JOBS_FIELD_NUMBER: _ClassVar[int]
    jobs: _containers.RepeatedCompositeFieldContainer[VodJobStatus]
    def __init__(self, jobs: _Optional[_Iterable[_Union[VodJobStatus, _Mapping]]] = ...) -> None: ...

class VodProgress(_message.Message):
    __slots__ = ("frames_done", "frames_total", "frames_total_estimated", "fraction", "segments_done", "segments_total", "processing_fps", "eta_seconds")
    FRAMES_DONE_FIELD_NUMBER: _ClassVar[int]
    FRAMES_TOTAL_FIELD_NUMBER: _ClassVar[int]
    FRAMES_TOTAL_ESTIMATED_FIELD_NUMBER: _ClassVar[int]
    FRACTION_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_DONE_FIELD_NUMBER: _ClassVar[int]
    SEGMENTS_TOTAL_FIELD_NUMBER: _ClassVar[int]
    PROCESSING_FPS_FIELD_NUMBER: _ClassVar[int]
    ETA_SECONDS_FIELD_NUMBER: _ClassVar[int]
    frames_done: int
    frames_total: int
    frames_total_estimated: bool
    fraction: float
    segments_done: int
    segments_total: int
    processing_fps: float
    eta_seconds: float
    def __init__(self, frames_done: _Optional[int] = ..., frames_total: _Optional[int] = ..., frames_total_estimated: _Optional[bool] = ..., fraction: _Optional[float] = ..., segments_done: _Optional[int] = ..., segments_total: _Optional[int] = ..., processing_fps: _Optional[float] = ..., eta_seconds: _Optional[float] = ...) -> None: ...

class VodJobStatus(_message.Message):
    __slots__ = ("job_id", "state", "progress", "checkpoint_token", "pause_reason", "error_code", "error_message", "output_path", "updated_at_unix_ms", "width", "height", "avg_frame_rate", "duration_seconds", "has_audio", "input_size_bytes", "worker_target")
    JOB_ID_FIELD_NUMBER: _ClassVar[int]
    STATE_FIELD_NUMBER: _ClassVar[int]
    PROGRESS_FIELD_NUMBER: _ClassVar[int]
    CHECKPOINT_TOKEN_FIELD_NUMBER: _ClassVar[int]
    PAUSE_REASON_FIELD_NUMBER: _ClassVar[int]
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    OUTPUT_PATH_FIELD_NUMBER: _ClassVar[int]
    UPDATED_AT_UNIX_MS_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    AVG_FRAME_RATE_FIELD_NUMBER: _ClassVar[int]
    DURATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    HAS_AUDIO_FIELD_NUMBER: _ClassVar[int]
    INPUT_SIZE_BYTES_FIELD_NUMBER: _ClassVar[int]
    WORKER_TARGET_FIELD_NUMBER: _ClassVar[int]
    job_id: str
    state: VodJobState
    progress: VodProgress
    checkpoint_token: str
    pause_reason: VodPauseReason
    error_code: str
    error_message: str
    output_path: str
    updated_at_unix_ms: int
    width: int
    height: int
    avg_frame_rate: float
    duration_seconds: float
    has_audio: bool
    input_size_bytes: int
    worker_target: str
    def __init__(self, job_id: _Optional[str] = ..., state: _Optional[_Union[VodJobState, str]] = ..., progress: _Optional[_Union[VodProgress, _Mapping]] = ..., checkpoint_token: _Optional[str] = ..., pause_reason: _Optional[_Union[VodPauseReason, str]] = ..., error_code: _Optional[str] = ..., error_message: _Optional[str] = ..., output_path: _Optional[str] = ..., updated_at_unix_ms: _Optional[int] = ..., width: _Optional[int] = ..., height: _Optional[int] = ..., avg_frame_rate: _Optional[float] = ..., duration_seconds: _Optional[float] = ..., has_audio: _Optional[bool] = ..., input_size_bytes: _Optional[int] = ..., worker_target: _Optional[str] = ...) -> None: ...
