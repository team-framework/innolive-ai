#!/usr/bin/env python3
"""Production gRPC entry point for the B1-640 face video pipeline."""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

import grpc
import numpy as np
from google.protobuf import empty_pb2
from grpc_health.v1 import health, health_pb2, health_pb2_grpc

from protos import ai_processor_pb2 as messages
from protos import ai_processor_pb2_grpc
from service.adaface_backbones import ADAFACE_ARCHITECTURES
from service.adaface_model import (
    DEFAULT_ADAFACE_ARCHITECTURE,
    DEFAULT_ADAFACE_WEIGHTS,
    DEFAULT_FACE_DETECTOR,
    AdaFaceConfig,
    AdaFaceRuntime,
    AdaFaceUnavailable,
    FaceAlignmentError,
    FaceCountError,
    FaceTooSmallError,
)
from service.frame import MAX_LONG_EDGE, MIN_FRAME_DIMENSION, FrameLimits, decode_image, decode_jpeg
from service.grpc_config import listen_address, server_options
from service.mosaic import mosaic_jpeg
from service.protocol import MAX_GRPC_RESPONSE_BYTES, MAX_JPEG_BYTES
from service.recognition import (
    RecognitionConfig,
    SessionInUseError,
    SessionLimitError,
    SessionNotFoundError,
    SessionRegistry,
    SessionState,
    SessionSummary,
    StreamRecognition,
    WhitelistEntryNotFoundError,
    WhitelistLimitError,
    validate_entry_id,
    validate_session_id,
)
from service.runtime import (
    BACKENDS,
    DEFAULT_CHECKPOINT,
    DEFAULT_ENGINE,
    InferenceFailure,
    RuntimeConfig,
    RuntimeManager,
    TrackingFailure,
)
from service.tracking import StreamTracker

LOGGER = logging.getLogger("innolive.grpc")
ROOT = Path(__file__).resolve().parent
DEFAULT_TRACKER = ROOT / "config" / "botsort.yaml"
SERVICE_NAME = "AiProcessor"
PROFILE = "B1-640-Q90-W5"
MAX_SESSIONS = 1_024
MOSAIC_MAX_INFLIGHT = 2


def _output_mode(value: int) -> int:
    if value == messages.VIDEO_OUTPUT_MODE_UNSPECIFIED:
        return messages.VIDEO_OUTPUT_MODE_MOSAIC_JPEG
    if value in {
        messages.VIDEO_OUTPUT_MODE_METADATA_ONLY,
        messages.VIDEO_OUTPUT_MODE_MOSAIC_JPEG,
    }:
        return value
    raise ValueError("output_mode is not supported")


@dataclass(frozen=True, slots=True)
class GrpcServerSettings:
    runtime: RuntimeConfig
    adaface: AdaFaceConfig = field(default_factory=AdaFaceConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    tracker_config: Path = DEFAULT_TRACKER
    host: str = "127.0.0.1"
    port: int = 50_051
    max_sessions: int = MAX_SESSIONS
    max_whitelist_entries: int = 32
    max_jpeg_bytes: int = MAX_JPEG_BYTES
    max_long_edge: int = MAX_LONG_EDGE
    inference_timeout_seconds: float = 1.5
    shutdown_grace_seconds: float = 5.0
    ssl_certfile: Path | None = None
    ssl_keyfile: Path | None = None

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("host must not be empty")
        if not 0 <= self.port <= 65_535:
            raise ValueError("port must be between 0 and 65535")
        if not 1 <= self.max_sessions <= MAX_SESSIONS:
            raise ValueError(f"max_sessions must be in 1..{MAX_SESSIONS}")
        if self.max_whitelist_entries < 1:
            raise ValueError("max_whitelist_entries must be at least one")
        if not 1 <= self.max_jpeg_bytes <= MAX_JPEG_BYTES:
            raise ValueError(f"max_jpeg_bytes must be in 1..{MAX_JPEG_BYTES}")
        if self.max_long_edge < MIN_FRAME_DIMENSION:
            raise ValueError(f"max_long_edge must be at least {MIN_FRAME_DIMENSION}")
        if not math.isfinite(self.inference_timeout_seconds) or self.inference_timeout_seconds <= 0:
            raise ValueError("inference_timeout_seconds must be positive")
        if not math.isfinite(self.shutdown_grace_seconds) or self.shutdown_grace_seconds < 0:
            raise ValueError("shutdown_grace_seconds cannot be negative")
        if (self.ssl_certfile is None) != (self.ssl_keyfile is None):
            raise ValueError("ssl_certfile and ssl_keyfile must be provided together")


@dataclass(slots=True)
class FrameOutcome:
    response: Any
    fatal: bool = False
    pending_inference: asyncio.Task | None = None


@dataclass(slots=True)
class StreamState:
    session_id: str
    output_mode: int
    recognition: StreamRecognition
    session: SessionState | None = None


class AiProcessorServicer(ai_processor_pb2_grpc.AiProcessorServicer):
    """One ordered B1 inference/tracker lane per ProcessVideo RPC."""

    def __init__(
        self,
        runtime: Any,
        settings: GrpcServerSettings,
        *,
        sessions: SessionRegistry,
        adaface: Any | None,
        tracker_factory: Callable[..., Any] = StreamTracker,
        mark_unhealthy: Callable[[], Awaitable[None]] | None = None,
    ):
        self.runtime = runtime
        self.settings = settings
        self.sessions = sessions
        self.adaface = adaface
        self.tracker_factory = tracker_factory
        self.mark_unhealthy = mark_unhealthy
        self.active_streams = 0
        self.frame_limits = FrameLimits(
            max_jpeg_bytes=settings.max_jpeg_bytes,
            max_long_edge=settings.max_long_edge,
            max_pixels=settings.max_long_edge * settings.max_long_edge,
        )
        self._inference_tasks: set[asyncio.Task] = set()
        self._mosaic_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="frame-mosaic",
        )
        self._mosaic_slots = asyncio.BoundedSemaphore(MOSAIC_MAX_INFLIGHT)
        self._accepting = False

    async def ProcessVideo(self, request_iterator, context) -> AsyncIterator:
        if not self._accepting or not self.runtime.ready:
            await context.abort(grpc.StatusCode.UNAVAILABLE, "runtime is not ready")
        self.active_streams += 1

        tracker = None
        stream = None
        frame_sequence = 0
        pending_inference: asyncio.Task | None = None
        try:
            try:
                tracker = self.tracker_factory(
                    self.settings.tracker_config,
                    getattr(self.runtime, "device", self.settings.runtime.device),
                )
            except Exception:
                LOGGER.exception("tracker initialization failed")
                await self._mark_runtime_unhealthy()
                await context.abort(
                    grpc.StatusCode.INTERNAL,
                    "tracker initialization failed",
                )

            async for request in request_iterator:
                try:
                    request_session_id = validate_session_id(request.session_id)
                except ValueError as error:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
                try:
                    output_mode = _output_mode(request.output_mode)
                except ValueError as error:
                    await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
                if stream is None:
                    stream = StreamState(
                        session_id=request_session_id,
                        output_mode=output_mode,
                        recognition=StreamRecognition(
                            self.adaface,
                            self.settings.recognition,
                            owner=request_session_id,
                        ),
                    )
                elif request_session_id != stream.session_id:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "session_id cannot change within a ProcessVideo stream",
                    )
                elif output_mode != stream.output_mode:
                    await context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT,
                        "output_mode cannot change within a ProcessVideo stream",
                    )
                frame_sequence += 1
                try:
                    outcome = await self._process_request(
                        request,
                        tracker,
                        stream,
                        frame_sequence,
                    )
                except SessionLimitError as error:
                    await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(error))
                pending_inference = outcome.pending_inference
                yield outcome.response
                if outcome.fatal:
                    return
                pending_inference = None
        finally:
            if pending_inference is not None:
                await self._settle_inference(pending_inference)
            if stream is not None:
                stream.recognition.close()
                if stream.session is not None:
                    self.sessions.release_stream(stream.session)
            if tracker is not None:
                try:
                    tracker.reset()
                except Exception:
                    LOGGER.exception("tracker reset failed")
            self.active_streams = max(0, self.active_streams - 1)

    async def _process_request(
        self,
        request: Any,
        tracker: Any,
        stream: StreamState,
        frame_sequence: int,
    ) -> FrameOutcome:
        received_at = time.perf_counter()
        if request.batch_size not in (0, 1):
            return FrameOutcome(
                self._error_response(
                    request,
                    "INVALID_BATCH_SIZE",
                    "batch_size must be zero or one for the B1 profile",
                    received_at,
                )
            )

        try:
            image = await asyncio.to_thread(
                decode_jpeg,
                bytes(request.data),
                self.frame_limits,
            )
        except (TypeError, ValueError):
            LOGGER.info("frame %d JPEG decode rejected", request.frame_id)
            return FrameOutcome(
                self._error_response(
                    request,
                    "DECODE_FAILED",
                    "frame could not be decoded within the B1 limits",
                    received_at,
                )
            )
        decoded_at = time.perf_counter()

        if stream.session is None:
            stream.session = self.sessions.acquire_stream(stream.session_id)

        result, failure = await self._run_inference(request, image, tracker, received_at)
        if failure is not None:
            return failure
        try:
            objects = result.get("objects", [])
            if not isinstance(objects, list) or len(objects) > 100:
                raise ValueError("runtime objects exceed the response contract")
            result.update(
                stream.recognition.process(
                    image,
                    objects,
                    stream.session.snapshot(),
                    frame_sequence,
                )
            )
        except (TypeError, ValueError, OverflowError):
            LOGGER.exception("frame %d recognition result failed validation", request.frame_id)
            return FrameOutcome(
                self._error_response(
                    request,
                    "SERIALIZATION_FAILED",
                    "result metadata exceeded the serialization contract",
                    received_at,
                    inference_batch_size=1,
                ),
                fatal=True,
            )

        processed_data = b""
        blur_encode_ms = 0.0
        if stream.output_mode == messages.VIDEO_OUTPUT_MODE_MOSAIC_JPEG:
            mosaic_started = time.perf_counter()
            try:
                if any(item.get("whitelisted") is not True for item in objects):
                    async with self._mosaic_slots:
                        loop = asyncio.get_running_loop()
                        processed_data = await loop.run_in_executor(
                            self._mosaic_executor,
                            partial(
                                mosaic_jpeg,
                                image,
                                objects,
                                max_bytes=self.settings.max_jpeg_bytes,
                            ),
                        )
                else:
                    processed_data = bytes(request.data)
            except Exception:
                LOGGER.exception("frame %d mosaic composition failed", request.frame_id)
                return FrameOutcome(
                    self._error_response(
                        request,
                        "MOSAIC_FAILED",
                        "server-side mosaic composition failed",
                        received_at,
                        inference_batch_size=1,
                    ),
                    fatal=True,
                )
            blur_encode_ms = (time.perf_counter() - mosaic_started) * 1_000

        try:
            response = self._success_response(
                request,
                image,
                result,
                received_at,
                decoded_at,
                processed_data,
                blur_encode_ms,
            )
            if response.ByteSize() > MAX_GRPC_RESPONSE_BYTES:
                raise ValueError("protobuf response exceeds the byte limit")
            return FrameOutcome(response)
        except (TypeError, ValueError, OverflowError):
            LOGGER.exception("frame %d protobuf serialization failed", request.frame_id)
            return FrameOutcome(
                self._error_response(
                    request,
                    "SERIALIZATION_FAILED",
                    "result metadata exceeded the serialization contract",
                    received_at,
                    inference_batch_size=1,
                ),
                fatal=True,
            )

    async def _run_inference(
        self,
        request: Any,
        image: np.ndarray,
        tracker: Any,
        received_at: float,
    ) -> tuple[dict[str, Any] | None, FrameOutcome | None]:
        task = asyncio.create_task(
            self.runtime.infer(image, tracker),
            name=f"grpc-infer-{request.frame_id}",
        )
        self._inference_tasks.add(task)
        task.add_done_callback(self._inference_tasks.discard)
        try:
            result = await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self.settings.inference_timeout_seconds,
            )
            return result, None
        except TimeoutError:
            await self._mark_runtime_unhealthy()
            return None, self._fatal_inference_outcome(
                request,
                "INFERENCE_TIMEOUT",
                "frame inference exceeded the server timeout",
                received_at,
                pending=task,
            )
        except TrackingFailure:
            LOGGER.exception("frame %d tracking failed", request.frame_id)
            return None, self._fatal_inference_outcome(
                request,
                "TRACKING_FAILED",
                "frame tracking failed",
                received_at,
            )
        except InferenceFailure:
            LOGGER.exception("frame %d inference failed", request.frame_id)
            await self._mark_runtime_unhealthy()
            return None, self._fatal_inference_outcome(
                request,
                "INFERENCE_FAILED",
                "frame inference failed",
                received_at,
            )
        except asyncio.CancelledError:
            await self._settle_inference(task)
            raise
        except Exception:
            LOGGER.exception("frame %d processing failed", request.frame_id)
            await self._mark_runtime_unhealthy()
            return None, self._fatal_inference_outcome(
                request,
                "PROCESSING_FAILED",
                "frame processing failed",
                received_at,
            )

    def _fatal_inference_outcome(
        self,
        request: Any,
        code: str,
        message: str,
        received_at: float,
        *,
        pending: asyncio.Task | None = None,
    ) -> FrameOutcome:
        return FrameOutcome(
            self._error_response(
                request,
                code,
                message,
                received_at,
                inference_batch_size=1,
            ),
            fatal=True,
            pending_inference=pending,
        )

    async def AddWhitelist(self, request, context):
        try:
            session_id = validate_session_id(request.session_id)
        except ValueError as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        if not request.data:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, "face image must not be empty")
        if len(request.data) > self.settings.max_jpeg_bytes:
            await context.abort(
                grpc.StatusCode.RESOURCE_EXHAUSTED,
                f"face image exceeds the {self.settings.max_jpeg_bytes} byte limit",
            )
        if self.adaface is None or not getattr(self.adaface, "ready", False):
            await context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                "AdaFace is not available",
            )
        try:
            image = await asyncio.to_thread(
                decode_image,
                bytes(request.data),
                self.frame_limits,
            )
        except (TypeError, ValueError):
            await context.abort(
                grpc.StatusCode.INVALID_ARGUMENT,
                "face image must be a bounded JPEG, PNG, or WebP image",
            )

        future = self.adaface.submit_enrollment(image, owner=session_id)
        if future is None:
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, "AdaFace queue is full")
        try:
            embedding = await asyncio.wait_for(
                asyncio.shield(future),
                timeout=self.settings.inference_timeout_seconds,
            )
        except TimeoutError:
            await context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, "AdaFace inference timed out")
        except (FaceCountError, FaceTooSmallError, FaceAlignmentError) as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        except AdaFaceUnavailable as error:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        except Exception:
            LOGGER.exception("AdaFace whitelist enrollment failed")
            await context.abort(grpc.StatusCode.INTERNAL, "AdaFace inference failed")

        try:
            entry, entry_count, version = self.sessions.append(session_id, embedding)
        except SessionLimitError as error:
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(error))
        except WhitelistLimitError as error:
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(error))
        except ValueError:
            LOGGER.exception("AdaFace returned an invalid whitelist embedding")
            await context.abort(grpc.StatusCode.INTERNAL, "AdaFace returned an invalid embedding")
        return messages.WhitelistResponse(
            status_message="success",
            timestamp=time.time_ns(),
            entry_id=entry.entry_id,
            entry_count=entry_count,
            whitelist_version=version,
        )

    async def GetWhitelistStatus(self, request, context):
        try:
            session_id = validate_session_id(request.session_id)
        except ValueError as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        snapshot = self.sessions.snapshot(session_id)
        return messages.GetWhitelistStatusResponse(
            session_id=session_id,
            entry_count=len(snapshot.entries),
            whitelist_version=snapshot.version,
            entry_ids=[entry.entry_id for entry in snapshot.entries],
        )

    async def DeleteWhitelist(self, request, context):
        try:
            session_id = validate_session_id(request.session_id)
            entry_id = validate_entry_id(request.entry_id)
        except ValueError as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        try:
            deleted_entry_id, entry_count, version = self.sessions.delete_whitelist_entry(
                session_id, entry_id
            )
        except (SessionNotFoundError, WhitelistEntryNotFoundError) as error:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(error))
        return messages.WhitelistResponse(
            status_message="success",
            timestamp=time.time_ns(),
            entry_id=deleted_entry_id,
            entry_count=entry_count,
            whitelist_version=version,
        )

    async def CreateSession(self, request, context):
        del request
        try:
            summary = self.sessions.create()
        except SessionLimitError as error:
            await context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, str(error))
        return self._session_info(summary)

    async def ListSessions(self, request, context):
        del request
        del context
        return messages.ListSessionsResponse(
            sessions=[self._session_info(summary) for summary in self.sessions.list_summaries()]
        )

    async def DeleteSession(self, request, context):
        try:
            session_id = validate_session_id(request.session_id)
        except ValueError as error:
            await context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        try:
            self.sessions.delete(session_id)
        except SessionNotFoundError as error:
            await context.abort(grpc.StatusCode.NOT_FOUND, str(error))
        except SessionInUseError as error:
            await context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        return empty_pb2.Empty()

    @staticmethod
    def _session_info(summary: SessionSummary) -> messages.SessionInfo:
        return messages.SessionInfo(
            session_id=summary.session_id,
            entry_count=summary.entry_count,
            whitelist_version=summary.whitelist_version,
            created_at_unix_ms=summary.created_at_unix_ms,
            active_stream_count=summary.active_stream_count,
        )

    async def close(self) -> None:
        self._accepting = False
        tasks = tuple(self._inference_tasks)
        for task in tasks:
            await self._settle_inference(task)
        await asyncio.to_thread(
            self._mosaic_executor.shutdown,
            wait=True,
            cancel_futures=True,
        )

    def set_accepting(self, accepting: bool) -> None:
        self._accepting = accepting

    async def _mark_runtime_unhealthy(self) -> None:
        self._accepting = False
        if self.mark_unhealthy is not None:
            try:
                await self.mark_unhealthy()
            except Exception:
                LOGGER.exception("failed to publish NOT_SERVING health state")

    @staticmethod
    async def _settle_inference(task: asyncio.Task) -> None:
        """Do not release/reset tracker state before shielded GPU work settles."""

        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                continue
            except Exception:
                break
        with suppress(asyncio.CancelledError, Exception):
            task.result()

    def _success_response(
        self,
        request,
        image: np.ndarray,
        result: dict[str, Any],
        received_at: float,
        decoded_at: float,
        processed_data: bytes,
        blur_encode_ms: float,
    ):
        serialize_started = time.perf_counter()
        objects = result.get("objects", [])
        faces = [self._face_metadata(item) for item in objects]
        timing = result.get("timing_ms", {})
        response = messages.ProcessedVideoChunk(
            data=processed_data,
            timestamp=request.timestamp,
            status_message="success",
            faces=faces,
            width=int(image.shape[1]),
            height=int(image.shape[0]),
            frame_id=request.frame_id,
            timing=messages.ProcessingTiming(
                queue_ms=float(timing.get("queue", 0.0)),
                decode_ms=(decoded_at - received_at) * 1_000,
                inference_ms=float(timing.get("inference", 0.0)),
                tracking_ms=float(timing.get("tracking", 0.0)),
                blur_encode_ms=blur_encode_ms,
                inference_batch_size=1,
                runtime_total_ms=float(timing.get("runtime_total", 0.0)),
            ),
            stats=messages.FrameStats(
                detections=int(result.get("detections", 0)),
                raw_detections=int(result.get("raw_detections", 0)),
                continuation_candidates=int(result.get("continuation_candidates", 0)),
                detector_backed_tracks=int(result.get("detector_backed_tracks", 0)),
                low_confidence_continuations=int(result.get("low_confidence_continuations", 0)),
                held_tracks=int(result.get("held_tracks", 0)),
                tracks=int(result.get("tracks", 0)),
                tracker_frame=int(result.get("tracker_frame", 0)),
                adaface_calls=int(result.get("adaface_calls", 0)),
                adaface_queue_overflow=int(result.get("adaface_queue_overflow", 0)),
                whitelisted_tracks=int(result.get("whitelisted_tracks", 0)),
            ),
        )
        response.timing.serialize_ms = (
            float(timing.get("serialize", 0.0)) + (time.perf_counter() - serialize_started) * 1_000
        )
        response.timing.server_total_ms = (time.perf_counter() - received_at) * 1_000
        response.processing_ms = response.timing.server_total_ms
        return response

    @staticmethod
    def _face_metadata(item: dict[str, Any]):
        bbox = item.get("bbox")
        polygon = item.get("mask_polygon")
        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
            or not all(math.isfinite(float(value)) for value in bbox)
        ):
            raise ValueError("face bbox is invalid")
        if not isinstance(polygon, list) or not 3 <= len(polygon) <= 64:
            raise ValueError("face polygon is invalid")
        points = []
        for point in polygon:
            if (
                not isinstance(point, list)
                or len(point) != 2
                or not all(math.isfinite(float(value)) for value in point)
            ):
                raise ValueError("face polygon point is invalid")
            points.append(messages.Point(x=float(point[0]), y=float(point[1])))
        confidence = float(item.get("confidence", 0.0))
        mask_area = float(item.get("mask_area_px", 0.0))
        if not math.isfinite(confidence) or not math.isfinite(mask_area):
            raise ValueError("face confidence or area is invalid")
        face = messages.FaceMetadata(
            bbox=messages.BoundingBox(
                x1=float(bbox[0]),
                y1=float(bbox[1]),
                x2=float(bbox[2]),
                y2=float(bbox[3]),
            ),
            confidence=confidence,
            polygon=points,
            source=str(item.get("source", "detected")),
            held=bool(item.get("held", False)),
            hold_frames=int(item.get("hold_frames", 0)),
            class_name=str(item.get("class_name", "face")),
            mask_area_px=mask_area,
            whitelisted=bool(item.get("whitelisted", False)),
        )
        if "track_id" in item and item["track_id"] is not None:
            face.track_id = int(item["track_id"])
        return face

    @staticmethod
    def _error_response(
        request,
        code: str,
        message: str,
        received_at: float,
        *,
        inference_batch_size: int = 0,
    ):
        elapsed_ms = (time.perf_counter() - received_at) * 1_000
        return messages.ProcessedVideoChunk(
            data=b"",
            timestamp=request.timestamp,
            status_message="failed",
            frame_id=request.frame_id,
            processing_ms=elapsed_ms,
            timing=messages.ProcessingTiming(
                inference_batch_size=inference_batch_size,
                server_total_ms=elapsed_ms,
            ),
            error_code=code,
            error_message=message,
        )


@dataclass(slots=True)
class GrpcServerBundle:
    server: grpc.aio.Server
    servicer: AiProcessorServicer
    health_servicer: health.aio.HealthServicer
    bound_port: int

    async def set_serving(self, serving: bool) -> None:
        status = (
            health_pb2.HealthCheckResponse.SERVING
            if serving
            else health_pb2.HealthCheckResponse.NOT_SERVING
        )
        self.servicer.set_accepting(serving)
        await self.health_servicer.set("", status)
        await self.health_servicer.set(SERVICE_NAME, status)


def build_grpc_server(
    settings: GrpcServerSettings,
    runtime: Any,
    *,
    sessions: SessionRegistry | None = None,
    adaface: Any | None = None,
    tracker_factory: Callable[..., Any] = StreamTracker,
) -> GrpcServerBundle:
    server = grpc.aio.server(options=server_options())
    health_servicer = health.aio.HealthServicer()

    async def mark_unhealthy() -> None:
        status = health_pb2.HealthCheckResponse.NOT_SERVING
        await health_servicer.set("", status)
        await health_servicer.set(SERVICE_NAME, status)

    servicer = AiProcessorServicer(
        runtime,
        settings,
        sessions=sessions
        or SessionRegistry(
            max_sessions=settings.max_sessions,
            max_entries_per_session=settings.max_whitelist_entries,
        ),
        adaface=adaface,
        tracker_factory=tracker_factory,
        mark_unhealthy=mark_unhealthy,
    )
    ai_processor_pb2_grpc.add_AiProcessorServicer_to_server(servicer, server)
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)
    address = listen_address(settings.host, settings.port)
    if settings.ssl_certfile is not None:
        keyfile = settings.ssl_keyfile
        if keyfile is None:
            raise AssertionError("TLS key is required with a certificate")
        private_key = keyfile.read_bytes()
        certificate_chain = settings.ssl_certfile.read_bytes()
        credentials = grpc.ssl_server_credentials(((private_key, certificate_chain),))
        bound_port = server.add_secure_port(address, credentials)
    else:
        bound_port = server.add_insecure_port(address)
    if bound_port == 0:
        raise RuntimeError(f"failed to bind gRPC server to {address}")
    return GrpcServerBundle(server, servicer, health_servicer, bound_port)


async def serve(settings: GrpcServerSettings) -> None:
    runtime = None
    adaface = None
    bundle = None
    try:
        runtime = await asyncio.to_thread(RuntimeManager, settings.runtime)
        tracker_probe = await asyncio.to_thread(
            StreamTracker,
            settings.tracker_config,
            runtime.device,
        )
        tracker_probe.reset()
        if not runtime.ready:
            raise RuntimeError("runtime warm-up did not reach ready state")
        adaface = await asyncio.to_thread(
            AdaFaceRuntime,
            settings.adaface,
            fallback_device=runtime.device,
        )
        sessions = SessionRegistry(
            max_sessions=settings.max_sessions,
            max_entries_per_session=settings.max_whitelist_entries,
        )
        bundle = build_grpc_server(
            settings,
            runtime,
            sessions=sessions,
            adaface=adaface,
        )
        await bundle.set_serving(True)
        await bundle.server.start()
        LOGGER.info(
            "gRPC %s listening on %s:%d (backend=%s device=%s)",
            PROFILE,
            settings.host,
            bundle.bound_port,
            runtime.backend,
            runtime.device,
        )
        if adaface.ready:
            LOGGER.info("AdaFace ready: %s", adaface.health())
        else:
            LOGGER.warning(
                "AdaFace unavailable; all faces remain protected: %s", adaface.load_error
            )
        await bundle.server.wait_for_termination()
    finally:
        if bundle is not None:
            bundle.servicer.set_accepting(False)
            await bundle.health_servicer.enter_graceful_shutdown()
            await bundle.server.stop(settings.shutdown_grace_seconds)
            await bundle.servicer.close()
        if adaface is not None:
            await asyncio.to_thread(adaface.close)
        if runtime is not None:
            await asyncio.to_thread(runtime.close)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least one")
    return parsed


def port_number(value: str) -> int:
    parsed = int(value)
    if not 1 <= parsed <= 65_535:
        raise argparse.ArgumentTypeError("port must be between 1 and 65535")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="InnoLive B1-640 protected-video gRPC server")
    parser.add_argument("--host", default=os.getenv("GRPC_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port",
        type=port_number,
        default=port_number(os.getenv("GRPC_PORT", "50051")),
    )
    parser.add_argument("--engine", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--model", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--backend",
        choices=sorted(BACKENDS),
        default=os.getenv("AI_BACKEND", "auto").casefold(),
    )
    parser.add_argument("--tracker-config", type=Path, default=DEFAULT_TRACKER)
    parser.add_argument("--device", default=os.getenv("AI_DEVICE", "auto"))
    parser.add_argument(
        "--warmup-runs",
        type=positive_int,
        default=positive_int(os.getenv("AI_WARMUP_RUNS", "2")),
    )
    parser.add_argument(
        "--adaface-architecture",
        choices=ADAFACE_ARCHITECTURES,
        default=os.getenv("ADAFACE_ARCHITECTURE", DEFAULT_ADAFACE_ARCHITECTURE).casefold(),
    )
    parser.add_argument(
        "--adaface-weights",
        type=Path,
        default=Path(os.getenv("ADAFACE_WEIGHTS", str(DEFAULT_ADAFACE_WEIGHTS))),
    )
    parser.add_argument(
        "--adaface-detector",
        type=Path,
        default=Path(os.getenv("ADAFACE_DETECTOR", str(DEFAULT_FACE_DETECTOR))),
    )
    parser.add_argument("--adaface-device", default=os.getenv("ADAFACE_DEVICE", "auto"))
    parser.add_argument(
        "--adaface-threshold",
        type=float,
        default=float(os.getenv("ADAFACE_COSINE_THRESHOLD", "0.4")),
    )
    parser.add_argument(
        "--adaface-min-face-size",
        type=positive_int,
        default=positive_int(os.getenv("ADAFACE_MIN_FACE_SIZE", "40")),
    )
    parser.add_argument(
        "--adaface-queue-capacity",
        type=positive_int,
        default=positive_int(os.getenv("ADAFACE_QUEUE_CAPACITY", "8")),
    )
    parser.add_argument(
        "--adaface-warmup-runs",
        type=positive_int,
        default=positive_int(os.getenv("ADAFACE_WARMUP_RUNS", "1")),
    )
    parser.add_argument(
        "--adaface-revalidate-frames",
        type=positive_int,
        default=positive_int(os.getenv("ADAFACE_REVALIDATE_FRAMES", "30")),
    )
    parser.add_argument(
        "--adaface-missing-track-frames",
        type=positive_int,
        default=positive_int(os.getenv("ADAFACE_MISSING_TRACK_FRAMES", "3")),
    )
    parser.add_argument(
        "--adaface-max-pending-per-stream",
        type=positive_int,
        default=positive_int(os.getenv("ADAFACE_MAX_PENDING_PER_STREAM", "4")),
    )
    parser.add_argument(
        "--adaface-pending-timeout",
        type=float,
        default=float(os.getenv("ADAFACE_PENDING_TIMEOUT", "1.0")),
    )
    parser.add_argument(
        "--max-sessions",
        type=positive_int,
        default=positive_int(os.getenv("MAX_SESSIONS", "1024")),
    )
    parser.add_argument(
        "--max-whitelist-entries",
        type=positive_int,
        default=positive_int(os.getenv("MAX_WHITELIST_ENTRIES", "32")),
    )
    parser.add_argument(
        "--max-long-edge",
        type=positive_int,
        default=positive_int(os.getenv("MAX_LONG_EDGE", str(MAX_LONG_EDGE))),
        help="Long-edge ceiling for decoded frames (default supports FHD; lower to pin the profile, e.g. 640).",
    )
    parser.add_argument(
        "--inference-timeout",
        type=float,
        default=float(os.getenv("GRPC_INFERENCE_TIMEOUT", "1.5")),
    )
    parser.add_argument(
        "--shutdown-grace",
        type=float,
        default=float(os.getenv("GRPC_SHUTDOWN_GRACE", "5")),
    )
    parser.add_argument("--ssl-certfile", type=Path)
    parser.add_argument("--ssl-keyfile", type=Path)
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    logging.basicConfig(
        level=getattr(logging, arguments.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = GrpcServerSettings(
        runtime=RuntimeConfig(
            checkpoint=arguments.model,
            engine=arguments.engine,
            backend=arguments.backend,
            device=arguments.device,
            warmup_runs=arguments.warmup_runs,
        ),
        adaface=AdaFaceConfig(
            architecture=arguments.adaface_architecture,
            weights=arguments.adaface_weights,
            detector=arguments.adaface_detector,
            device=arguments.adaface_device,
            min_face_size=arguments.adaface_min_face_size,
            queue_capacity=arguments.adaface_queue_capacity,
            warmup_runs=arguments.adaface_warmup_runs,
        ),
        recognition=RecognitionConfig(
            cosine_threshold=arguments.adaface_threshold,
            revalidate_frames=arguments.adaface_revalidate_frames,
            missing_track_frames=arguments.adaface_missing_track_frames,
            max_pending_per_stream=arguments.adaface_max_pending_per_stream,
            pending_timeout_seconds=arguments.adaface_pending_timeout,
            min_face_size=arguments.adaface_min_face_size,
        ),
        tracker_config=arguments.tracker_config,
        host=arguments.host,
        port=arguments.port,
        max_sessions=arguments.max_sessions,
        max_whitelist_entries=arguments.max_whitelist_entries,
        max_long_edge=arguments.max_long_edge,
        inference_timeout_seconds=arguments.inference_timeout,
        shutdown_grace_seconds=arguments.shutdown_grace,
        ssl_certfile=arguments.ssl_certfile,
        ssl_keyfile=arguments.ssl_keyfile,
    )
    with suppress(KeyboardInterrupt):
        asyncio.run(serve(settings))


if __name__ == "__main__":
    main()
