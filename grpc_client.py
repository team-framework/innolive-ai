"""Small fail-closed client for the InnoLive ProcessVideo bidi stream."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import AsyncIterable, AsyncIterator, Iterable
from contextlib import aclosing, suppress
from dataclasses import dataclass
from typing import Any, Self

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

from protos import ai_processor_pb2, ai_processor_pb2_grpc
from service.grpc_config import channel_options
from service.protocol import MAX_JPEG_BYTES, MAX_RESPONSE_BYTES

MAX_WINDOW = 5
MAX_FRAME_ID = 2**32 - 1


class VideoClientError(RuntimeError):
    """Base exception for transport and ProcessVideo contract failures."""


class VideoProtocolError(VideoClientError):
    """The peer violated the ProcessVideo wire contract."""


class VideoFrameError(VideoClientError):
    """The server returned a terminal processing error for one frame."""

    def __init__(self, frame_id: int, detail: str):
        super().__init__(f"frame {frame_id} failed: {detail}")
        self.frame_id = frame_id
        self.detail = detail


@dataclass(frozen=True, slots=True)
class VideoFrame:
    """One complete JPEG request; timestamp is an opaque signed int64 value."""

    data: bytes
    timestamp: int
    frame_id: int


@dataclass(frozen=True, slots=True)
class VideoResult:
    """Metadata response paired with the exact JPEG sent for that frame."""

    source_jpeg: bytes
    response: ai_processor_pb2.ProcessedVideoChunk


@dataclass(slots=True)
class _StreamState:
    pending: deque[VideoFrame]
    slots: asyncio.Semaphore
    source_exhausted: asyncio.Event
    writer_observed: bool = False
    completed: bool = False


FrameSource = Iterable[VideoFrame] | AsyncIterable[VideoFrame]
JpegSource = Iterable[bytes] | AsyncIterable[bytes]


class VideoProcessorClient:
    """Async context manager exposing one bounded ProcessVideo operation."""

    def __init__(
        self,
        target: str,
        *,
        credentials: grpc.ChannelCredentials | None = None,
        connect_timeout: float = 10.0,
    ) -> None:
        if not isinstance(target, str) or not target.strip():
            raise ValueError("target must be a non-empty host:port string")
        if connect_timeout <= 0:
            raise ValueError("connect_timeout must be positive")
        self.target = target.strip()
        self.credentials = credentials
        self.connect_timeout = float(connect_timeout)
        self._channel: grpc.aio.Channel | None = None
        self._stub: ai_processor_pb2_grpc.AiProcessorStub | None = None
        self._health_stub: health_pb2_grpc.HealthStub | None = None
        self._calls: set[Any] = set()
        self._max_inflight_observed = 0

    @property
    def max_inflight_observed(self) -> int:
        """Largest number of source JPEGs retained by the latest stream."""

        return self._max_inflight_observed

    async def __aenter__(self) -> Self:
        if self._channel is not None:
            raise RuntimeError("VideoProcessorClient is already open")
        if self.credentials is None:
            channel = grpc.aio.insecure_channel(
                self.target,
                options=channel_options(),
            )
        else:
            channel = grpc.aio.secure_channel(
                self.target,
                self.credentials,
                options=channel_options(),
            )
        self._channel = channel
        self._stub = ai_processor_pb2_grpc.AiProcessorStub(channel)
        self._health_stub = health_pb2_grpc.HealthStub(channel)
        try:
            await asyncio.wait_for(
                channel.channel_ready(),
                timeout=self.connect_timeout,
            )
        except BaseException:
            await channel.close()
            self._channel = None
            self._stub = None
            self._health_stub = None
            raise
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        calls = tuple(self._calls)
        for call in calls:
            call.cancel()
        channel = self._channel
        self._channel = None
        self._stub = None
        self._health_stub = None
        self._calls.clear()
        if channel is not None:
            await channel.close()

    async def is_serving(self, service: str = "AiProcessor", *, timeout: float = 2.0) -> bool:
        """Return whether the connected standard gRPC health service is serving."""

        if self._health_stub is None:
            raise RuntimeError("use VideoProcessorClient with 'async with'")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        response = await self._health_stub.Check(
            health_pb2.HealthCheckRequest(service=service),
            timeout=timeout,
        )
        return response.status == health_pb2.HealthCheckResponse.SERVING

    async def process_video(
        self,
        frames: FrameSource,
        *,
        window: int = MAX_WINDOW,
        timeout: float | None = None,
        raise_frame_errors: bool = True,
    ) -> AsyncIterator[VideoResult]:
        """Send frames on one persistent bidi call and yield ordered metadata.

        A writer coroutine fills at most ``window`` slots while this generator
        serially reads responses. Each response must match the oldest retained
        JPEG. Any transport, protocol, cancellation, or early-EOF failure cancels
        the RPC and drops every unmatched source frame. Processing failures raise
        by default.
        Set ``raise_frame_errors=False`` only for adapters that must forward the
        server's typed per-frame error response to another client.
        """

        if type(window) is not int or not 1 <= window <= MAX_WINDOW:
            raise ValueError(f"window must be an integer in 1..{MAX_WINDOW}")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be positive when supplied")
        if type(raise_frame_errors) is not bool:
            raise TypeError("raise_frame_errors must be a boolean")
        if self._stub is None:
            raise RuntimeError("use VideoProcessorClient with 'async with'")
        if self._calls:
            raise RuntimeError("one VideoProcessorClient supports one active stream")

        call = self._stub.ProcessVideo(timeout=timeout)
        self._calls.add(call)
        self._max_inflight_observed = 0
        state = _StreamState(deque(), asyncio.Semaphore(window), asyncio.Event())
        writer = asyncio.create_task(
            self._write_frames(
                call,
                frames,
                state.pending,
                state.slots,
                state.source_exhausted,
            ),
            name="innolive-grpc-writer",
        )

        try:
            async for result in self._read_results(
                call,
                writer,
                state,
                raise_frame_errors=raise_frame_errors,
            ):
                yield result
        except grpc.aio.AioRpcError as error:
            raise VideoClientError(
                f"ProcessVideo RPC failed with {error.code().name}: {error.details()}"
            ) from error
        finally:
            if not state.completed:
                call.cancel()
            if not writer.done():
                writer.cancel()
            await asyncio.gather(writer, return_exceptions=True)
            state.pending.clear()
            self._calls.discard(call)

    async def _read_results(
        self,
        call: Any,
        writer: asyncio.Task,
        state: _StreamState,
        *,
        raise_frame_errors: bool,
    ) -> AsyncIterator[VideoResult]:
        read_task: asyncio.Task[Any] = asyncio.create_task(call.read())
        try:
            while True:
                watched = {read_task}
                if not state.writer_observed:
                    watched.add(writer)
                done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
                if writer in done:
                    state.writer_observed = True
                    writer_error = writer.exception()
                    if writer_error is not None:
                        raise writer_error
                if read_task not in done:
                    continue

                response = read_task.result()
                if response is grpc.aio.EOF:
                    await self._validate_end_of_stream(writer, state)
                    state.completed = True
                    return
                if not state.pending:
                    raise VideoProtocolError("ProcessVideo returned an unknown or duplicate frame")

                source = state.pending.popleft()
                self._validate_response(
                    source,
                    response,
                    raise_frame_errors=raise_frame_errors,
                )
                state.slots.release()
                yield VideoResult(source_jpeg=source.data, response=response)
                read_task = asyncio.create_task(call.read())
        finally:
            if not read_task.done():
                read_task.cancel()
                with suppress(asyncio.CancelledError):
                    await read_task
            else:
                with suppress(asyncio.CancelledError, Exception):
                    read_task.result()

    @staticmethod
    async def _validate_end_of_stream(writer: asyncio.Task, state: _StreamState) -> None:
        if not state.source_exhausted.is_set():
            raise VideoProtocolError("ProcessVideo ended before the request source completed")
        if not state.writer_observed:
            await writer
            state.writer_observed = True
        if state.pending:
            raise VideoProtocolError(
                f"ProcessVideo ended with {len(state.pending)} frame(s) unresolved"
            )

    async def process_jpegs(
        self,
        jpegs: JpegSource,
        *,
        start_frame_id: int = 1,
        window: int = MAX_WINDOW,
        timeout: float | None = None,
        raise_frame_errors: bool = True,
    ) -> AsyncIterator[VideoResult]:
        """Convenience API that assigns monotonic IDs and timestamps to JPEGs."""

        if type(start_frame_id) is not int or not 0 <= start_frame_id <= MAX_FRAME_ID:
            raise ValueError(f"start_frame_id must be in 0..{MAX_FRAME_ID}")

        async def frames() -> AsyncIterator[VideoFrame]:
            frame_id = start_frame_id
            async for jpeg in _iterate_jpegs(jpegs):
                if frame_id > MAX_FRAME_ID:
                    raise ValueError("frame_id wrapped; open a new ProcessVideo stream")
                yield VideoFrame(
                    data=jpeg,
                    timestamp=time.perf_counter_ns(),
                    frame_id=frame_id,
                )
                frame_id += 1

        async with aclosing(
            self.process_video(
                frames(),
                window=window,
                timeout=timeout,
                raise_frame_errors=raise_frame_errors,
            )
        ) as results:
            async for result in results:
                yield result

    async def _write_frames(
        self,
        call: Any,
        frames: FrameSource,
        pending: deque[VideoFrame],
        slots: asyncio.Semaphore,
        source_exhausted: asyncio.Event,
    ) -> None:
        last_frame_id = -1
        iterator = _iterate_frames(frames).__aiter__()
        try:
            while True:
                await slots.acquire()
                try:
                    frame = await anext(iterator)
                except StopAsyncIteration:
                    slots.release()
                    source_exhausted.set()
                    break
                except BaseException:
                    slots.release()
                    raise

                try:
                    normalized = _validate_frame(frame, last_frame_id)
                except BaseException:
                    slots.release()
                    raise
                last_frame_id = normalized.frame_id
                pending.append(normalized)
                self._max_inflight_observed = max(
                    self._max_inflight_observed,
                    len(pending),
                )
                await call.write(
                    ai_processor_pb2.VideoChunk(
                        data=normalized.data,
                        timestamp=normalized.timestamp,
                        frame_id=normalized.frame_id,
                        batch_size=1,
                    )
                )
            await call.done_writing()
        finally:
            with suppress(Exception):
                await iterator.aclose()

    @staticmethod
    def _validate_response(
        source: VideoFrame,
        response: Any,
        *,
        raise_frame_errors: bool,
    ) -> None:
        if int(response.ByteSize()) > MAX_RESPONSE_BYTES:
            raise VideoProtocolError(
                f"frame {source.frame_id} response exceeded the metadata limit"
            )
        if bytes(response.data):
            raise VideoProtocolError(
                f"frame {source.frame_id} response echoed forbidden pixel data"
            )
        response_frame_id = int(response.frame_id)
        if response_frame_id != source.frame_id:
            raise VideoProtocolError(
                "ProcessVideo response order mismatch: "
                f"expected {source.frame_id}, received {response_frame_id}"
            )
        if int(response.timestamp) != source.timestamp:
            raise VideoProtocolError(
                f"frame {source.frame_id} response changed the opaque timestamp"
            )

        status, detail = _response_status(response)
        if status == "error":
            if raise_frame_errors:
                raise VideoFrameError(source.frame_id, detail)
            return
        if response.error_code or response.error_message:
            raise VideoProtocolError(f"frame {source.frame_id} success response contained an error")


async def _iterate_frames(frames: FrameSource) -> AsyncIterator[VideoFrame]:
    if isinstance(frames, AsyncIterable):
        iterator = frames.__aiter__()
        try:
            async for frame in iterator:
                yield frame
        finally:
            close = getattr(iterator, "aclose", None)
            if close is not None:
                with suppress(Exception):
                    await close()
        return
    if not isinstance(frames, Iterable):
        raise TypeError("frames must be an iterable or async iterable")
    iterator = iter(frames)
    try:
        for frame in iterator:
            yield frame
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            with suppress(Exception):
                close()


async def _iterate_jpegs(jpegs: JpegSource) -> AsyncIterator[bytes]:
    if isinstance(jpegs, AsyncIterable):
        async for jpeg in jpegs:
            yield jpeg
        return
    if not isinstance(jpegs, Iterable):
        raise TypeError("jpegs must be an iterable or async iterable")
    for jpeg in jpegs:
        yield jpeg


def _validate_frame(frame: VideoFrame, last_frame_id: int) -> VideoFrame:
    if not isinstance(frame, VideoFrame):
        raise TypeError("frames must yield VideoFrame instances")
    if not isinstance(frame.data, (bytes, bytearray, memoryview)):
        raise TypeError("VideoFrame.data must be bytes-like")
    jpeg = frame.data if isinstance(frame.data, bytes) else bytes(frame.data)
    if not jpeg:
        raise ValueError("VideoFrame.data must not be empty")
    if len(jpeg) > MAX_JPEG_BYTES:
        raise ValueError(f"JPEG exceeds the {MAX_JPEG_BYTES} byte limit")
    if not jpeg.startswith(b"\xff\xd8") or not jpeg.endswith(b"\xff\xd9"):
        raise ValueError("VideoFrame.data must contain one complete JPEG")
    if type(frame.timestamp) is not int or not -(2**63) <= frame.timestamp < 2**63:
        raise ValueError("VideoFrame.timestamp must be a signed int64")
    if type(frame.frame_id) is not int or not 0 <= frame.frame_id <= MAX_FRAME_ID:
        raise ValueError(f"VideoFrame.frame_id must be in 0..{MAX_FRAME_ID}")
    if frame.frame_id <= last_frame_id:
        raise ValueError("VideoFrame.frame_id must be strictly increasing")
    return VideoFrame(jpeg, frame.timestamp, frame.frame_id)


def _response_status(response: Any) -> tuple[str, str]:
    status_message = str(getattr(response, "status_message", "")).strip()
    normalized = status_message.casefold()
    if normalized == "success":
        return "success", status_message
    if normalized == "error" or normalized.startswith(("error:", "failed")):
        return "error", _error_detail(response, status_message)
    raise VideoProtocolError(f"unknown ProcessVideo status: {status_message!r}")


def _error_detail(response: Any, fallback: str) -> str:
    code = str(getattr(response, "error_code", "")).strip()
    message = str(getattr(response, "error_message", "")).strip()
    if code and message:
        return f"{code}: {message}"
    return code or message or fallback
