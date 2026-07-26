"""Session whitelist state and stream-local track recognition cache."""

from __future__ import annotations

import asyncio
import math
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

MAX_SESSION_ID_BYTES = 256


class SessionLimitError(RuntimeError):
    """The in-memory session registry reached its configured limit."""


class SessionNotFoundError(LookupError):
    """The requested session does not exist."""


class SessionInUseError(RuntimeError):
    """An active video stream currently owns the session."""


class WhitelistLimitError(RuntimeError):
    """A session reached its configured whitelist entry limit."""


def validate_session_id(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("session_id must not be empty or whitespace-only")
    if len(value.encode("utf-8")) > MAX_SESSION_ID_BYTES:
        raise ValueError(f"session_id exceeds {MAX_SESSION_ID_BYTES} UTF-8 bytes")
    return value


@dataclass(frozen=True, slots=True)
class WhitelistEntry:
    entry_id: str
    embedding: np.ndarray


@dataclass(frozen=True, slots=True)
class WhitelistSnapshot:
    entries: tuple[WhitelistEntry, ...]
    version: int


@dataclass(frozen=True, slots=True)
class SessionSummary:
    session_id: str
    entry_count: int
    whitelist_version: int
    created_at_unix_ms: int
    active_stream_count: int


class SessionState:
    def __init__(self, session_id: str, max_entries: int) -> None:
        self.session_id = session_id
        self.max_entries = max_entries
        self.created_at_unix_ms = time.time_ns() // 1_000_000
        self._entries: tuple[WhitelistEntry, ...] = ()
        self._version = 0
        self._lock = threading.Lock()

    def snapshot(self) -> WhitelistSnapshot:
        with self._lock:
            return WhitelistSnapshot(self._entries, self._version)

    def append(self, embedding: np.ndarray) -> tuple[WhitelistEntry, int, int]:
        normalized = _normalized_embedding(embedding)
        entry = WhitelistEntry(uuid.uuid4().hex, normalized)
        with self._lock:
            if len(self._entries) >= self.max_entries:
                raise WhitelistLimitError(
                    f"session whitelist is limited to {self.max_entries} entries"
                )
            self._entries = (*self._entries, entry)
            self._version += 1
            return entry, len(self._entries), self._version

    def summary(self, *, active_stream_count: int = 0) -> SessionSummary:
        with self._lock:
            return SessionSummary(
                session_id=self.session_id,
                entry_count=len(self._entries),
                whitelist_version=self._version,
                created_at_unix_ms=self.created_at_unix_ms,
                active_stream_count=active_stream_count,
            )


class SessionRegistry:
    def __init__(self, *, max_sessions: int = 1_024, max_entries_per_session: int = 32):
        if max_sessions < 1:
            raise ValueError("max_sessions must be at least one")
        if max_entries_per_session < 1:
            raise ValueError("max_entries_per_session must be at least one")
        self.max_sessions = max_sessions
        self.max_entries_per_session = max_entries_per_session
        self._sessions: dict[str, SessionState] = {}
        self._active_stream_counts: dict[str, int] = {}
        self._lock = threading.Lock()

    def get_or_create(self, session_id: str) -> SessionState:
        validated = validate_session_id(session_id)
        with self._lock:
            return self._get_or_create_locked(validated)

    def create(self) -> SessionSummary:
        with self._lock:
            self._ensure_capacity_locked()
            while True:
                session_id = f"session-{uuid.uuid4().hex}"
                if session_id not in self._sessions:
                    break
            session = SessionState(session_id, self.max_entries_per_session)
            self._sessions[session_id] = session
            self._active_stream_counts[session_id] = 0
            return SessionSummary(
                session_id=session_id,
                entry_count=0,
                whitelist_version=0,
                created_at_unix_ms=session.created_at_unix_ms,
                active_stream_count=0,
            )

    def list_summaries(self) -> tuple[SessionSummary, ...]:
        with self._lock:
            summaries = tuple(
                session.summary(active_stream_count=self._active_stream_counts[session_id])
                for session_id, session in self._sessions.items()
            )
        return tuple(
            sorted(
                summaries,
                key=lambda item: (item.created_at_unix_ms, item.session_id),
                reverse=True,
            )
        )

    def snapshot(self, session_id: str) -> WhitelistSnapshot:
        validated = validate_session_id(session_id)
        with self._lock:
            session = self._sessions.get(validated)
            if session is None:
                return WhitelistSnapshot((), 0)
            return session.snapshot()

    def append(
        self,
        session_id: str,
        embedding: np.ndarray,
    ) -> tuple[WhitelistEntry, int, int]:
        validated = validate_session_id(session_id)
        with self._lock:
            session = self._get_or_create_locked(validated)
            return session.append(embedding)

    def acquire_stream(self, session_id: str) -> SessionState:
        validated = validate_session_id(session_id)
        with self._lock:
            session = self._get_or_create_locked(validated)
            self._active_stream_counts[validated] += 1
            return session

    def release_stream(self, session: SessionState) -> None:
        with self._lock:
            current = self._sessions.get(session.session_id)
            if current is not session:
                raise RuntimeError("cannot release a stale session stream lease")
            active_stream_count = self._active_stream_counts[session.session_id]
            if active_stream_count < 1:
                raise RuntimeError("session stream lease count is already zero")
            self._active_stream_counts[session.session_id] = active_stream_count - 1

    def delete(self, session_id: str) -> SessionSummary:
        validated = validate_session_id(session_id)
        with self._lock:
            session = self._sessions.get(validated)
            if session is None:
                raise SessionNotFoundError(f"session {validated!r} does not exist")
            active_stream_count = self._active_stream_counts[validated]
            if active_stream_count:
                raise SessionInUseError(f"session has {active_stream_count} active video stream(s)")
            summary = session.summary(active_stream_count=0)
            del self._sessions[validated]
            del self._active_stream_counts[validated]
            return summary

    def _get_or_create_locked(self, session_id: str) -> SessionState:
        session = self._sessions.get(session_id)
        if session is not None:
            return session
        self._ensure_capacity_locked()
        session = SessionState(session_id, self.max_entries_per_session)
        self._sessions[session_id] = session
        self._active_stream_counts[session_id] = 0
        return session

    def _ensure_capacity_locked(self) -> None:
        if len(self._sessions) >= self.max_sessions:
            raise SessionLimitError(f"session registry is limited to {self.max_sessions}")


@dataclass(frozen=True, slots=True)
class RecognitionConfig:
    cosine_threshold: float = 0.4
    revalidate_frames: int = 30
    quick_retry_frames: int = 5
    quick_retry_limit: int = 2
    missing_track_frames: int = 3
    max_pending_per_stream: int = 4
    min_face_size: int = 40
    query_min_face_size: int = 24
    crop_margin: float = 0.25

    def __post_init__(self) -> None:
        if not math.isfinite(self.cosine_threshold):
            raise ValueError("recognition cosine_threshold must be finite")
        if not -1.0 <= self.cosine_threshold <= 1.0:
            raise ValueError("recognition cosine_threshold must be in -1..1")
        if self.revalidate_frames < 1:
            raise ValueError("revalidate_frames must be at least one")
        if self.quick_retry_frames < 1:
            raise ValueError("quick_retry_frames must be at least one")
        if self.quick_retry_limit < 0:
            raise ValueError("quick_retry_limit must not be negative")
        if self.missing_track_frames < 1:
            raise ValueError("missing_track_frames must be at least one")
        if self.max_pending_per_stream < 1:
            raise ValueError("max_pending_per_stream must be at least one")
        if self.min_face_size < 16:
            raise ValueError("min_face_size must be at least 16")
        if self.query_min_face_size < 16:
            raise ValueError("query_min_face_size must be at least 16")
        if not math.isfinite(self.crop_margin) or not 0 <= self.crop_margin <= 1:
            raise ValueError("crop_margin must be in 0..1")


@dataclass(frozen=True, slots=True)
class RecognitionToken:
    track_id: int
    generation: int
    frame_sequence: int
    whitelist_version: int


@dataclass(slots=True)
class TrackDecision:
    generation: int
    status: str = "unknown"
    last_seen_frame: int = 0
    last_checked_frame: int = -1
    whitelist_version: int = 0
    pending_token: RecognitionToken | None = None
    status_before_pending: str = "unknown"
    quick_retry_count: int = 0


@dataclass(slots=True)
class PendingRecognition:
    future: asyncio.Future[np.ndarray]
    entries: tuple[WhitelistEntry, ...]


class StreamRecognition:
    """Track decisions owned by exactly one ProcessVideo RPC."""

    def __init__(self, runtime: Any, config: RecognitionConfig, *, owner: str):
        self.runtime = runtime
        self.config = config
        self.owner = owner
        self._states: dict[int, TrackDecision] = {}
        self._generations: dict[int, int] = {}
        self._pending: dict[RecognitionToken, PendingRecognition] = {}
        self.stale_results = 0

    def process(
        self,
        image: np.ndarray,
        objects: list[dict[str, Any]],
        snapshot: WhitelistSnapshot,
        frame_sequence: int,
    ) -> dict[str, int]:
        self._apply_completed(snapshot.version)
        current_ids = {
            int(item["track_id"]) for item in objects if item.get("track_id") is not None
        }
        self._retire_missing(current_ids, frame_sequence)

        for track_id in current_ids:
            state = self._states.get(track_id)
            if state is None:
                generation = self._generations.get(track_id, 0)
                state = TrackDecision(generation=generation)
                self._states[track_id] = state
            state.last_seen_frame = frame_sequence

        scheduled = 0
        overflow = 0
        if snapshot.entries and getattr(self.runtime, "ready", False):
            for item in objects:
                outcome = self._schedule(item, image, snapshot, frame_sequence)
                scheduled += outcome == "scheduled"
                overflow += outcome == "overflow"

        whitelisted = 0
        for item in objects:
            track_id = item.get("track_id")
            state = self._states.get(int(track_id)) if track_id is not None else None
            allowed = state is not None and state.status == "whitelisted"
            item["whitelisted"] = allowed
            whitelisted += allowed

        return {
            "adaface_calls": scheduled,
            "adaface_queue_overflow": overflow,
            "whitelisted_tracks": whitelisted,
        }

    def _schedule(
        self,
        item: dict[str, Any],
        image: np.ndarray,
        snapshot: WhitelistSnapshot,
        frame_sequence: int,
    ) -> str:
        track_id_value = item.get("track_id")
        if track_id_value is None or item.get("held", False):
            return "skipped"
        track_id = int(track_id_value)
        state = self._states[track_id]
        if not self._eligible(state, snapshot.version, frame_sequence):
            return "skipped"
        if len(self._pending) >= self.config.max_pending_per_stream:
            return "overflow"
        crop = _face_crop(image, item.get("bbox"), self.config)
        if crop is None:
            return "skipped"
        future = self.runtime.submit(crop, owner=self.owner)
        if future is None:
            return "overflow"

        if state.whitelist_version != snapshot.version:
            state.quick_retry_count = 0
        elif (
            state.last_checked_frame >= 0
            and state.status in {"unknown", "non_whitelisted"}
            and state.quick_retry_count < self.config.quick_retry_limit
        ):
            state.quick_retry_count += 1

        token = RecognitionToken(
            track_id=track_id,
            generation=state.generation,
            frame_sequence=frame_sequence,
            whitelist_version=snapshot.version,
        )
        state.status_before_pending = state.status
        state.status = "pending"
        state.pending_token = token
        self._pending[token] = PendingRecognition(future, snapshot.entries)
        return "scheduled"

    def _eligible(
        self,
        state: TrackDecision,
        whitelist_version: int,
        frame_sequence: int,
    ) -> bool:
        if state.pending_token is not None:
            return False
        if state.last_checked_frame < 0:
            return True
        if (
            state.status in {"unknown", "non_whitelisted"}
            and state.whitelist_version != whitelist_version
        ):
            return True
        interval = self.config.revalidate_frames
        if (
            state.status in {"unknown", "non_whitelisted"}
            and state.quick_retry_count < self.config.quick_retry_limit
        ):
            interval = self.config.quick_retry_frames
        return frame_sequence - state.last_checked_frame >= interval

    def _apply_completed(self, current_whitelist_version: int) -> None:
        for token, pending in tuple(self._pending.items()):
            if not pending.future.done():
                continue
            del self._pending[token]
            state = self._states.get(token.track_id)
            if not self._token_is_current(state, token, current_whitelist_version):
                if (
                    state is not None
                    and state.generation == token.generation
                    and state.pending_token == token
                ):
                    state.pending_token = None
                    state.status = (
                        "whitelisted" if state.status_before_pending == "whitelisted" else "unknown"
                    )
                self.stale_results += 1
                continue
            state.pending_token = None
            state.last_checked_frame = token.frame_sequence
            state.whitelist_version = token.whitelist_version
            try:
                embedding = _normalized_embedding(pending.future.result())
                similarity = max(
                    float(np.dot(embedding, entry.embedding)) for entry in pending.entries
                )
                if similarity >= self.config.cosine_threshold:
                    state.status = "whitelisted"
                    state.quick_retry_count = 0
                else:
                    state.status = "non_whitelisted"
            except Exception:
                state.status = "unknown"

    @staticmethod
    def _token_is_current(
        state: TrackDecision | None,
        token: RecognitionToken,
        current_whitelist_version: int,
    ) -> bool:
        return bool(
            state is not None
            and state.generation == token.generation
            and state.pending_token == token
            and token.whitelist_version == current_whitelist_version
        )

    def _retire_missing(self, current_ids: set[int], frame_sequence: int) -> None:
        for track_id, state in tuple(self._states.items()):
            if track_id in current_ids:
                continue
            if frame_sequence - state.last_seen_frame <= self.config.missing_track_frames:
                continue
            del self._states[track_id]
            self._generations[track_id] = state.generation + 1

    def close(self) -> None:
        for pending in self._pending.values():
            pending.future.cancel()
        self._pending.clear()
        self._states.clear()


def _normalized_embedding(value: np.ndarray) -> np.ndarray:
    embedding = np.asarray(value, dtype=np.float32).reshape(-1).copy()
    if embedding.size == 0 or not np.isfinite(embedding).all():
        raise ValueError("embedding must contain finite values")
    norm = float(np.linalg.norm(embedding))
    if not math.isfinite(norm) or norm <= 0:
        raise ValueError("embedding norm must be positive")
    embedding /= norm
    embedding.setflags(write=False)
    return embedding


def _face_crop(
    image: np.ndarray,
    bbox: Any,
    config: RecognitionConfig,
) -> np.ndarray | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    width = x2 - x1
    height = y2 - y1
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    query_min_face_size = min(config.min_face_size, config.query_min_face_size)
    if min(width, height) < query_min_face_size:
        return None
    margin_x = width * config.crop_margin
    margin_y = height * config.crop_margin
    left = max(0, math.floor(x1 - margin_x))
    top = max(0, math.floor(y1 - margin_y))
    right = min(image.shape[1], math.ceil(x2 + margin_x))
    bottom = min(image.shape[0], math.ceil(y2 + margin_y))
    if right <= left or bottom <= top:
        return None
    return image[top:bottom, left:right].copy()
