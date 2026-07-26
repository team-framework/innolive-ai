from __future__ import annotations

import asyncio
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from service.recognition import (
    RecognitionConfig,
    SessionInUseError,
    SessionLimitError,
    SessionNotFoundError,
    SessionRegistry,
    StreamRecognition,
    WhitelistEntryNotFoundError,
    WhitelistLimitError,
    validate_entry_id,
    validate_session_id,
)


def _object(track_id: int) -> dict[str, Any]:
    return {
        "track_id": track_id,
        "bbox": [10.0, 10.0, 70.0, 70.0],
        "mask_polygon": [[10.0, 10.0], [70.0, 10.0], [70.0, 70.0]],
        "confidence": 0.9,
    }


class FakeRecognitionRuntime:
    ready = True

    def __init__(self, embeddings: list[np.ndarray] | None = None, *, deferred: bool = False):
        self.embeddings = list(embeddings or [np.asarray([1.0, 0.0], dtype=np.float32)])
        self.deferred = deferred
        self.overflow = False
        self.calls = 0
        self.futures: list[asyncio.Future[np.ndarray]] = []

    def submit(self, image: np.ndarray, *, owner: str):
        del image
        del owner
        if self.overflow:
            return None
        self.calls += 1
        future = asyncio.get_running_loop().create_future()
        self.futures.append(future)
        if not self.deferred:
            index = min(self.calls - 1, len(self.embeddings) - 1)
            future.set_result(self.embeddings[index].copy())
        return future


class PixelRecognitionRuntime(FakeRecognitionRuntime):
    def submit(self, image: np.ndarray, *, owner: str):
        del owner
        self.calls += 1
        future = asyncio.get_running_loop().create_future()
        embedding = [1.0, 0.0] if float(image.mean()) < 100 else [0.0, 1.0]
        future.set_result(np.asarray(embedding, dtype=np.float32))
        self.futures.append(future)
        return future


class StreamRecognitionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.image = np.zeros((96, 96, 3), dtype=np.uint8)
        self.config = RecognitionConfig(revalidate_frames=50, missing_track_frames=1)

    async def test_session_whitelists_are_isolated(self):
        sessions = SessionRegistry()
        sessions.get_or_create("session-a").append(np.asarray([1.0, 0.0]))
        sessions.get_or_create("session-b").append(np.asarray([0.0, 1.0]))

        first = sessions.get_or_create("session-a").snapshot()
        second = sessions.get_or_create("session-b").snapshot()

        self.assertEqual(len(first.entries), 1)
        self.assertEqual(len(second.entries), 1)
        self.assertTrue(np.array_equal(first.entries[0].embedding, [1.0, 0.0]))
        self.assertTrue(np.array_equal(second.entries[0].embedding, [0.0, 1.0]))

    async def test_session_id_validation_does_not_normalize_the_value(self):
        self.assertEqual(validate_session_id(" Session-A "), " Session-A ")
        with self.assertRaises(ValueError):
            validate_session_id("   ")

        self.assertEqual(validate_entry_id(" Entry-A "), " Entry-A ")
        with self.assertRaises(ValueError):
            validate_entry_id("   ")

    async def test_session_and_whitelist_limits_are_enforced(self):
        sessions = SessionRegistry(max_sessions=1, max_entries_per_session=1)
        session = sessions.get_or_create("session-a")
        session.append(np.asarray([1.0, 0.0]))

        with self.assertRaises(WhitelistLimitError):
            session.append(np.asarray([0.0, 1.0]))
        with self.assertRaises(SessionLimitError):
            sessions.get_or_create("session-b")

    async def test_status_lookup_does_not_create_a_session(self):
        sessions = SessionRegistry(max_sessions=1)

        missing = sessions.snapshot("missing-session")
        created = sessions.get_or_create("session-a")

        self.assertEqual((len(missing.entries), missing.version), (0, 0))
        self.assertEqual(created.snapshot().version, 0)

    async def test_generated_sessions_are_unique_and_listed_with_manual_sessions(self):
        sessions = SessionRegistry(max_sessions=40)
        manual = sessions.get_or_create("manual-session")
        manual.append(np.asarray([1.0, 0.0]))

        with ThreadPoolExecutor(max_workers=8) as executor:
            generated = list(executor.map(lambda _: sessions.create(), range(32)))

        generated_ids = {summary.session_id for summary in generated}
        self.assertEqual(len(generated_ids), 32)
        self.assertTrue(all(value.startswith("session-") for value in generated_ids))
        listed = {summary.session_id: summary for summary in sessions.list_summaries()}
        self.assertEqual(set(listed), {"manual-session", *generated_ids})
        self.assertEqual(
            (listed["manual-session"].entry_count, listed["manual-session"].whitelist_version),
            (1, 1),
        )

    async def test_generated_session_obeys_the_registry_limit(self):
        sessions = SessionRegistry(max_sessions=1)
        sessions.create()

        with self.assertRaises(SessionLimitError):
            sessions.create()

    async def test_whitelist_entry_deletion_is_atomic_and_allowed_during_a_stream(self):
        sessions = SessionRegistry()
        session = sessions.get_or_create("session")
        first, _, _ = session.append(np.asarray([1.0, 0.0]))
        second, _, _ = session.append(np.asarray([0.0, 1.0]))
        lease = sessions.acquire_stream("session")

        deleted_id, entry_count, version = sessions.delete_whitelist_entry(
            "session",
            first.entry_id,
        )
        snapshot = session.snapshot()

        self.assertEqual((deleted_id, entry_count, version), (first.entry_id, 1, 3))
        self.assertEqual([entry.entry_id for entry in snapshot.entries], [second.entry_id])
        self.assertEqual(snapshot.version, 3)
        self.assertEqual(sessions.list_summaries()[0].active_stream_count, 1)
        with self.assertRaises(WhitelistEntryNotFoundError):
            sessions.delete_whitelist_entry("session", first.entry_id)
        with self.assertRaises(SessionNotFoundError):
            sessions.delete_whitelist_entry("missing", second.entry_id)

        sessions.release_stream(lease)

    async def test_concurrent_whitelist_mutations_are_linearizable(self):
        sessions = SessionRegistry()
        first, _, _ = sessions.append("session", np.asarray([1.0, 0.0]))

        def delete_once():
            try:
                return sessions.delete_whitelist_entry("session", first.entry_id)
            except WhitelistEntryNotFoundError:
                return None

        with ThreadPoolExecutor(max_workers=2) as executor:
            deleted = list(executor.map(lambda _: delete_once(), range(2)))

        self.assertEqual(sum(result is not None for result in deleted), 1)
        self.assertEqual(sessions.snapshot("session").version, 2)

        second, _, _ = sessions.append("session", np.asarray([1.0, 0.0]))
        barrier = threading.Barrier(2)

        def append_entry():
            barrier.wait()
            return sessions.append("session", np.asarray([0.0, 1.0]))

        def delete_entry():
            barrier.wait()
            return sessions.delete_whitelist_entry("session", second.entry_id)

        with ThreadPoolExecutor(max_workers=2) as executor:
            added_future = executor.submit(append_entry)
            deleted_future = executor.submit(delete_entry)
            added_entry, _, _ = added_future.result()
            deleted_future.result()

        snapshot = sessions.snapshot("session")
        self.assertEqual([entry.entry_id for entry in snapshot.entries], [added_entry.entry_id])
        self.assertEqual(snapshot.version, 5)

    async def test_active_stream_lease_blocks_deletion_and_is_listed(self):
        sessions = SessionRegistry()
        session = sessions.get_or_create("active-session")
        first = sessions.acquire_stream("active-session")
        second = sessions.acquire_stream("active-session")

        self.assertIs(first, session)
        self.assertIs(second, session)
        self.assertEqual(sessions.list_summaries()[0].active_stream_count, 2)
        with self.assertRaises(SessionInUseError):
            sessions.delete("active-session")

        sessions.release_stream(first)
        sessions.release_stream(second)
        deleted = sessions.delete("active-session")

        self.assertEqual(deleted.session_id, "active-session")
        self.assertEqual(sessions.list_summaries(), ())
        with self.assertRaises(SessionNotFoundError):
            sessions.delete("active-session")

    async def test_stream_acquire_and_delete_are_atomic(self):
        for _ in range(50):
            sessions = SessionRegistry()
            original = sessions.get_or_create("raced-session")
            barrier = threading.Barrier(2)

            def acquire(registry=sessions, start=barrier):
                start.wait()
                return registry.acquire_stream("raced-session")

            def delete(registry=sessions, start=barrier):
                start.wait()
                try:
                    return registry.delete("raced-session")
                except SessionInUseError:
                    return None

            with ThreadPoolExecutor(max_workers=2) as executor:
                acquired_future = executor.submit(acquire)
                deleted_future = executor.submit(delete)
                acquired = acquired_future.result()
                deleted = deleted_future.result()

            if deleted is None:
                self.assertIs(acquired, original)
            else:
                self.assertIsNot(acquired, original)
            active = sessions.list_summaries()
            self.assertEqual(len(active), 1)
            self.assertEqual(active[0].active_stream_count, 1)
            sessions.release_stream(acquired)
            sessions.delete("raced-session")

    async def test_same_session_streams_share_only_the_whitelist(self):
        session = SessionRegistry().get_or_create("shared")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime()
        first = StreamRecognition(runtime, self.config, owner="shared")
        second = StreamRecognition(runtime, self.config, owner="shared")

        first_object = _object(1)
        second_object = _object(1)
        first.process(self.image, [first_object], session.snapshot(), 1)
        second.process(self.image, [second_object], session.snapshot(), 1)
        first.process(self.image, [first_object], session.snapshot(), 2)
        second.process(self.image, [second_object], session.snapshot(), 2)

        self.assertEqual(runtime.calls, 2)
        self.assertTrue(first_object["whitelisted"])
        self.assertTrue(second_object["whitelisted"])
        self.assertIsNot(first._states, second._states)

    async def test_empty_whitelist_never_calls_adaface_and_protects_every_face(self):
        session = SessionRegistry().get_or_create("empty")
        runtime = FakeRecognitionRuntime()
        recognition = StreamRecognition(runtime, self.config, owner="session")
        face = _object(1)

        metrics = recognition.process(self.image, [face], session.snapshot(), 1)

        self.assertEqual(runtime.calls, 0)
        self.assertEqual(metrics["adaface_calls"], 0)
        self.assertFalse(face["whitelisted"])

    async def test_unavailable_adaface_protects_faces_with_a_nonempty_whitelist(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime()
        runtime.ready = False
        recognition = StreamRecognition(runtime, self.config, owner="session")
        face = _object(1)

        recognition.process(self.image, [face], session.snapshot(), 1)

        self.assertEqual(runtime.calls, 0)
        self.assertFalse(face["whitelisted"])

    async def test_same_track_is_not_recognized_on_every_frame(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime()
        recognition = StreamRecognition(runtime, self.config, owner="session")
        face = _object(1)

        for frame_sequence in range(1, 21):
            recognition.process(self.image, [face], session.snapshot(), frame_sequence)

        self.assertEqual(runtime.calls, 1)
        self.assertTrue(face["whitelisted"])

    async def test_non_match_retries_twice_before_using_the_long_interval(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime(embeddings=[np.asarray([0.0, 1.0], dtype=np.float32)] * 4)
        config = RecognitionConfig(revalidate_frames=30, missing_track_frames=1)
        recognition = StreamRecognition(runtime, config, owner="session")
        face = _object(1)

        for frame_sequence in range(1, 41):
            recognition.process(self.image, [face], session.snapshot(), frame_sequence)

        self.assertEqual(runtime.calls, 3)
        self.assertFalse(face["whitelisted"])
        recognition.process(self.image, [face], session.snapshot(), 41)
        self.assertEqual(runtime.calls, 4)

    async def test_failed_query_retries_quickly_and_success_resets_retry_count(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime(deferred=True)
        recognition = StreamRecognition(runtime, self.config, owner="session")
        face = _object(1)

        recognition.process(self.image, [face], session.snapshot(), 1)
        runtime.futures[0].set_exception(RuntimeError("alignment failed"))
        recognition.process(self.image, [face], session.snapshot(), 2)
        for frame_sequence in range(3, 6):
            recognition.process(self.image, [face], session.snapshot(), frame_sequence)

        self.assertEqual(runtime.calls, 1)
        recognition.process(self.image, [face], session.snapshot(), 6)
        self.assertEqual(runtime.calls, 2)
        runtime.futures[1].set_result(np.asarray([1.0, 0.0], dtype=np.float32))
        recognition.process(self.image, [face], session.snapshot(), 7)

        self.assertTrue(face["whitelisted"])
        self.assertEqual(recognition._states[1].quick_retry_count, 0)

    async def test_video_query_accepts_a_face_smaller_than_enrollment_minimum(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime()
        recognition = StreamRecognition(runtime, self.config, owner="session")
        face = _object(1)
        face["bbox"] = [10.0, 10.0, 34.0, 34.0]

        recognition.process(self.image, [face], session.snapshot(), 1)

        self.assertEqual(runtime.calls, 1)

    async def test_new_whitelist_entry_rechecks_an_active_non_whitelisted_track(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime(
            [
                np.asarray([0.0, 1.0], dtype=np.float32),
                np.asarray([0.0, 1.0], dtype=np.float32),
            ]
        )
        recognition = StreamRecognition(runtime, self.config, owner="session")
        face = _object(1)

        recognition.process(self.image, [face], session.snapshot(), 1)
        recognition.process(self.image, [face], session.snapshot(), 2)
        self.assertFalse(face["whitelisted"])
        session.append(np.asarray([0.0, 1.0]))
        recognition.process(self.image, [face], session.snapshot(), 3)
        recognition.process(self.image, [face], session.snapshot(), 4)

        self.assertEqual(runtime.calls, 2)
        self.assertTrue(face["whitelisted"])

    async def test_new_entry_does_not_invalidate_an_already_whitelisted_track(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime()
        recognition = StreamRecognition(runtime, self.config, owner="session")
        face = _object(1)

        recognition.process(self.image, [face], session.snapshot(), 1)
        recognition.process(self.image, [face], session.snapshot(), 2)
        session.append(np.asarray([0.0, 1.0]))
        recognition.process(self.image, [face], session.snapshot(), 3)

        self.assertTrue(face["whitelisted"])
        self.assertEqual(runtime.calls, 2)

    async def test_periodic_revalidation_keeps_a_valid_match_until_negative_result(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime(deferred=True)
        config = RecognitionConfig(revalidate_frames=3)
        recognition = StreamRecognition(runtime, config, owner="session")
        face = _object(1)

        recognition.process(self.image, [face], session.snapshot(), 1)
        runtime.futures[0].set_result(np.asarray([1.0, 0.0], dtype=np.float32))
        recognition.process(self.image, [face], session.snapshot(), 2)
        recognition.process(self.image, [face], session.snapshot(), 3)
        recognition.process(self.image, [face], session.snapshot(), 4)

        self.assertTrue(face["whitelisted"])
        self.assertIsNotNone(recognition._states[1].pending_token)
        runtime.futures[1].set_result(np.asarray([0.0, 1.0], dtype=np.float32))
        recognition.process(self.image, [face], session.snapshot(), 5)

        self.assertFalse(face["whitelisted"])
        self.assertIsNone(recognition._states[1].matched_entry_id)

    async def test_periodic_revalidation_error_immediately_removes_the_match(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime(deferred=True)
        config = RecognitionConfig(revalidate_frames=3)
        recognition = StreamRecognition(runtime, config, owner="session")
        face = _object(1)

        recognition.process(self.image, [face], session.snapshot(), 1)
        runtime.futures[0].set_result(np.asarray([1.0, 0.0], dtype=np.float32))
        recognition.process(self.image, [face], session.snapshot(), 2)
        recognition.process(self.image, [face], session.snapshot(), 3)
        recognition.process(self.image, [face], session.snapshot(), 4)
        self.assertTrue(face["whitelisted"])

        runtime.futures[1].set_exception(RuntimeError("alignment failed"))
        recognition.process(self.image, [face], session.snapshot(), 5)

        self.assertFalse(face["whitelisted"])
        self.assertIsNone(recognition._states[1].matched_entry_id)

    async def test_stale_revalidation_negative_or_error_remains_fail_closed(self):
        for outcome in ("negative", "error"):
            with self.subTest(outcome=outcome):
                session = SessionRegistry().get_or_create("session")
                session.append(np.asarray([1.0, 0.0]))
                runtime = FakeRecognitionRuntime(deferred=True)
                config = RecognitionConfig(revalidate_frames=3)
                recognition = StreamRecognition(runtime, config, owner="session")
                face = _object(1)

                recognition.process(self.image, [face], session.snapshot(), 1)
                runtime.futures[0].set_result(np.asarray([1.0, 0.0], dtype=np.float32))
                recognition.process(self.image, [face], session.snapshot(), 2)
                recognition.process(self.image, [face], session.snapshot(), 3)
                recognition.process(self.image, [face], session.snapshot(), 4)
                session.append(np.asarray([0.0, 1.0]))
                if outcome == "negative":
                    runtime.futures[1].set_result(np.asarray([0.0, 1.0], dtype=np.float32))
                else:
                    runtime.futures[1].set_exception(RuntimeError("alignment failed"))

                recognition.process(self.image, [face], session.snapshot(), 5)

                self.assertFalse(face["whitelisted"])
                self.assertIsNone(recognition._states[1].matched_entry_id)
                self.assertEqual(recognition.stale_results, 1)
                recognition.close()

    async def test_revalidation_admission_failure_immediately_protects_the_face(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime(deferred=True)
        config = RecognitionConfig(revalidate_frames=3)
        recognition = StreamRecognition(runtime, config, owner="session")
        face = _object(1)

        recognition.process(self.image, [face], session.snapshot(), 1)
        runtime.futures[0].set_result(np.asarray([1.0, 0.0], dtype=np.float32))
        recognition.process(self.image, [face], session.snapshot(), 2)
        recognition.process(self.image, [face], session.snapshot(), 3)
        runtime.overflow = True
        metrics = recognition.process(self.image, [face], session.snapshot(), 4)

        self.assertFalse(face["whitelisted"])
        self.assertEqual(metrics["adaface_queue_overflow"], 1)
        self.assertIsNone(recognition._states[1].matched_entry_id)

    async def test_revalidation_timeout_revokes_the_match_and_discards_the_result(self):
        now = 0.0
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime(deferred=True)
        config = RecognitionConfig(
            revalidate_frames=3,
            pending_timeout_seconds=0.5,
        )
        recognition = StreamRecognition(
            runtime,
            config,
            owner="session",
            clock=lambda: now,
        )
        face = _object(1)

        recognition.process(self.image, [face], session.snapshot(), 1)
        runtime.futures[0].set_result(np.asarray([1.0, 0.0], dtype=np.float32))
        recognition.process(self.image, [face], session.snapshot(), 2)
        recognition.process(self.image, [face], session.snapshot(), 3)
        recognition.process(self.image, [face], session.snapshot(), 4)
        self.assertTrue(face["whitelisted"])

        now = 0.6
        recognition.process(self.image, [face], session.snapshot(), 5)

        self.assertTrue(runtime.futures[1].cancelled())
        self.assertFalse(face["whitelisted"])
        self.assertEqual(recognition.stale_results, 1)
        recognition.close()

    async def test_runtime_unavailable_revokes_existing_matches(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime(deferred=True)
        recognition = StreamRecognition(runtime, self.config, owner="session")
        face = _object(1)

        recognition.process(self.image, [face], session.snapshot(), 1)
        runtime.futures[0].set_result(np.asarray([1.0, 0.0], dtype=np.float32))
        recognition.process(self.image, [face], session.snapshot(), 2)
        self.assertTrue(face["whitelisted"])

        runtime.ready = False
        recognition.process(self.image, [face], session.snapshot(), 3)

        self.assertFalse(face["whitelisted"])
        self.assertIsNone(recognition._states[1].matched_entry_id)

    async def test_removing_the_matched_entry_revokes_access_during_pending_work(self):
        session = SessionRegistry().get_or_create("session")
        matched, _, _ = session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime(deferred=True)
        config = RecognitionConfig(revalidate_frames=3)
        recognition = StreamRecognition(runtime, config, owner="session")
        face = _object(1)

        original = session.snapshot()
        recognition.process(self.image, [face], original, 1)
        runtime.futures[0].set_result(np.asarray([1.0, 0.0], dtype=np.float32))
        recognition.process(self.image, [face], original, 2)
        self.assertEqual(recognition._states[1].matched_entry_id, matched.entry_id)
        recognition.process(self.image, [face], original, 3)
        recognition.process(self.image, [face], original, 4)
        self.assertTrue(face["whitelisted"])

        session.delete_entry(matched.entry_id)
        empty = session.snapshot()
        recognition.process(self.image, [face], empty, 5)
        self.assertFalse(face["whitelisted"])
        self.assertIsNone(recognition._states[1].matched_entry_id)

        runtime.futures[1].set_result(np.asarray([1.0, 0.0], dtype=np.float32))
        recognition.process(self.image, [face], empty, 6)

        self.assertFalse(face["whitelisted"])
        self.assertEqual(recognition.stale_results, 1)

    async def test_removing_an_unmatched_exemplar_preserves_the_current_match(self):
        session = SessionRegistry().get_or_create("session")
        matched, _, _ = session.append(np.asarray([1.0, 0.0]))
        unmatched, _, _ = session.append(np.asarray([0.0, 1.0]))
        runtime = FakeRecognitionRuntime(deferred=True)
        recognition = StreamRecognition(runtime, self.config, owner="session")
        face = _object(1)

        original = session.snapshot()
        recognition.process(self.image, [face], original, 1)
        runtime.futures[0].set_result(np.asarray([1.0, 0.0], dtype=np.float32))
        recognition.process(self.image, [face], original, 2)
        session.delete_entry(unmatched.entry_id)
        remaining = session.snapshot()
        recognition.process(self.image, [face], remaining, 3)

        self.assertTrue(face["whitelisted"])
        self.assertEqual(recognition._states[1].matched_entry_id, matched.entry_id)
        self.assertEqual(runtime.calls, 2)
        recognition.close()

    async def test_recognition_failure_keeps_the_face_protected(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime(deferred=True)
        recognition = StreamRecognition(runtime, self.config, owner="session")
        face = _object(1)

        recognition.process(self.image, [face], session.snapshot(), 1)
        runtime.futures[0].set_exception(RuntimeError("injected failure"))
        recognition.process(self.image, [face], session.snapshot(), 2)

        self.assertFalse(face["whitelisted"])
        self.assertEqual(runtime.calls, 1)

    async def test_late_result_cannot_apply_to_a_reused_track_id(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime(deferred=True)
        recognition = StreamRecognition(runtime, self.config, owner="session")

        original = _object(1)
        recognition.process(self.image, [original], session.snapshot(), 1)
        recognition.process(self.image, [], session.snapshot(), 2)
        recognition.process(self.image, [], session.snapshot(), 3)
        reused = _object(1)
        recognition.process(self.image, [reused], session.snapshot(), 4)

        runtime.futures[0].set_result(np.asarray([1.0, 0.0], dtype=np.float32))
        recognition.process(self.image, [reused], session.snapshot(), 5)
        self.assertFalse(reused["whitelisted"])
        runtime.futures[1].set_result(np.asarray([0.0, 1.0], dtype=np.float32))
        recognition.process(self.image, [reused], session.snapshot(), 6)

        self.assertFalse(reused["whitelisted"])
        self.assertEqual(recognition.stale_results, 1)

    async def test_periodic_result_cannot_apply_to_a_reused_whitelisted_track_id(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime(deferred=True)
        config = RecognitionConfig(revalidate_frames=3, missing_track_frames=1)
        recognition = StreamRecognition(runtime, config, owner="session")
        face = _object(1)

        recognition.process(self.image, [face], session.snapshot(), 1)
        runtime.futures[0].set_result(np.asarray([1.0, 0.0], dtype=np.float32))
        recognition.process(self.image, [face], session.snapshot(), 2)
        recognition.process(self.image, [face], session.snapshot(), 3)
        recognition.process(self.image, [face], session.snapshot(), 4)
        self.assertTrue(face["whitelisted"])

        recognition.process(self.image, [], session.snapshot(), 5)
        recognition.process(self.image, [], session.snapshot(), 6)
        reused = _object(1)
        recognition.process(self.image, [reused], session.snapshot(), 7)
        runtime.futures[1].set_result(np.asarray([1.0, 0.0], dtype=np.float32))
        recognition.process(self.image, [reused], session.snapshot(), 8)

        self.assertFalse(reused["whitelisted"])
        runtime.futures[2].set_result(np.asarray([0.0, 1.0], dtype=np.float32))
        recognition.process(self.image, [reused], session.snapshot(), 9)
        self.assertFalse(reused["whitelisted"])
        self.assertEqual(recognition.stale_results, 1)

    async def test_queue_overflow_and_stream_pending_limit_are_fail_closed(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime(deferred=True)
        config = RecognitionConfig(max_pending_per_stream=1)
        recognition = StreamRecognition(runtime, config, owner="session")
        faces = [_object(1), _object(2)]

        metrics = recognition.process(self.image, faces, session.snapshot(), 1)

        self.assertEqual(runtime.calls, 1)
        self.assertEqual(metrics["adaface_calls"], 1)
        self.assertEqual(metrics["adaface_queue_overflow"], 1)
        self.assertTrue(all(not face["whitelisted"] for face in faces))

        overflow_runtime = FakeRecognitionRuntime()
        overflow_runtime.overflow = True
        overflow_recognition = StreamRecognition(
            overflow_runtime,
            self.config,
            owner="session",
        )
        face = _object(3)
        overflow_metrics = overflow_recognition.process(
            self.image,
            [face],
            session.snapshot(),
            1,
        )
        self.assertEqual(overflow_metrics["adaface_queue_overflow"], 1)
        self.assertFalse(face["whitelisted"])

    async def test_multiple_sessions_and_streams_do_not_mix_results(self):
        sessions = SessionRegistry()
        session_a = sessions.get_or_create("session-a")
        session_b = sessions.get_or_create("session-b")
        session_a.append(np.asarray([1.0, 0.0]))
        session_b.append(np.asarray([0.0, 1.0]))
        runtime = PixelRecognitionRuntime()

        async def recognize(session_id, session, image):
            recognition = StreamRecognition(runtime, self.config, owner=session_id)
            face = _object(1)
            recognition.process(image, [face], session.snapshot(), 1)
            await asyncio.sleep(0)
            recognition.process(image, [face], session.snapshot(), 2)
            return face["whitelisted"]

        results = await asyncio.gather(
            recognize("session-a", session_a, np.zeros_like(self.image)),
            recognize("session-a", session_a, np.full_like(self.image, 255)),
            recognize("session-b", session_b, np.full_like(self.image, 255)),
            recognize("session-b", session_b, np.zeros_like(self.image)),
        )

        self.assertEqual(results, [True, False, True, False])
        self.assertEqual(runtime.calls, 4)


if __name__ == "__main__":
    unittest.main()
