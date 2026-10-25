from __future__ import annotations

import asyncio
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
from ultralytics.engine.results import Boxes

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
from service.tracking import StreamTracker


def _object(track_id: int) -> dict[str, Any]:
    return {
        "track_id": track_id,
        "class_id": 0,
        "class_name": "face",
        "bbox": [10.0, 10.0, 70.0, 70.0],
        "mask_polygon": [[10.0, 10.0], [70.0, 10.0], [70.0, 70.0]],
        "confidence": 0.9,
    }


def _number_plate(track_id: int) -> dict[str, Any]:
    item = _object(track_id)
    item.update(class_id=1, class_name="number_plate")
    return item


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

    async def test_import_keeps_entry_ids_and_skips_present_entries(self):
        source = SessionRegistry()
        first, _, _ = source.append("session-a", np.asarray([2.0, 0.0]))
        second, _, _ = source.append("session-a", np.asarray([0.0, 3.0]))
        exported = [
            (entry.entry_id, entry.embedding) for entry in source.snapshot("session-a").entries
        ]
        target = SessionRegistry()

        added, count, version = target.import_entries("session-a", exported)
        again = target.import_entries("session-a", exported)
        snapshot = target.snapshot("session-a")

        self.assertEqual((added, count, version), (2, 2, 1))
        self.assertEqual(again, (0, 2, 1))
        self.assertEqual(
            [entry.entry_id for entry in snapshot.entries], [first.entry_id, second.entry_id]
        )
        self.assertTrue(np.allclose(snapshot.entries[1].embedding, [0.0, 1.0]))

    async def test_import_is_all_or_nothing(self):
        sessions = SessionRegistry(max_entries_per_session=2)
        sessions.append("session-a", np.asarray([1.0, 0.0]))

        with self.assertRaises(WhitelistLimitError):
            sessions.import_entries(
                "session-a",
                [("entry-b", np.asarray([0.0, 1.0])), ("entry-c", np.asarray([1.0, 1.0]))],
            )
        with self.assertRaises(ValueError):
            sessions.import_entries(
                "session-a",
                [("entry-b", np.asarray([0.0, 1.0])), ("entry-c", np.asarray([0.0, 0.0]))],
            )
        with self.assertRaises(ValueError):
            sessions.import_entries("session-a", [("   ", np.asarray([0.0, 1.0]))])

        self.assertEqual(len(sessions.snapshot("session-a").entries), 1)

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

    async def test_number_plate_never_enters_face_recognition(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime()
        recognition = StreamRecognition(runtime, self.config, owner="session")
        plate = _number_plate(1)

        metrics = recognition.process(self.image, [plate], session.snapshot(), 1)

        self.assertEqual(runtime.calls, 0)
        self.assertEqual(metrics["adaface_calls"], 0)
        self.assertFalse(plate["whitelisted"])

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
        self.assertEqual(runtime.calls, 1)

    async def test_approved_track_stays_unblurred_beyond_revalidation_intervals(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime(
            [
                np.asarray([1.0, 0.0]),
                np.asarray([0.0, 1.0]),
            ]
        )
        recognition = StreamRecognition(runtime, RecognitionConfig(revalidate_frames=3), owner="s")
        face = _object(1)
        recognition.process(self.image, [face], session.snapshot(), 1)
        self.assertFalse(face["whitelisted"])

        for frame in range(2, 301):
            metrics = recognition.process(self.image, [face], session.snapshot(), frame)
            self.assertTrue(face["whitelisted"], frame)
            self.assertEqual(metrics["adaface_calls"], 0)

        self.assertEqual(runtime.calls, 1)
        self.assertIsNone(recognition._states[1].pending_token)

    async def test_approved_track_does_not_need_another_valid_crop_or_queue_slot(self):
        for condition in ("held", "small", "invalid_bbox", "runtime_overflow", "stream_limit"):
            with self.subTest(condition=condition):
                session = SessionRegistry().get_or_create("session")
                session.append(np.asarray([1.0, 0.0]))
                runtime = FakeRecognitionRuntime(deferred=True)
                config = RecognitionConfig(revalidate_frames=3, max_pending_per_stream=1)
                recognition = StreamRecognition(runtime, config, owner="s")
                face = _object(1)
                recognition.process(self.image, [face], session.snapshot(), 1)
                runtime.futures[0].set_result(np.asarray([1.0, 0.0]))
                recognition.process(self.image, [face], session.snapshot(), 2)
                if condition == "held":
                    face["held"] = True
                elif condition == "small":
                    face["bbox"] = [10, 10, 20, 20]
                elif condition == "invalid_bbox":
                    face["bbox"] = None
                elif condition == "runtime_overflow":
                    runtime.overflow = True
                else:
                    recognition.process(self.image, [face, _object(2)], session.snapshot(), 3)

                metrics = recognition.process(self.image, [face], session.snapshot(), 4)
                self.assertTrue(face["whitelisted"])
                self.assertEqual(metrics["adaface_calls"], 0)
                self.assertEqual(metrics["adaface_queue_overflow"], 0)
                self.assertEqual(runtime.calls, 2 if condition == "stream_limit" else 1)
                recognition.close()

    async def test_stale_initial_negative_or_error_remains_protected_and_retries(self):
        for outcome in ("negative", "error"):
            with self.subTest(outcome=outcome):
                session = SessionRegistry().get_or_create("session")
                session.append(np.asarray([1.0, 0.0]))
                runtime = FakeRecognitionRuntime(deferred=True)
                recognition = StreamRecognition(runtime, self.config, owner="session")
                face = _object(1)
                recognition.process(self.image, [face], session.snapshot(), 1)
                session.append(np.asarray([0.0, 1.0]))
                if outcome == "negative":
                    runtime.futures[0].set_result(np.asarray([0.0, 1.0]))
                else:
                    runtime.futures[0].set_exception(RuntimeError("alignment failed"))
                recognition.process(self.image, [face], session.snapshot(), 2)
                self.assertFalse(face["whitelisted"])
                self.assertEqual(recognition.stale_results, 1)
                self.assertEqual(runtime.calls, 2)
                runtime.futures[1].set_result(np.asarray([0.0, 1.0]))
                recognition.process(self.image, [face], session.snapshot(), 3)
                self.assertTrue(face["whitelisted"])
                recognition.close()

    async def test_non_match_can_pass_a_periodic_retry_and_then_stops_retrying(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime([np.asarray([0.0, 1.0])] * 3 + [np.asarray([1.0, 0.0])])
        recognition = StreamRecognition(runtime, RecognitionConfig(), owner="session")
        face = _object(1)
        for frame in range(1, 101):
            metrics = recognition.process(self.image, [face], session.snapshot(), frame)
            self.assertEqual(face["whitelisted"], frame >= 42)
            self.assertEqual(metrics["adaface_calls"], int(frame in (1, 6, 11, 41)))
        self.assertEqual(runtime.calls, 4)

    async def test_initial_timeout_keeps_protection_and_retries(self):
        now = 0.0
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime(deferred=True)
        config = RecognitionConfig(pending_timeout_seconds=0.5)
        recognition = StreamRecognition(runtime, config, owner="s", clock=lambda: now)
        face = _object(1)
        recognition.process(self.image, [face], session.snapshot(), 1)
        now = 0.6
        recognition.process(self.image, [face], session.snapshot(), 2)
        self.assertTrue(runtime.futures[0].cancelled())
        self.assertFalse(face["whitelisted"])
        self.assertEqual(recognition.stale_results, 1)
        self.assertEqual(runtime.calls, 2)
        runtime.futures[1].set_result(np.asarray([1.0, 0.0]))
        recognition.process(self.image, [face], session.snapshot(), 3)
        self.assertTrue(face["whitelisted"])
        recognition.close()

    async def test_runtime_unavailable_preserves_approval_but_cannot_approve_new_tracks(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime()
        recognition = StreamRecognition(runtime, self.config, owner="session")
        face, newcomer = _object(1), _object(2)
        recognition.process(self.image, [face], session.snapshot(), 1)
        recognition.process(self.image, [face], session.snapshot(), 2)
        runtime.ready = False
        recognition.process(self.image, [face, newcomer], session.snapshot(), 100)
        self.assertTrue(face["whitelisted"])
        self.assertFalse(newcomer["whitelisted"])
        self.assertEqual(runtime.calls, 1)
        runtime.ready = True
        recognition.process(self.image, [face, newcomer], session.snapshot(), 101)
        recognition.process(self.image, [face, newcomer], session.snapshot(), 102)
        self.assertTrue(newcomer["whitelisted"])
        self.assertEqual(runtime.calls, 2)

    async def test_removing_the_matched_entry_revokes_approval_even_if_runtime_is_unavailable(self):
        session = SessionRegistry().get_or_create("session")
        matched, _, _ = session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime()
        recognition = StreamRecognition(runtime, self.config, owner="session")
        face = _object(1)
        recognition.process(self.image, [face], session.snapshot(), 1)
        recognition.process(self.image, [face], session.snapshot(), 2)
        self.assertTrue(face["whitelisted"])
        session.delete_entry(matched.entry_id)
        runtime.ready = False
        recognition.process(self.image, [face], session.snapshot(), 3)
        self.assertFalse(face["whitelisted"])
        self.assertIsNone(recognition._states[1].matched_entry_id)

    async def test_deleted_entry_cannot_be_approved_by_a_pending_result(self):
        session = SessionRegistry().get_or_create("session")
        matched, _, _ = session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime(deferred=True)
        recognition = StreamRecognition(runtime, self.config, owner="session")
        face = _object(1)
        recognition.process(self.image, [face], session.snapshot(), 1)
        session.delete_entry(matched.entry_id)
        runtime.futures[0].set_result(np.asarray([1.0, 0.0]))
        recognition.process(self.image, [face], session.snapshot(), 2)
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
        self.assertEqual(runtime.calls, 1)
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

    async def test_retired_approval_cannot_apply_to_a_reused_track_id(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime(deferred=True)
        recognition = StreamRecognition(runtime, self.config, owner="session")
        face = _object(1)
        recognition.process(self.image, [face], session.snapshot(), 1)
        runtime.futures[0].set_result(np.asarray([1.0, 0.0]))
        recognition.process(self.image, [face], session.snapshot(), 2)
        recognition.process(self.image, [], session.snapshot(), 3, live_track_ids=frozenset())
        reused = _object(1)
        recognition.process(self.image, [reused], session.snapshot(), 4)
        self.assertFalse(reused["whitelisted"])
        runtime.futures[1].set_result(np.asarray([0.0, 1.0]))
        recognition.process(self.image, [reused], session.snapshot(), 5)
        self.assertFalse(reused["whitelisted"])
        self.assertEqual(runtime.calls, 2)

    async def test_new_stream_requires_approval_for_the_same_numeric_id(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime()
        first = StreamRecognition(runtime, self.config, owner="session")
        face = _object(1)
        first.process(self.image, [face], session.snapshot(), 1)
        first.process(self.image, [face], session.snapshot(), 2)
        self.assertTrue(face["whitelisted"])
        first.close()
        second = StreamRecognition(runtime, self.config, owner="session")
        second.process(self.image, [face], session.snapshot(), 1)
        self.assertFalse(face["whitelisted"])
        self.assertEqual(runtime.calls, 2)
        second.close()

    async def test_botsort_recovery_preserves_approval_until_the_track_is_retired(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime([np.asarray([1.0, 0.0]), np.asarray([0.0, 1.0])])
        recognition = StreamRecognition(runtime, RecognitionConfig(), owner="session")
        tracker = StreamTracker(gmc_method="none")

        def process_frame(detected):
            rows = [[10, 10, 70, 70, 0.9, 0]] if detected else []
            detections = Boxes(np.asarray(rows, dtype=np.float32).reshape(-1, 6), (96, 96))
            tracks = tracker.update(detections.numpy(), self.image)
            objects = [_object(int(track[4])) for track in tracks]
            objects, _ = tracker.stabilize(objects, 96, 96)
            recognition.process(
                self.image,
                objects,
                session.snapshot(),
                tracker.frame_id,
                live_track_ids=tracker.live_track_ids,
            )
            return objects

        self.assertFalse(process_frame(True)[0]["whitelisted"])
        self.assertTrue(process_frame(True)[0]["whitelisted"])
        # The first missing frame has a held mask; subsequent frames have no object.
        self.assertTrue(process_frame(False)[0]["whitelisted"])
        for _ in range(10):
            self.assertEqual(process_frame(False), [])
            self.assertIn(1, tracker.live_track_ids)
        recovered = process_frame(True)[0]
        self.assertEqual(recovered["track_id"], 1)
        self.assertTrue(recovered["whitelisted"])
        self.assertEqual(runtime.calls, 1)

        for _ in range(32):
            process_frame(False)
        self.assertNotIn(1, tracker.live_track_ids)
        self.assertNotIn(1, recognition._states)
        process_frame(True)  # BoT-SORT confirms a newly activated ID on its next detection.
        newcomer = process_frame(True)[0]
        self.assertNotEqual(newcomer["track_id"], 1)
        self.assertFalse(newcomer["whitelisted"])
        self.assertFalse(process_frame(True)[0]["whitelisted"])
        self.assertEqual(runtime.calls, 2)
        tracker.reset()
        recognition.close()
        self.assertEqual(tracker.live_track_ids, frozenset())

    async def test_deleting_the_matched_entry_revokes_a_lost_tracks_approval(self):
        session = SessionRegistry().get_or_create("session")
        matched, _, _ = session.append(np.asarray([1.0, 0.0]))
        session.append(np.asarray([0.0, 1.0]))
        runtime = FakeRecognitionRuntime()
        recognition = StreamRecognition(runtime, self.config, owner="session")
        face = _object(1)
        recognition.process(self.image, [face], session.snapshot(), 1)
        recognition.process(self.image, [face], session.snapshot(), 2)
        for frame in range(3, 10):
            recognition.process(
                self.image, [], session.snapshot(), frame, live_track_ids=frozenset({1})
            )
        session.delete_entry(matched.entry_id)
        recognition.process(self.image, [], session.snapshot(), 10, live_track_ids=frozenset({1}))
        recognition.process(self.image, [face], session.snapshot(), 11)
        recognition.process(self.image, [face], session.snapshot(), 12)
        self.assertFalse(face["whitelisted"])
        self.assertEqual(runtime.calls, 2)

    async def test_live_but_unapproved_track_keeps_the_original_missing_timeout(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime([np.asarray([0.0, 1.0])])
        recognition = StreamRecognition(runtime, self.config, owner="session")
        face = _object(1)
        recognition.process(self.image, [face], session.snapshot(), 1)
        recognition.process(self.image, [face], session.snapshot(), 2)
        for frame in (3, 4):
            recognition.process(
                self.image, [], session.snapshot(), frame, live_track_ids=frozenset({1})
            )
        recognition.process(self.image, [face], session.snapshot(), 5)
        self.assertFalse(face["whitelisted"])
        self.assertEqual(runtime.calls, 2)

    async def test_number_plate_cannot_inherit_approval_from_a_face_track_id(self):
        session = SessionRegistry().get_or_create("session")
        session.append(np.asarray([1.0, 0.0]))
        runtime = FakeRecognitionRuntime()
        recognition = StreamRecognition(runtime, self.config, owner="session")
        face = _object(1)
        recognition.process(self.image, [face], session.snapshot(), 1)
        recognition.process(self.image, [face], session.snapshot(), 2)
        self.assertTrue(face["whitelisted"])
        plate = _number_plate(1)
        metrics = recognition.process(self.image, [plate], session.snapshot(), 100)
        self.assertFalse(plate["whitelisted"])
        self.assertEqual(metrics["whitelisted_tracks"], 0)
        self.assertEqual(runtime.calls, 1)

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
