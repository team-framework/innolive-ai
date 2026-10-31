from __future__ import annotations

import asyncio
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import grpc
import numpy as np

from ai_processor_server import GrpcServerSettings, build_grpc_server
from protos import ai_processor_pb2 as messages
from protos import ai_processor_pb2_grpc
from service import vod
from service.recognition import SessionRegistry
from service.runtime import RuntimeConfig


def _generate_sample(path: Path, *, seconds: int = 3) -> None:
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=320x240:rate=30",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=48000",
            "-t",
            str(seconds),
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-shortest",
            str(path),
        ],
        check=True,
        capture_output=True,
    )


def _frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return int(json.loads(result.stdout)["streams"][0]["nb_read_frames"])


class FakeTracker:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.frame_id = 0
        self.live_track_ids: frozenset[int] = frozenset()

    def reset(self) -> None:
        self.frame_id = 0


class SlowRuntime:
    ready = True

    def __init__(self, delay: float = 0.01) -> None:
        self.delay = delay

    async def infer(self, image: np.ndarray, tracker: FakeTracker, *, live: bool = True):
        await asyncio.sleep(self.delay)
        tracker.frame_id += 1
        return {"objects": [], "timing_ms": {}}


class FakeAdaFace:
    ready = True
    embedding_model = "fake:1"


class VodLoopback:
    def __init__(
        self,
        runtime: Any,
        vod_root: Path,
        *,
        adaface: Any | None = None,
        vod_max_jobs: int | None = 1,
        segment_frames: int = 10,
    ) -> None:
        self.runtime = runtime
        self.vod_root = vod_root
        self.adaface = adaface
        self.vod_max_jobs = vod_max_jobs
        self.segment_frames = segment_frames
        self.sessions = SessionRegistry()
        self.bundle = None
        self.channel = None
        self.stub = None

    def _tracker_factory(self, *_args: Any) -> FakeTracker:
        return FakeTracker()

    async def __aenter__(self) -> VodLoopback:
        # vod_max_jobs=None omits the field so the dataclass default applies.
        extra = {} if self.vod_max_jobs is None else {"vod_max_jobs": self.vod_max_jobs}
        settings = GrpcServerSettings(
            runtime=RuntimeConfig(engine=Path("unused.engine")),
            tracker_config=Path("unused.yaml"),
            host="127.0.0.1",
            port=0,
            shutdown_grace_seconds=0,
            vod_root=self.vod_root,
            vod_segment_frames=self.segment_frames,
            **extra,
        )
        self.bundle = build_grpc_server(
            settings,
            self.runtime,
            sessions=self.sessions,
            adaface=self.adaface,
            tracker_factory=self._tracker_factory,
            exit_process=lambda reason: None,
        )
        await self.bundle.set_serving(True)
        await self.bundle.server.start()
        self.channel = grpc.aio.insecure_channel(f"127.0.0.1:{self.bundle.bound_port}")
        await asyncio.wait_for(self.channel.channel_ready(), timeout=2.0)
        self.stub = ai_processor_pb2_grpc.VodProcessorStub(self.channel)
        return self

    async def __aexit__(self, *_exc_info: object) -> None:
        if self.channel is not None:
            await self.channel.close()
        if self.bundle is not None:
            await self.bundle.server.stop(0)
            if self.bundle.vod_servicer is not None:
                await self.bundle.vod_servicer.close()
            await self.bundle.servicer.close()


def _start_request(tmp: Path, name: str, **overrides: Any) -> messages.StartVodJobRequest:
    request = messages.StartVodJobRequest(
        job_id=name,
        input_path=str(tmp / "sample.mp4"),
        output_path=str(tmp / f"{name}.mp4"),
        work_dir=str(tmp / f"{name}-work"),
        segment_frames=10,
    )
    for key, value in overrides.items():
        setattr(request, key, value)
    return request


async def _watch_to_terminal(stub: Any, job_id: str, timeout: float = 20.0):
    async def collect():
        last = None
        async for status in stub.WatchVodJob(messages.VodJobRef(job_id=job_id)):
            last = status
        return last

    return await asyncio.wait_for(collect(), timeout=timeout)


async def _poll_until(stub: Any, job_id: str, predicate, timeout: float = 10.0):
    async def wait():
        while True:
            status = await stub.GetVodJob(messages.VodJobRef(job_id=job_id))
            if predicate(status):
                return status
            await asyncio.sleep(0.01)

    return await asyncio.wait_for(wait(), timeout=timeout)


class VodProcessorGrpcTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._shared = Path(tempfile.mkdtemp(prefix="vod-grpc-sample-"))
        _generate_sample(cls._shared / "sample.mp4")
        cls.input_frames = _frame_count(cls._shared / "sample.mp4")

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vod-grpc-"))
        (self.tmp / "sample.mp4").write_bytes((self._shared / "sample.mp4").read_bytes())

    async def test_start_then_watch_reaches_done(self):
        async with VodLoopback(SlowRuntime(0.004), self.tmp) as server:
            started = await server.stub.StartVodJob(_start_request(self.tmp, "job-done"))
            self.assertEqual(started.state, messages.VOD_JOB_STATE_PROCESSING)
            self.assertEqual(started.worker_target, server.bundle.vod_servicer.worker_target)
            self.assertGreater(started.width, 0)

            final = await _watch_to_terminal(server.stub, "job-done")
            self.assertEqual(final.state, messages.VOD_JOB_STATE_DONE)
            self.assertEqual(_frame_count(Path(final.output_path)), self.input_frames)

    async def test_pause_then_resume_completes(self):
        async with VodLoopback(SlowRuntime(0.01), self.tmp) as server:
            await server.stub.StartVodJob(_start_request(self.tmp, "job-pause"))
            await _poll_until(
                server.stub,
                "job-pause",
                lambda status: bool(status.checkpoint_token),
            )
            paused = await server.stub.PauseVodJob(messages.PauseVodJobRequest(job_id="job-pause"))
            self.assertEqual(paused.state, messages.VOD_JOB_STATE_PAUSED)
            self.assertTrue(paused.checkpoint_token)

            resume = _start_request(
                self.tmp,
                "job-pause",
                resume=True,
                checkpoint_token=paused.checkpoint_token,
            )
            restarted = await server.stub.StartVodJob(resume)
            self.assertEqual(restarted.state, messages.VOD_JOB_STATE_PROCESSING)
            final = await _watch_to_terminal(server.stub, "job-pause")
            self.assertEqual(final.state, messages.VOD_JOB_STATE_DONE)
            self.assertEqual(_frame_count(Path(final.output_path)), self.input_frames)

    async def test_cancel_removes_partial_output_and_purges_work_dir(self):
        async with VodLoopback(SlowRuntime(0.02), self.tmp) as server:
            await server.stub.StartVodJob(_start_request(self.tmp, "job-cancel"))
            cancelled = await server.stub.CancelVodJob(
                messages.CancelVodJobRequest(job_id="job-cancel", purge_work_dir=True)
            )
            self.assertEqual(cancelled.state, messages.VOD_JOB_STATE_CANCELLED)
            self.assertEqual(cancelled.error_code, "CANCELLED")
            self.assertFalse((self.tmp / "job-cancel.mp4.part").exists())
            self.assertFalse((self.tmp / "job-cancel.mp4").exists())
            self.assertFalse((self.tmp / "job-cancel-work").exists())

    async def test_second_job_is_resource_exhausted(self):
        async with VodLoopback(SlowRuntime(0.02), self.tmp, vod_max_jobs=1) as server:
            await server.stub.StartVodJob(_start_request(self.tmp, "job-a"))
            with self.assertRaises(grpc.aio.AioRpcError) as rejected:
                await server.stub.StartVodJob(_start_request(self.tmp, "job-b"))
            self.assertEqual(rejected.exception.code(), grpc.StatusCode.RESOURCE_EXHAUSTED)

    async def test_disconnected_start_releases_admission_slot(self):
        entered, interrupted = asyncio.Event(), asyncio.Event()
        validate = vod.VodJob.validate

        async def delayed(job, **kwargs):
            if job.job_id == "disconnected":
                entered.set()
                try:
                    await asyncio.Future()
                finally:
                    interrupted.set()
            return await validate(job, **kwargs)

        async with VodLoopback(SlowRuntime(), self.tmp) as server:
            with patch.object(vod.VodJob, "validate", delayed):
                call = server.stub.StartVodJob(_start_request(self.tmp, "disconnected"))
                await asyncio.wait_for(entered.wait(), 2)
                call.cancel()
                await asyncio.wait_for(interrupted.wait(), 2)
                cancelled = await _poll_until(
                    server.stub,
                    "disconnected",
                    lambda status: status.state == messages.VOD_JOB_STATE_CANCELLED,
                )
                self.assertEqual(cancelled.error_code, "CANCELLED")
                self.assertFalse((self.tmp / "disconnected-work").exists())
                replacement = await server.stub.StartVodJob(_start_request(self.tmp, "replacement"))
                self.assertEqual(replacement.state, messages.VOD_JOB_STATE_PROCESSING)
                final = await _watch_to_terminal(server.stub, "replacement")
                self.assertEqual(final.state, messages.VOD_JOB_STATE_DONE)

    async def test_cancel_rpc_during_validation_never_starts_processing(self):
        entered = asyncio.Event()

        async def delayed(_job, **_kwargs):
            entered.set()
            await asyncio.Future()

        async with VodLoopback(SlowRuntime(), self.tmp) as server:
            with patch.object(vod.VodJob, "validate", delayed):
                start = server.stub.StartVodJob(_start_request(self.tmp, "cancel-admission"))
                await asyncio.wait_for(entered.wait(), 2)
                cancelled = await server.stub.CancelVodJob(
                    messages.CancelVodJobRequest(job_id="cancel-admission", purge_work_dir=True),
                    timeout=2,
                )
                initial = await asyncio.wait_for(start, 2)
                self.assertEqual(initial.state, messages.VOD_JOB_STATE_CANCELLED)
                self.assertEqual(cancelled.state, messages.VOD_JOB_STATE_CANCELLED)
                self.assertFalse((self.tmp / "cancel-admission.mp4").exists())
                self.assertFalse((self.tmp / "cancel-admission-work").exists())

    async def test_pause_during_validation_remains_resumable(self):
        await self._pause_during_validation(drain=False)

    async def test_drain_during_validation_remains_resumable(self):
        await self._pause_during_validation(drain=True)

    async def _pause_during_validation(self, *, drain):
        entered, release, pause_requested = (asyncio.Event(), asyncio.Event(), asyncio.Event())
        validate, request_pause = vod.VodJob.validate, vod.VodJob.request_pause

        async def delayed(job, **kwargs):
            entered.set()
            await release.wait()
            return await validate(job, **kwargs)

        def pause(job, reason=vod.PAUSE_REQUESTED):
            request_pause(job, reason)
            pause_requested.set()

        job_id = f"pause-admission-{drain}"
        async with VodLoopback(SlowRuntime(), self.tmp) as server:
            with (
                patch.object(vod.VodJob, "validate", delayed),
                patch.object(vod.VodJob, "request_pause", pause),
            ):
                start = server.stub.StartVodJob(_start_request(self.tmp, job_id))
                await asyncio.wait_for(entered.wait(), 2)
                if drain:
                    server.bundle.servicer.set_accepting(False)
                    pause_call = None
                else:
                    pause_call = server.stub.PauseVodJob(messages.PauseVodJobRequest(job_id=job_id))
                await asyncio.wait_for(pause_requested.wait(), 2)
                release.set()
                initial = await asyncio.wait_for(start, 2)
                self.assertEqual(initial.state, messages.VOD_JOB_STATE_PROCESSING)
                self.assertEqual((initial.width, initial.height), (320, 240))
                paused = await _watch_to_terminal(server.stub, job_id)
                self.assertEqual(paused.state, messages.VOD_JOB_STATE_PAUSED)
                expected = vod.PAUSE_WORKER_DRAINING if drain else vod.PAUSE_REQUESTED
                self.assertEqual(paused.pause_reason, expected)
                if pause_call is not None:
                    self.assertEqual((await pause_call).state, messages.VOD_JOB_STATE_PAUSED)
            server.bundle.servicer.set_accepting(True)
            await server.stub.StartVodJob(_start_request(self.tmp, job_id, resume=True))
            self.assertEqual(
                (await _watch_to_terminal(server.stub, job_id)).state,
                messages.VOD_JOB_STATE_DONE,
            )

    async def test_unexpected_validation_error_releases_slot(self):
        async with VodLoopback(SlowRuntime(), self.tmp) as server:
            with patch.object(vod.VodJob, "validate", side_effect=OSError("probe failed")):
                failed = await server.stub.StartVodJob(_start_request(self.tmp, "probe-error"))
            self.assertEqual(failed.state, messages.VOD_JOB_STATE_FAILED)
            self.assertEqual(failed.error_code, "INTERNAL")
            replacement = await server.stub.StartVodJob(_start_request(self.tmp, "after-error"))
            self.assertEqual(replacement.state, messages.VOD_JOB_STATE_PROCESSING)

    async def test_draining_auto_pauses_and_rejects_new_jobs(self):
        async with VodLoopback(SlowRuntime(0.01), self.tmp) as server:
            await server.stub.StartVodJob(_start_request(self.tmp, "job-drain"))
            await _poll_until(
                server.stub, "job-drain", lambda status: bool(status.checkpoint_token)
            )
            # Simulate rolling-replace drain on the live servicer.
            server.bundle.servicer.set_accepting(False)

            paused = await _poll_until(
                server.stub,
                "job-drain",
                lambda status: status.state == messages.VOD_JOB_STATE_PAUSED,
            )
            self.assertEqual(paused.pause_reason, messages.VOD_PAUSE_REASON_WORKER_DRAINING)

            with self.assertRaises(grpc.aio.AioRpcError) as rejected:
                await server.stub.StartVodJob(_start_request(self.tmp, "job-new"))
            self.assertEqual(rejected.exception.code(), grpc.StatusCode.UNAVAILABLE)

    async def test_high_quality_is_failed_precondition(self):
        async with VodLoopback(SlowRuntime(), self.tmp) as server:
            request = _start_request(self.tmp, "job-hi", quality=messages.VOD_QUALITY_HIGH)
            with self.assertRaises(grpc.aio.AioRpcError) as rejected:
                await server.stub.StartVodJob(request)
            self.assertEqual(rejected.exception.code(), grpc.StatusCode.FAILED_PRECONDITION)

    async def test_embedding_model_mismatch_returns_failed_status(self):
        async with VodLoopback(SlowRuntime(), self.tmp, adaface=FakeAdaFace()) as server:
            request = _start_request(self.tmp, "job-wl")
            request.whitelist.embedding_model = "other-model"
            request.whitelist.entries.add(entry_id="e1", values=[1.0, 0.0])
            status = await server.stub.StartVodJob(request)
            self.assertEqual(status.state, messages.VOD_JOB_STATE_FAILED)
            self.assertEqual(status.error_code, "EMBEDDING_MODEL_MISMATCH")

    async def test_disabled_worker_is_unavailable(self):
        async with VodLoopback(SlowRuntime(), self.tmp, vod_max_jobs=0) as server:
            with self.assertRaises(grpc.aio.AioRpcError) as rejected:
                await server.stub.StartVodJob(_start_request(self.tmp, "job-x"))
            self.assertEqual(rejected.exception.code(), grpc.StatusCode.UNAVAILABLE)

    async def test_default_settings_disable_all_vod_rpcs(self):
        # vod_max_jobs=None -> use the GrpcServerSettings default (0, disabled).
        async with VodLoopback(SlowRuntime(), self.tmp, vod_max_jobs=None) as server:
            with self.assertRaises(grpc.aio.AioRpcError) as start:
                await server.stub.StartVodJob(_start_request(self.tmp, "job-y"))
            self.assertEqual(start.exception.code(), grpc.StatusCode.UNAVAILABLE)

            with self.assertRaises(grpc.aio.AioRpcError) as get:
                await server.stub.GetVodJob(messages.VodJobRef(job_id="job-y"))
            self.assertEqual(get.exception.code(), grpc.StatusCode.UNAVAILABLE)

            with self.assertRaises(grpc.aio.AioRpcError) as listed:
                await server.stub.ListVodJobs(messages.ListVodJobsRequest())
            self.assertEqual(listed.exception.code(), grpc.StatusCode.UNAVAILABLE)

            # The live AiProcessor service stays healthy on a disabled worker.
            self.assertIsNotNone(server.bundle.servicer)


class VodAdmissionSchedulingTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_before_worker_task_starts_completes_admission(self):
        from ai_processor_server import VodProcessorServicer

        settings = GrpcServerSettings(
            runtime=RuntimeConfig(engine=Path("unused.engine")),
            tracker_config=Path("unused.yaml"),
            vod_max_jobs=1,
        )
        servicer = VodProcessorServicer(
            SlowRuntime(),
            settings,
            sessions=SessionRegistry(),
            adaface=None,
        )
        servicer.set_accepting(True)
        create_task = asyncio.create_task

        def cancel_before_running(coroutine, **kwargs):
            task = create_task(coroutine, **kwargs)
            task.cancel()
            return task

        with patch("ai_processor_server.asyncio.create_task", cancel_before_running):
            result = await asyncio.wait_for(
                servicer.StartVodJob(_start_request(Path("unused"), "pre-start"), _FakeContext()),
                timeout=2,
            )
        self.assertEqual(result.state, messages.VOD_JOB_STATE_CANCELLED)
        self.assertTrue(servicer._jobs["pre-start"].task.cancelled())


class VodMaxJobsDefaultTests(unittest.TestCase):
    def test_settings_default_disables_vod(self):
        settings = GrpcServerSettings(
            runtime=RuntimeConfig(engine=Path("unused.engine")),
            tracker_config=Path("unused.yaml"),
        )
        self.assertEqual(settings.vod_max_jobs, 0)

    def test_parse_args_vod_max_jobs_default_and_override(self):
        import os
        import sys
        from unittest import mock

        from ai_processor_server import parse_args

        with mock.patch.object(sys, "argv", ["ai_processor_server"]):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("VOD_MAX_JOBS", None)
                self.assertEqual(parse_args().vod_max_jobs, 0)
            with mock.patch.dict(os.environ, {"VOD_MAX_JOBS": "2"}):
                self.assertEqual(parse_args().vod_max_jobs, 2)


class VodCancelCleanupTests(unittest.TestCase):
    def test_cancel_cleanup_never_leaves_the_vod_root(self):
        from ai_processor_server import VodProcessorServicer, _VodJobHandle

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "vod"
            outside = Path(tmp) / "outside"
            inside = root / "job" / "work"
            for directory in (outside, inside):
                directory.mkdir(parents=True)
                (directory / "keep.txt").write_text("x")

            handle = _VodJobHandle(status=messages.VodJobStatus(job_id="j"))
            handle.work_dir = str(outside)
            handle.output_path = str(outside / "out.mp4")
            (outside / "out.mp4.part").write_text("x")
            VodProcessorServicer._cleanup_cancel(handle, True, root)
            self.assertTrue((outside / "keep.txt").exists())
            self.assertTrue((outside / "out.mp4.part").exists())

            handle.work_dir = str(root)
            VodProcessorServicer._cleanup_cancel(handle, True, root)
            self.assertTrue(root.exists())

            handle.work_dir = str(inside)
            VodProcessorServicer._cleanup_cancel(handle, True, root)
            self.assertFalse(inside.exists())


class _FakeClock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class _FakeContext:
    async def abort(self, code: Any, details: str) -> None:
        raise AssertionError(f"unexpected abort: {code} {details}")


def _make_servicer(clock: _FakeClock, **settings_overrides: Any):
    from ai_processor_server import GrpcServerSettings, VodProcessorServicer

    settings = GrpcServerSettings(
        runtime=RuntimeConfig(engine=Path("unused.engine")),
        tracker_config=Path("unused.yaml"),
        vod_max_jobs=1,
        **settings_overrides,
    )
    return VodProcessorServicer(
        object(),
        settings,
        sessions=SessionRegistry(),
        adaface=None,
        clock=clock,
    )


def _insert(servicer: Any, job_id: str, state: int) -> Any:
    from ai_processor_server import _VodJobHandle

    handle = _VodJobHandle(status=messages.VodJobStatus(job_id=job_id, state=state))
    servicer._jobs[job_id] = handle
    return handle


class VodJobPruneTests(unittest.TestCase):
    def test_ttl_counts_from_terminal_time_not_insertion(self):
        clock = _FakeClock(0.0)
        servicer = _make_servicer(clock, vod_job_ttl_seconds=10.0, vod_max_retained_jobs=100)
        # Inserted while still PROCESSING; the TTL must not start ticking yet.
        handle = _insert(servicer, "job", messages.VOD_JOB_STATE_PROCESSING)
        clock.now = 100.0
        servicer._prune_jobs()
        self.assertIsNone(handle.terminal_at)
        # Becomes terminal at t=100; TTL now starts from there.
        handle.status.state = messages.VOD_JOB_STATE_DONE
        servicer._prune_jobs()
        self.assertEqual(handle.terminal_at, 100.0)

        clock.now = 109.0  # before ttl relative to terminal time (100 + 10)
        servicer._prune_jobs()
        self.assertIn("job", servicer._jobs)

        clock.now = 110.0  # at ttl
        servicer._prune_jobs()
        self.assertNotIn("job", servicer._jobs)

    def test_listvodjobs_drops_expired_job(self):
        clock = _FakeClock(0.0)
        servicer = _make_servicer(clock, vod_job_ttl_seconds=5.0, vod_max_retained_jobs=100)
        _insert(servicer, "done", messages.VOD_JOB_STATE_DONE)
        servicer._prune_jobs()  # stamps terminal_at = 0
        clock.now = 4.0
        listed = asyncio.run(servicer.ListVodJobs(messages.ListVodJobsRequest(), _FakeContext()))
        self.assertEqual([job.job_id for job in listed.jobs], ["done"])
        clock.now = 5.0
        listed = asyncio.run(servicer.ListVodJobs(messages.ListVodJobsRequest(), _FakeContext()))
        self.assertEqual(listed.jobs, [])

    def test_cap_evicts_oldest_terminal_first(self):
        clock = _FakeClock(1.0)
        servicer = _make_servicer(clock, vod_job_ttl_seconds=1e9, vod_max_retained_jobs=2)
        # Each job becomes terminal at a distinct clock value.
        _insert(servicer, "a", messages.VOD_JOB_STATE_DONE)
        servicer._prune_jobs()
        clock.now = 2.0
        _insert(servicer, "b", messages.VOD_JOB_STATE_DONE)
        servicer._prune_jobs()
        clock.now = 3.0
        _insert(servicer, "c", messages.VOD_JOB_STATE_DONE)
        servicer._prune_jobs()
        # cap=2, three terminal jobs → the oldest (a, terminal_at=1) is evicted.
        self.assertEqual(sorted(servicer._jobs), ["b", "c"])

    def test_cap_ignores_processing_jobs(self):
        clock = _FakeClock(1.0)
        servicer = _make_servicer(clock, vod_job_ttl_seconds=1e9, vod_max_retained_jobs=2)
        _insert(servicer, "p", messages.VOD_JOB_STATE_PROCESSING)
        _insert(servicer, "a", messages.VOD_JOB_STATE_DONE)
        servicer._prune_jobs()
        clock.now = 2.0
        _insert(servicer, "b", messages.VOD_JOB_STATE_DONE)
        servicer._prune_jobs()
        clock.now = 3.0
        _insert(servicer, "c", messages.VOD_JOB_STATE_DONE)
        servicer._prune_jobs()
        # Only terminal jobs count toward the cap; PROCESSING "p" is never evicted.
        self.assertEqual(sorted(servicer._jobs), ["b", "c", "p"])

    def test_non_terminal_survives_ttl_and_cap(self):
        clock = _FakeClock(0.0)
        servicer = _make_servicer(clock, vod_job_ttl_seconds=1.0, vod_max_retained_jobs=0)
        handle = _insert(servicer, "live", messages.VOD_JOB_STATE_PROCESSING)
        servicer._prune_jobs()
        clock.now = 1000.0  # far beyond ttl
        servicer._prune_jobs()
        self.assertIn("live", servicer._jobs)
        self.assertIsNone(handle.terminal_at)

    def test_paused_older_than_ttl_stays_in_get_and_list(self):
        clock = _FakeClock(0.0)
        servicer = _make_servicer(clock, vod_job_ttl_seconds=10.0, vod_max_retained_jobs=100)
        paused = _insert(servicer, "paused", messages.VOD_JOB_STATE_PAUSED)
        _insert(servicer, "done", messages.VOD_JOB_STATE_DONE)
        servicer._prune_jobs()
        self.assertIsNone(paused.terminal_at)

        clock.now = 1000.0  # far beyond ttl for both
        got = asyncio.run(servicer.GetVodJob(messages.VodJobRef(job_id="paused"), _FakeContext()))
        self.assertEqual(got.state, messages.VOD_JOB_STATE_PAUSED)
        listed = asyncio.run(servicer.ListVodJobs(messages.ListVodJobsRequest(), _FakeContext()))
        self.assertEqual([job.job_id for job in listed.jobs], ["paused"])

    def test_paused_jobs_above_cap_are_kept_while_finished_jobs_are_evicted(self):
        clock = _FakeClock(1.0)
        servicer = _make_servicer(clock, vod_job_ttl_seconds=1e9, vod_max_retained_jobs=2)
        for index in range(4):
            _insert(servicer, f"paused-{index}", messages.VOD_JOB_STATE_PAUSED)
        finished = [
            ("failed", messages.VOD_JOB_STATE_FAILED),
            ("cancelled", messages.VOD_JOB_STATE_CANCELLED),
            ("done", messages.VOD_JOB_STATE_DONE),
        ]
        for job_id, state in finished:
            _insert(servicer, job_id, state)
            servicer._prune_jobs()
            clock.now += 1.0
        # 4 PAUSED exceed the cap of 2 on their own yet all stay; only the oldest finished
        # job (failed) is evicted so DONE/FAILED/CANCELLED fit the cap.
        self.assertEqual(
            sorted(servicer._jobs),
            ["cancelled", "done", "paused-0", "paused-1", "paused-2", "paused-3"],
        )

    def test_failed_and_cancelled_expire_after_ttl(self):
        clock = _FakeClock(0.0)
        servicer = _make_servicer(clock, vod_job_ttl_seconds=5.0, vod_max_retained_jobs=100)
        _insert(servicer, "failed", messages.VOD_JOB_STATE_FAILED)
        _insert(servicer, "cancelled", messages.VOD_JOB_STATE_CANCELLED)
        servicer._prune_jobs()
        clock.now = 5.0
        servicer._prune_jobs()
        self.assertEqual(servicer._jobs, {})


class VodRetentionSettingsTests(unittest.TestCase):
    def test_defaults(self):
        settings = GrpcServerSettings(
            runtime=RuntimeConfig(engine=Path("unused.engine")),
            tracker_config=Path("unused.yaml"),
        )
        self.assertEqual(settings.vod_job_ttl_seconds, 86400.0)
        self.assertEqual(settings.vod_max_retained_jobs, 100)

    def test_rejects_non_positive_ttl(self):
        for ttl in (0.0, -1.0, float("inf")):
            with self.assertRaises(ValueError):
                GrpcServerSettings(
                    runtime=RuntimeConfig(engine=Path("unused.engine")),
                    tracker_config=Path("unused.yaml"),
                    vod_job_ttl_seconds=ttl,
                )

    def test_rejects_negative_cap(self):
        with self.assertRaises(ValueError):
            GrpcServerSettings(
                runtime=RuntimeConfig(engine=Path("unused.engine")),
                tracker_config=Path("unused.yaml"),
                vod_max_retained_jobs=-1,
            )

    def test_parse_args_defaults_and_env_overrides(self):
        import os
        import sys
        from unittest import mock

        from ai_processor_server import parse_args

        with mock.patch.object(sys, "argv", ["ai_processor_server"]):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("VOD_JOB_TTL_SECONDS", None)
                os.environ.pop("VOD_MAX_RETAINED_JOBS", None)
                args = parse_args()
                self.assertEqual(args.vod_job_ttl_seconds, 86400.0)
                self.assertEqual(args.vod_max_retained_jobs, 100)
            with mock.patch.dict(
                os.environ,
                {"VOD_JOB_TTL_SECONDS": "30", "VOD_MAX_RETAINED_JOBS": "5"},
            ):
                args = parse_args()
                self.assertEqual(args.vod_job_ttl_seconds, 30.0)
                self.assertEqual(args.vod_max_retained_jobs, 5)


if __name__ == "__main__":
    unittest.main()
