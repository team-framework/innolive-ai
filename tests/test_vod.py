from __future__ import annotations

import errno
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from service import vod


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


def _video_frame_count(path: Path) -> int:
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


def _audio_streams(path: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_name,channels",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout).get("streams", [])


class FakeTracker:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.frame_id = 0
        self.live_track_ids: frozenset[int] = frozenset()

    def reset(self) -> None:
        self.frame_id = 0


class FakeRuntime:
    ready = True

    def __init__(self) -> None:
        self.calls = 0
        self.live_flags: list[bool] = []

    async def infer(self, image: np.ndarray, tracker: FakeTracker, *, live: bool = True):
        self.calls += 1
        self.live_flags.append(live)
        tracker.frame_id += 1
        return {"objects": [], "timing_ms": {}}


def _job(
    tmp: Path,
    runtime: FakeRuntime,
    *,
    input_path: Path,
    output_name: str = "out.mp4",
    segment_frames: int = 30,
    limits: vod.VodLimits | None = None,
    progress_callback=None,
) -> vod.VodJob:
    return vod.VodJob(
        job_id="job-1",
        input_path=str(input_path),
        output_path=str(tmp / output_name),
        work_dir=str(tmp / "work"),
        runtime=runtime,
        adaface=None,
        config=vod.VodConfig(segment_frames=segment_frames),
        limits=limits or vod.VodLimits(),
        whitelist=vod.VodWhitelist("", ()),
        tracker_factory=FakeTracker,
        tracker_config=Path("unused.yaml"),
        device="cpu",
        progress_callback=progress_callback,
    )


class VodCoreTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._shared = Path(tempfile.mkdtemp(prefix="vod-sample-"))
        cls.sample = cls._shared / "sample.mp4"
        _generate_sample(cls.sample)
        cls.input_frames = _video_frame_count(cls.sample)

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vod-test-"))

    async def test_processes_all_frames_with_audio_and_monotonic_progress(self):
        runtime = FakeRuntime()
        snapshots: list[vod.VodProgressSnapshot] = []
        job = _job(
            self.tmp,
            runtime,
            input_path=self.sample,
            progress_callback=lambda snapshot, _committed: snapshots.append(snapshot),
        )
        result = await job.run()

        self.assertEqual(result.state, vod.STATE_DONE)
        output = Path(result.output_path)
        self.assertTrue(output.is_file())
        self.assertEqual(_video_frame_count(output), self.input_frames)
        self.assertEqual(runtime.calls, self.input_frames)
        self.assertTrue(all(flag is False for flag in runtime.live_flags))

        audio = _audio_streams(output)
        self.assertEqual(len(audio), 1)
        self.assertEqual(audio[0]["codec_name"], "aac")
        self.assertEqual(audio[0]["channels"], 1)

        frames = [snapshot.frames_done for snapshot in snapshots]
        self.assertEqual(frames, sorted(frames))
        self.assertGreaterEqual(len([s for s in snapshots if s.segments_done]), 1)
        self.assertTrue(all(snapshot.fraction < 1.0 for snapshot in snapshots[:-1]))
        self.assertEqual(snapshots[-1].fraction, 1.0)

    async def test_resume_without_committed_checkpoint_starts_fresh(self):
        runtime = FakeRuntime()
        job = _job(self.tmp, runtime, input_path=self.sample)
        result = await job.run(resume=True, checkpoint_token="")

        self.assertEqual(result.state, vod.STATE_DONE)
        self.assertEqual(_video_frame_count(Path(result.output_path)), self.input_frames)

    async def test_work_directory_disk_full_is_reported_before_inference(self):
        runtime = FakeRuntime()
        job = _job(self.tmp, runtime, input_path=self.sample)
        with (
            patch.object(Path, "mkdir", side_effect=OSError(errno.ENOSPC, "disk full")),
            self.assertRaises(vod.VodError) as caught,
        ):
            await job.run()
        self.assertEqual(caught.exception.code, "DISK_FULL")
        self.assertEqual(runtime.calls, 0)
        self.assertFalse((self.tmp / "out.mp4").exists())

    async def test_checkpoint_disk_full_preserves_durable_state_for_resume(self):
        job = _job(self.tmp, FakeRuntime(), input_path=self.sample)
        snapshots = []
        durable = {}

        def progress(snapshot, committed):
            snapshots.append(snapshot)
            if committed:
                durable["token"] = snapshot.checkpoint_token
                durable["bytes"] = (job.work_dir / "checkpoint.json").read_bytes()

        job._progress_callback = progress
        original_fsync = vod.os.fsync

        def fail_after_first_commit(fd):
            if durable:
                raise OSError(errno.ENOSPC, "disk full")
            return original_fsync(fd)

        with (
            patch("service.vod.os.fsync", side_effect=fail_after_first_commit),
            self.assertRaises(vod.VodError) as caught,
        ):
            await job.run()
        self.assertEqual(caught.exception.code, "DISK_FULL")
        self.assertEqual((job.work_dir / "checkpoint.json").read_bytes(), durable["bytes"])
        self.assertEqual(vod.read_checkpoint_token(job.work_dir), durable["token"])
        self.assertTrue(all(snapshot.fraction < 1.0 for snapshot in snapshots))
        self.assertFalse((self.tmp / "out.mp4").exists())

        resumed = await _job(self.tmp, FakeRuntime(), input_path=self.sample).run(
            resume=True, checkpoint_token=durable["token"]
        )
        self.assertEqual(resumed.state, vod.STATE_DONE)
        self.assertEqual(_video_frame_count(Path(resumed.output_path)), self.input_frames)
        self.assertEqual(_audio_streams(Path(resumed.output_path)), _audio_streams(self.sample))

    async def test_output_publish_disk_full_keeps_existing_output_and_incomplete_progress(self):
        output = self.tmp / "out.mp4"
        output.write_bytes(b"existing output")
        snapshots = []
        job = _job(
            self.tmp,
            FakeRuntime(),
            input_path=self.sample,
            progress_callback=lambda snapshot, _committed: snapshots.append(snapshot),
        )
        original_replace = vod.os.replace

        def fail_output_publish(source, destination):
            if Path(destination) == output.resolve():
                raise OSError(errno.ENOSPC, "disk full")
            return original_replace(source, destination)

        with (
            patch("service.vod.os.replace", side_effect=fail_output_publish),
            self.assertRaises(vod.VodError) as caught,
        ):
            await job.run(overwrite_output=True)
        self.assertEqual(caught.exception.code, "DISK_FULL")
        self.assertEqual(output.read_bytes(), b"existing output")
        self.assertTrue(vod.read_checkpoint_token(job.work_dir))
        self.assertTrue(all(snapshot.fraction < 1.0 for snapshot in snapshots))

    async def test_pause_after_last_frame_does_not_report_completion(self):
        snapshots = []
        job = _job(self.tmp, FakeRuntime(), input_path=self.sample)

        def progress(snapshot, committed):
            snapshots.append(snapshot)
            if committed and snapshot.frames_done == self.input_frames:
                job.request_pause()

        job._progress_callback = progress
        result = await job.run()
        self.assertEqual(result.state, vod.STATE_PAUSED)
        self.assertTrue(result.checkpoint_token)
        self.assertEqual(snapshots[-1].frames_done, self.input_frames)
        self.assertTrue(all(snapshot.fraction < 1.0 for snapshot in snapshots))
        self.assertFalse((self.tmp / "out.mp4").exists())

    async def test_pause_then_resume_completes_with_exact_frame_count(self):
        runtime = FakeRuntime()
        token: dict[str, str] = {}
        job = _job(self.tmp, runtime, input_path=self.sample)

        def on_progress(snapshot: vod.VodProgressSnapshot, committed: bool) -> None:
            if committed and snapshot.segments_done == 1:
                token["value"] = snapshot.checkpoint_token
                job.request_pause(vod.PAUSE_REQUESTED)

        job._progress_callback = on_progress
        paused = await job.run()
        self.assertEqual(paused.state, vod.STATE_PAUSED)
        self.assertEqual(paused.checkpoint_token, token["value"])
        self.assertFalse((self.tmp / "out.mp4").exists())

        resumed_job = _job(self.tmp, FakeRuntime(), input_path=self.sample)
        resumed = await resumed_job.run(resume=True, checkpoint_token=paused.checkpoint_token)
        self.assertEqual(resumed.state, vod.STATE_DONE)
        self.assertEqual(_video_frame_count(Path(resumed.output_path)), self.input_frames)

    async def test_checkpoint_contains_no_embeddings(self):
        values = np.linspace(0.1, 1.0, 512, dtype=np.float32)
        whitelist = vod.VodWhitelist("model:abc", (("entry-1", values),))
        job = vod.VodJob(
            job_id="job-wl",
            input_path=str(self.sample),
            output_path=str(self.tmp / "out.mp4"),
            work_dir=str(self.tmp / "work"),
            runtime=FakeRuntime(),
            adaface=None,
            config=vod.VodConfig(segment_frames=30),
            limits=vod.VodLimits(),
            whitelist=whitelist,
            tracker_factory=FakeTracker,
            tracker_config=Path("unused.yaml"),
            device="cpu",
        )
        await job.run()
        raw = (self.tmp / "work" / "checkpoint.json").read_text()
        checkpoint = json.loads(raw)
        self.assertIn("whitelist_fingerprint", checkpoint)
        self.assertEqual(checkpoint["embedding_model"], "model:abc")
        # Only the fingerprint and model name persist; no vectors and no "values" array.
        self.assertNotIn("values", raw)
        self.assertNotIn("entries", raw)
        normalized = (values / float(np.linalg.norm(values))).astype(np.float32)
        for component in normalized[:8].tolist():
            self.assertNotIn(repr(component), raw)

    async def test_resume_rejects_a_mismatched_token(self):
        runtime = FakeRuntime()
        token: dict[str, str] = {}
        job = _job(self.tmp, runtime, input_path=self.sample)

        def on_progress(snapshot: vod.VodProgressSnapshot, committed: bool) -> None:
            if committed and snapshot.segments_done == 1:
                token["value"] = snapshot.checkpoint_token
                job.request_pause()

        job._progress_callback = on_progress
        await job.run()

        resume_job = _job(self.tmp, FakeRuntime(), input_path=self.sample)
        with self.assertRaises(vod.VodError) as raised:
            await resume_job.run(resume=True, checkpoint_token="v1:1:deadbeefdeadbeef")
        self.assertEqual(raised.exception.code, "CHECKPOINT_MISMATCH")

    async def test_input_too_large(self):
        job = _job(
            self.tmp,
            FakeRuntime(),
            input_path=self.sample,
            limits=vod.VodLimits(max_input_bytes=16),
        )
        with self.assertRaises(vod.VodError) as raised:
            await job.run()
        self.assertEqual(raised.exception.code, "INPUT_TOO_LARGE")

    async def test_input_too_long(self):
        job = _job(
            self.tmp,
            FakeRuntime(),
            input_path=self.sample,
            limits=vod.VodLimits(max_duration_seconds=0.5),
        )
        with self.assertRaises(vod.VodError) as raised:
            await job.run()
        self.assertEqual(raised.exception.code, "INPUT_TOO_LONG")

    async def test_path_outside_root(self):
        root = self.tmp / "root"
        root.mkdir()
        job = _job(
            self.tmp,
            FakeRuntime(),
            input_path=self.sample,
            limits=vod.VodLimits(vod_root=root),
        )
        with self.assertRaises(vod.VodError) as raised:
            await job.run()
        self.assertEqual(raised.exception.code, "PATH_OUTSIDE_ROOT")

    async def test_output_exists_without_overwrite(self):
        (self.tmp / "out.mp4").write_bytes(b"existing")
        job = _job(self.tmp, FakeRuntime(), input_path=self.sample)
        with self.assertRaises(vod.VodError) as raised:
            await job.run()
        self.assertEqual(raised.exception.code, "OUTPUT_EXISTS")

    async def test_high_quality_is_a_precondition_error(self):
        with self.assertRaises(vod.VodPreconditionError):
            vod.VodJob(
                job_id="job-hi",
                input_path=str(self.sample),
                output_path=str(self.tmp / "out.mp4"),
                work_dir=str(self.tmp / "work"),
                runtime=FakeRuntime(),
                adaface=None,
                config=vod.VodConfig(quality=vod.QUALITY_HIGH),
                limits=vod.VodLimits(),
                whitelist=vod.VodWhitelist("", ()),
                tracker_factory=FakeTracker,
                tracker_config=Path("unused.yaml"),
                device="cpu",
            )


class VodProgressTests(unittest.TestCase):
    def test_estimated_total_never_completes_progress_before_finalization(self):
        snapshots = []
        job = _job(
            Path("unused"),
            FakeRuntime(),
            input_path=Path("unused.mp4"),
            progress_callback=lambda snapshot, _committed: snapshots.append(snapshot),
        )
        state = SimpleNamespace(frames_total=100, frames_total_estimated=True, segments=[])
        for frames in (0, 25, 99, 100, 120):
            job._emit_progress(frames, state, "token", committed=False)
        self.assertEqual(snapshots[1].fraction, 0.25)
        self.assertEqual([s.fraction for s in snapshots], sorted(s.fraction for s in snapshots))
        self.assertTrue(all(round(s.fraction * 100) < 100 for s in snapshots))
        self.assertEqual(snapshots[-1].frames_done, 120)
        self.assertEqual(snapshots[-1].frames_total, 100)
        self.assertTrue(snapshots[-1].frames_total_estimated)
        job._emit_progress(120, state, "token", committed=False, done=True)
        self.assertEqual(snapshots[-1].fraction, 1.0)


if __name__ == "__main__":
    unittest.main()
