from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL = ROOT / "models" / "best.pt"
# Git LFS pointer files are ~130 bytes; a real checkpoint is tens of MB.
MODEL_AVAILABLE = MODEL.is_file() and MODEL.stat().st_size > 1_000_000


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


def _video_frames(path: Path) -> int:
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


def _audio_streams(path: Path) -> list[dict]:
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


@unittest.skipUnless(
    MODEL_AVAILABLE,
    "models/best.pt is unavailable (unresolved Git LFS pointer); CLI needs the real model",
)
class VodCliTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="vod-cli-"))
        self.input = self.tmp / "in.mp4"
        self.output = self.tmp / "out.mp4"
        self.work = self.tmp / "work"
        _generate_sample(self.input)
        self.input_frames = _video_frames(self.input)

    def _command(self, *extra: str) -> list[str]:
        return [
            sys.executable,
            "-m",
            "service.vod_cli",
            "--input",
            str(self.input),
            "--output",
            str(self.output),
            "--work-dir",
            str(self.work),
            "--device",
            "cpu",
            "--segment-frames",
            "15",
            "--warmup-frames",
            "5",
            *extra,
        ]

    def test_sigterm_pauses_then_resume_completes(self):
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        proc = subprocess.Popen(
            self._command(),
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        events: list[dict] = []
        signalled = False
        deadline = time.monotonic() + 180
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(event)
            if event.get("event") == "checkpoint" and not signalled:
                proc.send_signal(signal.SIGTERM)
                signalled = True
            if time.monotonic() > deadline:
                proc.kill()
                self.fail("CLI did not reach a checkpoint in time")
        exit_code = proc.wait(timeout=30)

        self.assertTrue(signalled, "no checkpoint event was emitted before SIGTERM")
        self.assertEqual(exit_code, 75)
        kinds = [event["event"] for event in events]
        self.assertIn("started", kinds)
        self.assertIn("paused", kinds)
        self.assertFalse(self.output.exists())

        resumed = subprocess.run(
            self._command("--resume"),
            cwd=str(ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=300,
        )
        self.assertEqual(resumed.returncode, 0)
        resumed_events = [
            json.loads(line) for line in resumed.stdout.splitlines() if line.strip().startswith("{")
        ]
        self.assertIn("done", [event["event"] for event in resumed_events])
        self.assertTrue(self.output.is_file())
        self.assertEqual(_video_frames(self.output), self.input_frames)
        audio = _audio_streams(self.output)
        self.assertEqual(len(audio), 1)
        self.assertEqual(audio[0]["codec_name"], "aac")


if __name__ == "__main__":
    unittest.main()
