from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

import numpy as np

from service import vod


class VodProcessTests(unittest.TestCase):
    def child(self, program):
        popen = subprocess.Popen

        def launch(_command, **kwargs):
            return popen([sys.executable, "-c", program], **kwargs)

        return patch("service.vod.subprocess.Popen", side_effect=launch)

    def finish_with_deadline(self, worker, callback):
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            return pool.submit(callback).result(timeout=3)
        finally:
            # Also release a blocked pipe if this regression test runs on the old code.
            if worker._process.poll() is None:
                worker._process.kill()
                worker._process.wait(timeout=2)
            pool.shutdown(wait=True)
            worker.close()

    def test_decoder_drains_large_error_output_and_closes_handles(self):
        program = "import sys; sys.stderr.write('x' * 262144); sys.stdout.buffer.write(bytes(12))"
        with self.child(program):
            decoder = vod._FfmpegDecoder("ffmpeg", Path("unused.mp4"), 2, 2)

        def consume():
            self.assertEqual(decoder.read().shape, (2, 2, 3))
            self.assertIsNone(decoder.read())
            decoder.finish()
            self.assertTrue(decoder._stderr.closed)
            self.assertTrue(decoder._process.stdout.closed)

        self.finish_with_deadline(decoder, consume)

    def test_encoder_closes_handles_on_success_and_failure(self):
        for exit_code in (0, 1):
            with self.subTest(exit_code=exit_code):
                program = (
                    "import sys; sys.stdin.buffer.read(); "
                    "sys.stderr.write('x' * 262144 + 'encoder failure'); "
                    f"sys.exit({exit_code})"
                )
                with self.child(program):
                    encoder = vod._FfmpegSegmentEncoder(
                        "ffmpeg", Path("unused.mp4"), 2, 2, 30, vod.VodConfig(), "libx264"
                    )
                encoder.write(np.zeros((2, 2, 3), dtype=np.uint8))

                def finish(encoder=encoder, exit_code=exit_code):
                    if exit_code:
                        with self.assertRaises(vod.VodError) as caught:
                            encoder.finish()
                        self.assertEqual(caught.exception.code, "ENCODER_FAILED")
                        self.assertIn("encoder failure", str(caught.exception))
                    else:
                        encoder.finish()
                    self.assertTrue(encoder._stderr.closed)
                    self.assertTrue(encoder._process.stdin.closed)

                self.finish_with_deadline(encoder, finish)

    def test_spawn_failure_closes_error_file(self):
        with tempfile.TemporaryFile() as stderr:
            with (
                patch("service.vod.tempfile.TemporaryFile", return_value=stderr),
                patch("service.vod.subprocess.Popen", side_effect=OSError("cannot spawn")),
                self.assertRaises(OSError),
            ):
                vod._spawn_ffmpeg(["ffmpeg"])
            self.assertTrue(stderr.closed)

    def test_probe_timeout_keeps_input_error_contract(self):
        with (
            patch("service.vod._require_binary", return_value="ffprobe"),
            patch(
                "service.vod.subprocess.run",
                side_effect=subprocess.TimeoutExpired("ffprobe", 30),
            ) as run,
            self.assertRaises(vod.VodError) as caught,
        ):
            vod.probe_input(Path("unused.mp4"))
        self.assertEqual(caught.exception.code, "INPUT_UNREADABLE")
        self.assertEqual(run.call_args.kwargs["timeout"], vod.FFPROBE_TIMEOUT_SECONDS)


if __name__ == "__main__":
    unittest.main()
