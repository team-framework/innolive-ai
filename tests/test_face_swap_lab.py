from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "experiments" / "face_swap_lab.py"
SPEC = importlib.util.spec_from_file_location("face_swap_lab", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("could not load face_swap_lab")
face_swap_lab = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = face_swap_lab
SPEC.loader.exec_module(face_swap_lab)


class FaceSwapLabTests(unittest.TestCase):
    def test_percentile_interpolates(self):
        self.assertEqual(face_swap_lab.percentile([10.0, 20.0, 30.0], 50), 20.0)
        self.assertEqual(face_swap_lab.percentile([10.0, 20.0, 30.0], 95), 29.0)
        self.assertEqual(face_swap_lab.percentile([], 95), 0.0)

    def test_largest_face_uses_area(self):
        self.assertEqual(
            face_swap_lab.largest_face(((0, 0, 10, 20), (4, 5, 30, 10))),
            (4, 5, 30, 10),
        )
        self.assertIsNone(face_swap_lab.largest_face(()))

    def test_load_stats_normalizes_per_session_latency(self):
        stats = face_swap_lab.LoadStats.create()
        stats.add(80.0, multiplier=4, faces=1)
        summary = stats.summary(multiplier=4)
        self.assertIn("synthetic aggregate 12.5 fps", summary)
        self.assertIn("per-session 3.12 fps", summary)
        self.assertIn("one session p50 20.0 ms", summary)

    def test_preview_caption_states_load_method(self):
        caption = face_swap_lab.preview_caption("test model", 16)
        self.assertIn("16 independent synthetic session", caption)


if __name__ == "__main__":
    unittest.main()
