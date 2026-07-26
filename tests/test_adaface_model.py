from __future__ import annotations

import asyncio
import threading
import unittest
from pathlib import Path

import numpy as np

from service.adaface_model import AdaFaceConfig, AdaFaceRuntime


def unavailable_runtime(*, queue_capacity: int = 4) -> AdaFaceRuntime:
    return AdaFaceRuntime(
        AdaFaceConfig(
            weights=Path("missing-adaface.ckpt"),
            detector=Path("missing-yunet.onnx"),
            device="cpu",
            queue_capacity=queue_capacity,
        ),
        fallback_device="cpu",
    )


class AdaFaceRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_missing_model_assets_leave_runtime_unavailable(self):
        runtime = unavailable_runtime()
        try:
            self.assertFalse(runtime.ready)
            self.assertIn("not found", runtime.load_error)
        finally:
            runtime.close()

    async def test_one_owner_cannot_fill_the_global_queue(self):
        runtime = unavailable_runtime(queue_capacity=4)
        gate = threading.Event()

        def embedding(_image, _submitted_at):
            if not gate.wait(timeout=1.0):
                raise TimeoutError("test worker gate timed out")
            return np.asarray([1.0, 0.0], dtype=np.float32)

        runtime._queued_embedding = embedding
        runtime.ready = True
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            first = runtime.submit(image, owner="session-a")
            second = runtime.submit(image, owner="session-a")
            rejected = runtime.submit(image, owner="session-a")
            other = runtime.submit(image, owner="session-b")
            self.assertIsNotNone(first)
            self.assertIsNotNone(second)
            self.assertIsNone(rejected)
            self.assertIsNotNone(other)

            gate.set()
            await asyncio.gather(first, second, other)
            health = runtime.health()
            self.assertEqual(health["owner_capacity"], 2)
            self.assertEqual(health["queue_overflow"], 1)
            self.assertEqual(health["inflight"], 0)
        finally:
            gate.set()
            runtime.close()


if __name__ == "__main__":
    unittest.main()
