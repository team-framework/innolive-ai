from __future__ import annotations

import asyncio
import threading
import unittest
from pathlib import Path

import numpy as np
import torch

from service.adaface_model import (
    _REFERENCE_LANDMARKS,
    AdaFaceConfig,
    AdaFaceRuntime,
    FaceCountError,
    FaceTooSmallError,
)


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

    async def test_official_flip_fusion_uses_pre_norm_features(self):
        class FixedModel(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.inputs = None

            def forward(self, inputs):
                self.inputs = inputs.detach().clone()
                embeddings = torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0]],
                    dtype=torch.float32,
                    device=inputs.device,
                )
                norms = torch.tensor(
                    [[2.0], [1.0]],
                    dtype=torch.float32,
                    device=inputs.device,
                )
                return embeddings, norms

        runtime = unavailable_runtime()
        model = FixedModel()
        runtime._model = model
        runtime._aligned_face = lambda *_args, **_kwargs: np.arange(
            112 * 112 * 3,
            dtype=np.uint8,
        ).reshape(112, 112, 3)
        try:
            embedding = runtime._embedding(np.zeros((64, 64, 3), dtype=np.uint8))

            self.assertEqual(tuple(model.inputs.shape), (2, 3, 112, 112))
            self.assertTrue(torch.equal(model.inputs[1], torch.flip(model.inputs[0], (2,))))
            self.assertTrue(
                np.allclose(
                    embedding,
                    np.asarray([2.0, 1.0], dtype=np.float32) / np.sqrt(5.0),
                )
            )
        finally:
            runtime.close()

    async def test_query_alignment_relaxes_score_and_size_but_rejects_multiple_faces(self):
        class FixedDetector:
            def __init__(self, faces: np.ndarray) -> None:
                self.faces = faces

            def setInputSize(self, _size) -> None:
                pass

            def detect(self, _image):
                return 1, self.faces.copy()

        def face(score: float, size: float = 25.0) -> np.ndarray:
            detected = np.zeros(15, dtype=np.float32)
            detected[2:4] = size
            detected[4:14] = _REFERENCE_LANDMARKS.reshape(-1)
            detected[14] = score
            return detected

        runtime = unavailable_runtime()
        image = np.zeros((112, 112, 3), dtype=np.uint8)
        try:
            runtime._detector = FixedDetector(np.asarray([face(0.65)]))
            aligned = runtime._aligned_face(
                image,
                score_threshold=0.6,
                min_face_size=24,
            )
            self.assertEqual(aligned.shape, (112, 112, 3))

            with self.assertRaises(FaceCountError):
                runtime._aligned_face(
                    image,
                    score_threshold=0.9,
                    min_face_size=40,
                )

            runtime._detector = FixedDetector(np.asarray([face(0.95)]))
            with self.assertRaises(FaceTooSmallError):
                runtime._aligned_face(
                    image,
                    score_threshold=0.9,
                    min_face_size=40,
                )

            runtime._detector = FixedDetector(np.asarray([face(0.7), face(0.8)]))
            with self.assertRaises(FaceCountError):
                runtime._aligned_face(
                    image,
                    score_threshold=0.6,
                    min_face_size=24,
                )
        finally:
            runtime.close()

    async def test_query_and_enrollment_workers_use_separate_quality_floors(self):
        runtime = unavailable_runtime()
        settings = []

        def embedding(_image, **quality):
            settings.append(quality)
            return np.asarray([1.0, 0.0], dtype=np.float32)

        runtime._embedding = embedding
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            runtime._queued_embedding(image, 0.0)
            runtime._queued_enrollment_embedding(image, 0.0)
        finally:
            runtime.close()

        self.assertEqual(
            settings,
            [
                {"score_threshold": 0.6, "min_face_size": 24},
                {"score_threshold": 0.9, "min_face_size": 40},
            ],
        )


if __name__ == "__main__":
    unittest.main()
