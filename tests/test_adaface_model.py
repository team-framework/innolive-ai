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
    _square_face_crop,
)


def unavailable_runtime(*, queue_capacity: int = 4) -> AdaFaceRuntime:
    return AdaFaceRuntime(
        AdaFaceConfig(
            architecture="ir18",
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

            def forward(self, inputs, keypoints=None):
                self.assert_keypoints = keypoints
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
        runtime._prepared_face = lambda *_args, **_kwargs: (
            np.arange(112 * 112 * 3, dtype=np.uint8).reshape(112, 112, 3),
            None,
        )
        try:
            embedding = runtime._embedding(np.zeros((64, 64, 3), dtype=np.uint8))

            self.assertEqual(tuple(model.inputs.shape), (2, 3, 112, 112))
            self.assertIsNone(model.assert_keypoints)
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

    async def test_vit_preparation_uses_rgb_and_mirrors_landmarks(self):
        class FixedModel(torch.nn.Module):
            def forward(self, inputs, keypoints):
                self.inputs = inputs.detach().clone()
                self.keypoints = keypoints.detach().clone()
                embeddings = torch.tensor(
                    [[1.0, 0.0], [0.0, 1.0]],
                    dtype=torch.float32,
                    device=inputs.device,
                )
                norms = torch.ones((2, 1), dtype=torch.float32, device=inputs.device)
                return embeddings, norms

        runtime = unavailable_runtime()
        runtime.config = AdaFaceConfig(
            architecture="vit_base_kprpe",
            weights=Path("missing-adaface.ckpt"),
            detector=Path("missing-yunet.onnx"),
            device="cpu",
        )
        model = FixedModel()
        runtime._model = model
        prepared = np.zeros((112, 112, 3), dtype=np.uint8)
        prepared[..., 0] = 10
        prepared[..., 1] = 20
        prepared[..., 2] = 30
        keypoints = np.asarray(
            [[0.2, 0.3], [0.8, 0.3], [0.5, 0.5], [0.3, 0.8], [0.7, 0.8]],
            dtype=np.float32,
        )
        runtime._prepared_face = lambda *_args, **_kwargs: (prepared, keypoints)
        try:
            embedding = runtime._embedding(np.zeros((64, 64, 3), dtype=np.uint8))
        finally:
            runtime.close()

        self.assertTrue(np.allclose(embedding, np.asarray([1.0, 1.0]) / np.sqrt(2.0)))
        expected_rgb = torch.tensor([(30 / 127.5) - 1, (20 / 127.5) - 1, (10 / 127.5) - 1])
        self.assertTrue(torch.allclose(model.inputs[0, :, 0, 0], expected_rgb))
        self.assertTrue(torch.allclose(model.keypoints[0], torch.from_numpy(keypoints)))
        expected_mirror = keypoints[[1, 0, 2, 4, 3]].copy()
        expected_mirror[:, 0] = 1.0 - expected_mirror[:, 0]
        self.assertTrue(torch.allclose(model.keypoints[1], torch.from_numpy(expected_mirror)))

    async def test_vit_square_crop_maps_landmarks_into_normalized_crop(self):
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        face = np.zeros(15, dtype=np.float32)
        face[:4] = [100, 50, 80, 100]
        face[4:14] = np.asarray(
            [[120, 80], [160, 80], [140, 105], [125, 130], [155, 130]],
            dtype=np.float32,
        ).reshape(-1)

        cropped, keypoints = _square_face_crop(image, face)

        self.assertEqual(cropped.shape, (112, 112, 3))
        self.assertEqual(keypoints.shape, (5, 2))
        self.assertTrue(np.isfinite(keypoints).all())
        self.assertTrue(((keypoints >= 0) & (keypoints <= 1)).all())


if __name__ == "__main__":
    unittest.main()
