from __future__ import annotations

import asyncio
import json
import tempfile
import threading
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import grpc
import numpy as np
from test_grpc_server import (
    FaceRuntime,
    LoopbackServer,
    _collect,
    _request,
    _requests,
    _striped_jpeg,
)

from grpc_client import VideoFrame, VideoProcessorClient
from protos import ai_processor_pb2 as messages
from scripts.face_preset_client import run
from service.face_anonymization import FaceAnonymizationConfig, FaceAnonymizationRuntime
from service.face_metadata import AttributePrediction, FaceAttributes
from service.face_presets import PresetCatalog, StreamFaceIdentities
from service.inswapper import REFERENCE, alignment
from service.mosaic import mosaic_jpeg

ATTRS = FaceAttributes("female", "20s", "off", "none")
CHANGED = FaceAttributes("female", "20s", "on", "happy")


def face(track_id=1, **kwargs):
    return {
        "class_name": "face",
        "class_id": 0,
        "track_id": track_id,
        "bbox": [10.0, 2.0, 50.0, 34.0],
        "mask_polygon": [[10.0, 2.0], [50.0, 2.0], [50.0, 34.0], [10.0, 34.0]],
        **kwargs,
    }


class PresetTests(unittest.TestCase):
    def test_identity_stays_fixed_while_dynamic_attributes_change(self):
        stream = StreamFaceIdentities(choose=lambda _: 3)
        first, _ = stream.observe(1, ATTRS, 1)
        noisy = FaceAttributes("male", "over40s", "on", "happy")
        current, variant = stream.observe(1, noisy, 2)
        self.assertIs(first, current)
        self.assertEqual(current.key, "female/20s/4")
        self.assertEqual(variant, CHANGED)

    def test_five_slots_are_available_and_streams_are_independent(self):
        for index in range(5):
            stream = StreamFaceIdentities(choose=lambda _, index=index: index)
            identity, _ = stream.observe(1, ATTRS, 1)
            self.assertEqual(identity.identity, index + 1)
        one = StreamFaceIdentities(choose=lambda _: 0)
        two = StreamFaceIdentities(choose=lambda _: 4)
        self.assertNotEqual(one.observe(1, ATTRS, 1)[0].key, two.observe(1, ATTRS, 1)[0].key)

    def test_occlusion_retention_expiration_and_cleanup(self):
        stream = StreamFaceIdentities(retention_frames=30, choose=lambda _: 0)
        first, _ = stream.observe(1, ATTRS, 1)
        stream.cache_metadata(1, AttributePrediction(ATTRS, {}), 1)
        stream.expire(set(), 31)
        self.assertIn(1, stream.metadata)
        self.assertIs(first, stream.observe(1, CHANGED, 32)[0])
        stream.expire({1}, 32)
        stream.expire(set(), 63)
        self.assertNotIn(1, stream.tracks)
        self.assertNotIn(1, stream.metadata)
        stream.observe(2, ATTRS, 64)
        stream.cache_metadata(2, None, 64)
        stream.clear()
        self.assertEqual(stream.tracks, {})
        self.assertEqual(stream.metadata, {})

    def test_failed_metadata_without_identity_expires(self):
        stream = StreamFaceIdentities()
        stream.cache_metadata(1, None, 1)
        stream.expire(set(), 32)
        self.assertEqual(stream.metadata, {})

    def test_catalog_allows_partial_sets_but_no_duplicate_or_unsafe_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "manifest.json"
            entry = {**asdict(ATTRS), "identity": 1, "image": "pending.png"}
            manifest.write_text(json.dumps({"schema_version": 1, "presets": [entry]}))
            catalog = PresetCatalog(manifest)
            self.assertEqual(catalog.match(ATTRS, 1).key, "female/20s/1/off/none")
            self.assertIsNone(catalog.match(CHANGED, 1))
            for invalid in (
                [entry, entry],
                [{**entry, "identity": 0}],
                [{**entry, "image": "../escape.png"}],
                [{**entry, "glasses": "maybe"}],
            ):
                manifest.write_text(json.dumps({"schema_version": 1, "presets": invalid}))
                with self.assertRaises(ValueError):
                    PresetCatalog(manifest)

    def test_inswapper_alignment_uses_the_128_horizontal_margin(self):
        matrix = alignment(REFERENCE.copy(), 128)
        np.testing.assert_allclose(matrix, [[1, 0, 8], [0, 1, 0]], atol=1e-4)


class AnonymizationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.manifest = self.root / "manifest.json"
        self.manifest.write_text(json.dumps({"schema_version": 1, "presets": []}))
        self.runtime = FaceAnonymizationRuntime(
            FaceAnonymizationConfig(preset_manifest=self.manifest)
        )
        self.runtime.extractor = Mock()
        self.runtime.extractor.predict.return_value = [
            AttributePrediction(ATTRS, {k: 0.9 for k in asdict(ATTRS)})
        ]
        self.runtime.catalog = PresetCatalog(self.manifest)
        self.stream = StreamFaceIdentities(choose=lambda _: 0)
        self.image = np.zeros((36, 64, 3), np.uint8)
        self.image[:, ::2] = 255

    def process(self, objects, *, frame=1, mode=1, render=True, pix_fmt=""):
        return self.runtime.process(
            self.image,
            objects,
            self.stream,
            frame,
            mode,
            render=render,
            pix_fmt=pix_fmt,
            blur_radius=24.0,
            pixel_size=1,
            max_bytes=4 * 1024 * 1024,
        )

    def presets(self):
        path = self.root / "source.png"
        cv2.imwrite(str(path), self.image)
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "presets": [
                        {**asdict(attrs), "identity": 1, "image": path.name}
                        for attrs in (ATTRS, CHANGED)
                    ],
                }
            )
        )
        self.runtime.catalog = PresetCatalog(self.manifest)

    def test_missing_preset_returns_attributes_stable_identity_and_exact_blur(self):
        objects = [face()]
        payload = self.process(objects)
        info = objects[0]["anonymization"]
        self.assertEqual(info["reason"], "preset_missing")
        self.assertEqual(info["attributes"], asdict(ATTRS))
        self.assertEqual(info["identity_key"], "female/20s/1")
        self.assertEqual(payload, mosaic_jpeg(self.image, [face()]))
        self.assertIsNone(self.runtime.renderer)

    def test_dynamic_variant_refreshes_every_30_frames_without_identity_change(self):
        self.presets()
        self.runtime.renderer = Mock()
        self.runtime.renderer.swap.side_effect = lambda image, *_: np.full_like(image, 80)
        first = [face()]
        self.process(first)
        self.runtime.extractor.predict.return_value = [AttributePrediction(CHANGED, {})]
        for frame in range(2, 31):
            objects = [face()]
            self.process(objects, frame=frame)
            info = objects[0]["anonymization"]
            self.assertEqual(info["attributes"], asdict(ATTRS))
            self.assertEqual(info["confidence"]["exp"], 0.9)
            self.assertEqual(info["preset_key"], "female/20s/1/off/none")
            self.assertEqual(info["status"], "swapped")
        self.assertEqual(self.runtime.extractor.predict.call_count, 1)
        second = [face()]
        self.process(second, frame=31)
        info1, info2 = first[0]["anonymization"], second[0]["anonymization"]
        self.assertEqual(info1["identity_key"], info2["identity_key"])
        self.assertEqual(info1["preset_key"], "female/20s/1/off/none")
        self.assertEqual(info2["preset_key"], "female/20s/1/on/happy")
        self.assertEqual(info2["status"], "swapped")
        self.assertEqual(self.runtime.extractor.predict.call_count, 2)
        for frame in range(32, 61):
            self.process([face()], frame=frame)
        self.assertEqual(self.runtime.extractor.predict.call_count, 2)
        self.process([face()], frame=61)
        self.assertEqual(self.runtime.extractor.predict.call_count, 3)
        self.assertEqual(self.runtime.renderer.swap.call_count, 61)

    def test_tracks_refresh_on_independent_schedules_and_batch_only_due_faces(self):
        self.process([face(1)], mode=2)
        self.runtime.extractor.predict.return_value = [AttributePrediction(CHANGED, {})]
        objects = [face(1), face(2)]
        self.process(objects, frame=10, mode=2)
        self.assertEqual(objects[0]["anonymization"]["attributes"], asdict(ATTRS))
        self.assertEqual(objects[1]["anonymization"]["attributes"], asdict(CHANGED))
        self.assertEqual(len(self.runtime.extractor.predict.call_args.args[0]), 1)
        self.process([face(1), face(2)], frame=30, mode=2)
        self.assertEqual(self.runtime.extractor.predict.call_count, 2)
        self.process([face(1), face(2)], frame=31, mode=2)
        self.assertEqual(self.runtime.extractor.predict.call_count, 3)
        self.assertEqual(len(self.runtime.extractor.predict.call_args.args[0]), 1)
        self.process([face(1), face(2)], frame=39, mode=2)
        self.assertEqual(self.runtime.extractor.predict.call_count, 3)
        self.process([face(1), face(2)], frame=40, mode=2)
        self.assertEqual(self.runtime.extractor.predict.call_count, 4)
        self.assertEqual(len(self.runtime.extractor.predict.call_args.args[0]), 1)

    def test_new_tracks_due_together_are_batched(self):
        self.runtime.extractor.predict.return_value = [
            AttributePrediction(ATTRS, {}),
            AttributePrediction(CHANGED, {}),
        ]
        objects = [face(1), face(2)]
        self.process(objects, mode=2)
        self.runtime.extractor.predict.assert_called_once()
        self.assertEqual(len(self.runtime.extractor.predict.call_args.args[0]), 2)
        self.assertEqual(objects[0]["anonymization"]["attributes"], asdict(ATTRS))
        self.assertEqual(objects[1]["anonymization"]["attributes"], asdict(CHANGED))
        self.process([face(1), face(2)], frame=2, mode=2)
        self.runtime.extractor.predict.assert_called_once()

    def test_due_held_face_refreshes_when_fresh_detection_returns(self):
        self.process([face()], mode=2)
        self.runtime.extractor.predict.return_value = [AttributePrediction(CHANGED, {})]
        held = [face(held=True)]
        self.process(held, frame=31, mode=2)
        self.runtime.extractor.predict.assert_called_once()
        self.assertEqual(held[0]["anonymization"]["reason"], "held_face")
        objects = [face()]
        self.process(objects, frame=32, mode=2)
        self.assertEqual(self.runtime.extractor.predict.call_count, 2)
        self.assertEqual(objects[0]["anonymization"]["attributes"], asdict(CHANGED))

    def test_failed_refresh_blurs_until_next_attempt_and_preserves_identity(self):
        self.presets()
        self.process([face()], mode=2)
        self.runtime.extractor.predict.side_effect = ValueError("invalid logits")
        objects = [face()]
        with self.assertLogs("innolive.face_swap", level="WARNING"):
            result = self.process(objects, frame=31)
        self.assertEqual(result, mosaic_jpeg(self.image, [face()]))
        self.assertEqual(objects[0]["anonymization"]["identity_key"], "female/20s/1")
        self.assertNotIn("attributes", objects[0]["anonymization"])
        self.runtime.extractor.predict.side_effect = None
        self.runtime.extractor.predict.return_value = [AttributePrediction(CHANGED, {})]
        for frame in range(32, 61):
            objects = [face()]
            self.process(objects, frame=frame)
            self.assertEqual(objects[0]["anonymization"]["reason"], "metadata_unavailable")
        self.assertEqual(self.runtime.extractor.predict.call_count, 2)
        objects = [face()]
        self.process(objects, frame=61, mode=2)
        self.assertEqual(self.runtime.extractor.predict.call_count, 3)
        self.assertEqual(objects[0]["anonymization"]["attributes"], asdict(CHANGED))
        self.assertEqual(objects[0]["anonymization"]["identity_key"], "female/20s/1")

    def test_initial_failure_waits_30_frames_without_allocating_identity(self):
        self.runtime.extractor.predict.side_effect = ValueError("invalid logits")
        with self.assertLogs("innolive.face_swap", level="WARNING"):
            self.process([face()])
        self.assertEqual(self.stream.tracks, {})
        self.runtime.extractor.predict.side_effect = None
        self.process([face()], frame=30)
        self.runtime.extractor.predict.assert_called_once()
        self.process([face()], frame=31)
        self.assertEqual(self.runtime.extractor.predict.call_count, 2)
        self.assertIn(1, self.stream.tracks)

    def test_expired_track_loses_cached_metadata_and_refreshes_immediately(self):
        self.process([face()], mode=2)
        self.process([], frame=32, mode=2)
        self.assertEqual(self.stream.tracks, {})
        self.assertEqual(self.stream.metadata, {})
        self.runtime.extractor.predict.return_value = [AttributePrediction(CHANGED, {})]
        objects = [face()]
        self.process(objects, frame=33, mode=2)
        self.assertEqual(self.runtime.extractor.predict.call_count, 2)
        self.assertEqual(objects[0]["anonymization"]["attributes"], asdict(CHANGED))

    def test_missing_dynamic_variant_never_substitutes_another_identity(self):
        self.presets()
        self.process([face()], mode=2)
        self.runtime.extractor.predict.return_value = [
            AttributePrediction(FaceAttributes("female", "20s", "on", "anger"), {})
        ]
        objects = [face()]
        self.process(objects, frame=31)
        self.assertEqual(objects[0]["anonymization"]["identity_key"], "female/20s/1")
        self.assertEqual(objects[0]["anonymization"]["reason"], "preset_missing")

    def test_metadata_preview_and_metadata_wire_output_do_not_load_swapper(self):
        self.presets()
        objects = [face()]
        self.assertEqual(self.process(objects, mode=2), mosaic_jpeg(self.image, [face()]))
        self.assertEqual(objects[0]["anonymization"]["status"], "metadata_only")
        self.assertEqual(self.process([face()], render=False), b"")
        self.assertIsNone(self.runtime.renderer)

    def test_swap_failure_and_invalid_result_keep_original_blur(self):
        self.presets()
        self.runtime.renderer = Mock()
        for output in (RuntimeError("broken model"), np.zeros((1, 1, 3), np.uint8)):
            with self.subTest(output=type(output).__name__):
                self.runtime.renderer.swap.side_effect = (
                    output if isinstance(output, Exception) else None
                )
                self.runtime.renderer.swap.return_value = output
                objects = [face()]
                with self.assertLogs("innolive.face_swap", level="WARNING"):
                    result = self.process(objects)
                self.assertEqual(result, mosaic_jpeg(self.image, [face()]))
                self.assertEqual(objects[0]["anonymization"]["reason"], "swap_failed")

    def test_whitelist_number_plate_and_held_faces_skip_attribute_inference(self):
        objects = [
            face(1, whitelisted=True),
            face(2, class_name="number_plate", class_id=1),
            face(3, held=True),
        ]
        self.process(objects)
        self.runtime.extractor.predict.assert_not_called()
        self.assertNotIn("anonymization", objects[0])
        self.assertNotIn("anonymization", objects[1])
        self.assertEqual(objects[2]["anonymization"]["reason"], "held_face")

    def test_untracked_faces_extract_attributes_but_never_swap(self):
        objects = [face(None)]
        self.process(objects)
        self.assertEqual(objects[0]["anonymization"]["reason"], "untracked_face")
        self.assertEqual(self.stream.tracks, {})

    def test_metadata_failure_keeps_blur_and_raw_output_remains_yuv420p(self):
        self.runtime.extractor.predict.side_effect = ValueError("invalid logits")
        with self.assertLogs("innolive.face_swap", level="WARNING"):
            result = self.process([face()], pix_fmt="yuv420p")
        self.assertEqual(len(result), 64 * 36 * 3 // 2)


class GrpcAnonymizationTests(unittest.IsolatedAsyncioTestCase):
    async def test_refresh_uses_rpc_frame_sequence_and_cache_is_not_shared_between_rpcs(self):
        async with LoopbackServer(FaceRuntime()) as server:
            runtime = server.bundle.servicer.face_anonymization
            runtime.extractor = Mock(
                predict=Mock(
                    side_effect=[
                        [AttributePrediction(ATTRS, {"exp": 0.9})],
                        [AttributePrediction(CHANGED, {"exp": 0.8})],
                        [AttributePrediction(ATTRS, {"exp": 0.7})],
                    ]
                )
            )
            runtime.catalog = Mock(match=Mock(return_value=None))
            requests = [_request(frame_id=10000 - i * 100) for i in range(31)]
            for request in requests:
                request.anonymization_mode = messages.FACE_ANONYMIZATION_MODE_FACE_METADATA
            responses = await _collect(server.stub.ProcessVideo(_requests(*requests)))
            self.assertEqual(runtime.extractor.predict.call_count, 2)
            for response in responses[:30]:
                info = response.faces[0].anonymization
                self.assertEqual(info.attributes.exp, "none")
                self.assertAlmostEqual(info.attributes.confidence["exp"], 0.9)
                self.assertEqual(
                    info.identity_key, responses[0].faces[0].anonymization.identity_key
                )
            self.assertEqual(responses[30].faces[0].anonymization.attributes.exp, "happy")
            (reconnected,) = await _collect(server.stub.ProcessVideo(_requests(requests[0])))
            self.assertEqual(runtime.extractor.predict.call_count, 3)
            self.assertAlmostEqual(
                reconnected.faces[0].anonymization.attributes.confidence["exp"], 0.7
            )

    async def test_default_request_never_loads_metadata_or_renderer(self):
        async with LoopbackServer(FaceRuntime()) as server:
            runtime = server.bundle.servicer.face_anonymization
            with patch.object(
                runtime, "process", side_effect=AssertionError("experimental invoked")
            ):
                responses = await _collect(
                    server.stub.ProcessVideo(_requests(_request(data=_striped_jpeg())))
                )
            self.assertEqual(responses[0].status_message, "success")
            self.assertFalse(responses[0].faces[0].HasField("anonymization"))
            self.assertIsNone(runtime.extractor)
            self.assertIsNone(runtime.renderer)

    async def test_sdk_transmits_opt_in_and_roundtrips_attributes(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "empty.json"
            manifest.write_text('{"schema_version": 1, "presets": []}')
            async with LoopbackServer(FaceRuntime()) as server:
                runtime = server.bundle.servicer.face_anonymization
                runtime.extractor = Mock(
                    predict=Mock(return_value=[AttributePrediction(ATTRS, {"exp": 0.9})])
                )
                runtime.catalog = PresetCatalog(manifest)
                async with VideoProcessorClient(f"127.0.0.1:{server.bundle.bound_port}") as client:
                    results = [
                        item
                        async for item in client.process_video(
                            [
                                VideoFrame(_striped_jpeg(), i, i, anonymization_mode="face_swap")
                                for i in (1, 2)
                            ],
                            session_id="swap-sdk",
                            window=1,
                        )
                    ]
                infos = [r.response.faces[0].anonymization for r in results]
                self.assertEqual(infos[0].attributes.gender, "female")
                self.assertAlmostEqual(infos[0].attributes.confidence["exp"], 0.9)
                self.assertEqual(infos[0].identity_key, infos[1].identity_key)
                self.assertEqual(infos[1].fallback_reason, "preset_missing")
                self.assertTrue(results[0].processed_jpeg.startswith(b"\xff\xd8"))

    async def test_sdk_rejects_legacy_server_ignoring_experimental_mode(self):
        from grpc_client import VideoProtocolError

        async with LoopbackServer(FaceRuntime()) as server:
            with patch.object(
                server.bundle.servicer.face_anonymization, "process", return_value=_striped_jpeg()
            ):
                async with VideoProcessorClient(f"127.0.0.1:{server.bundle.bound_port}") as client:
                    with self.assertRaisesRegex(VideoProtocolError, "update the server"):
                        async for _ in client.process_video(
                            [VideoFrame(_striped_jpeg(), 1, 1, anonymization_mode="face_swap")],
                            session_id="legacy-check",
                        ):
                            pass

    async def test_unknown_mode_and_midstream_changes_are_rejected(self):
        for modes in ((99,), (0, 1)):
            async with LoopbackServer(FaceRuntime()) as server:
                requests = [_request(frame_id=i) for i in range(len(modes))]
                for request, mode in zip(requests, modes, strict=True):
                    request.anonymization_mode = mode
                with self.assertRaises(grpc.aio.AioRpcError) as error:
                    await _collect(server.stub.ProcessVideo(_requests(*requests)))
                self.assertEqual(error.exception.code(), grpc.StatusCode.INVALID_ARGUMENT)

    async def test_metadata_only_wire_response_has_attributes_without_image(self):
        async with LoopbackServer(FaceRuntime()) as server:
            runtime = server.bundle.servicer.face_anonymization
            runtime.extractor = Mock(predict=Mock(return_value=[AttributePrediction(ATTRS, {})]))
            runtime.catalog = Mock(match=Mock(return_value=None))
            request = _request(output_mode=messages.VIDEO_OUTPUT_MODE_METADATA_ONLY)
            request.anonymization_mode = messages.FACE_ANONYMIZATION_MODE_FACE_METADATA
            (response,) = await _collect(server.stub.ProcessVideo(_requests(request)))
            self.assertEqual(response.data, b"")
            self.assertEqual(response.faces[0].anonymization.attributes.age, "20s")
            self.assertIsNone(runtime.renderer)

    async def test_test_client_headless_uses_actual_loopback_and_saves_results(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "input.jpg"
            image.write_bytes(_striped_jpeg())
            output = Path(directory) / "results"
            async with LoopbackServer(FaceRuntime()) as server:
                args = SimpleNamespace(
                    image=image,
                    video=None,
                    camera=None,
                    headless=True,
                    frames=2,
                    long_edge=640,
                    root_cert=None,
                    output_dir=output,
                    mode="blur",
                    target=f"127.0.0.1:{server.bundle.bound_port}",
                    connect_timeout=2.0,
                    timeout=5.0,
                    session_id="client-test",
                )
                with patch("builtins.print"):
                    self.assertEqual(await run(args), 2)
            records = [
                json.loads(line) for line in (output / "metadata.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["requested_mode"], "blur")
            self.assertTrue((output / "last.jpg").read_bytes().startswith(b"\xff\xd8"))

    async def test_cancellation_waits_for_composition_before_releasing_rpc_state(self):
        async with LoopbackServer(FaceRuntime()) as server:
            entered, release, settled = threading.Event(), threading.Event(), threading.Event()

            def compose():
                entered.set()
                release.wait(3)
                settled.set()
                return b"done"

            task = asyncio.create_task(server.bundle.servicer._compose_frame(compose))
            self.assertTrue(await asyncio.to_thread(entered.wait, 1))
            task.cancel()
            await asyncio.sleep(0.02)
            self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertTrue(settled.is_set())
