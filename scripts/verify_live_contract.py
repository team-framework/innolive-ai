#!/usr/bin/env python3
"""Check an explicitly selected live worker's gRPC contract using temporary sessions.

Requires a JPEG containing exactly one clearly visible face. Does not compare
model-dependent scores or coordinates. All sessions created here are deleted.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
from pathlib import Path

import cv2
import grpc
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from grpc_client import VideoFrame, VideoProcessorClient
from protos import ai_processor_pb2 as pb
from protos import ai_processor_pb2_grpc as rpc


async def verify(target, image_path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError("unreadable JPEG")
    image = cv2.resize(image, (640, 480))
    jpeg = cv2.imencode(".jpg", image)[1].tobytes()
    raw = cv2.cvtColor(image, cv2.COLOR_BGR2YUV_I420).tobytes()
    sessions, checks, trace = [], [], []
    async with grpc.aio.insecure_channel(target) as channel:
        await asyncio.wait_for(channel.channel_ready(), 10)
        stub = rpc.AiProcessorStub(channel)

        async def create():
            r = await stub.CreateSession(pb.CreateSessionRequest(), timeout=10)
            sessions.append(r.session_id)
            assert r.entry_count == 0 and r.active_stream_count == 0
            return r.session_id

        async def stream(items):
            async def inputs():
                for item in items:
                    yield item

            return [r async for r in stub.ProcessVideo(inputs(), timeout=60)]

        async def expect_error(method, request, code):
            try:
                await getattr(stub, method)(request, timeout=10)
            except grpc.aio.AioRpcError as e:
                assert e.code().name == code, (method, e.code(), e.details())
                trace.append([method, code, e.details()])
            else:
                raise AssertionError(f"{method} accepted invalid input")

        def check_frame(r, source, metadata=False):
            assert (r.frame_id, r.timestamp) == (source.frame_id, source.timestamp)
            assert r.status_message == "success" and not r.error_code
            assert (r.width, r.height) == (640, 480)
            assert not r.mosaic_jpeg
            assert r.timing.inference_batch_size == 1
            assert math.isfinite(r.timing.server_total_ms) and r.timing.server_total_ms > 0
            if metadata:
                assert not r.data
            elif source.pix_fmt == "yuv420p":
                assert len(r.data) == 640 * 480 * 3 // 2
                decoded = cv2.cvtColor(
                    np.frombuffer(r.data, np.uint8).reshape(720, 640), cv2.COLOR_YUV2BGR_I420
                )
                assert decoded.shape == image.shape
            else:
                decoded = cv2.imdecode(np.frombuffer(r.data, np.uint8), cv2.IMREAD_COLOR)
                assert decoded is not None and decoded.shape == image.shape
            for face in r.faces:
                assert face.class_name in {"face", "number_plate"}
                assert face.HasField("track_id") and len(face.polygon) <= 64
                assert all(math.isfinite(p.x) and math.isfinite(p.y) for p in face.polygon)

        try:
            sid = await create()
            for mode in (0, 1, 2):
                for output in (0, 1, 2):
                    for pixels in (False, True):
                        items = [
                            pb.VideoChunk(
                                data=raw if pixels else jpeg,
                                width=640 if pixels else 0,
                                height=480 if pixels else 0,
                                pix_fmt="yuv420p" if pixels else "",
                                session_id=sid,
                                frame_id=0,
                                timestamp=100 + i,
                                anonymization_mode=mode,
                                output_mode=output,
                                mosaic_config=pb.MosaicConfig(
                                    mask_margin=i * 8, blur_radius=24, pixel_size=1
                                ),
                            )
                            for i in range(3)
                        ]
                        result = await stream(items)
                        assert len(result) == len(items)
                        for r, src in zip(result, items, strict=True):
                            check_frame(r, src, output == 1)
                        assert [r.stats.tracker_frame for r in result] == [1, 2, 3]
                        assert len(result[-1].faces) == 1, (
                            mode,
                            output,
                            pixels,
                            len(result[-1].faces),
                        )
                        if mode and output != 1:
                            expected = "swapped" if mode == 1 else "metadata_only"
                            assert result[-1].faces[0].anonymization.status == expected
                        checks.append(f"mode={mode},output={output},raw={pixels}")
            invalid = [
                ({"data": b"invalid"}, "DECODE_FAILED"),
                ({"batch_size": 2}, "INVALID_BATCH_SIZE"),
                (
                    {"mosaic_config": pb.MosaicConfig(blur_radius=float("nan"))},
                    "MOSAIC_CONFIG_INVALID",
                ),
                ({"mosaic_config": pb.MosaicConfig(mask_margin=65)}, "MOSAIC_CONFIG_INVALID"),
            ]
            for overrides, expected in invalid:
                fields = dict(data=jpeg, session_id=sid, frame_id=-4, timestamp=-9)
                fields.update(overrides)
                good = pb.VideoChunk(data=jpeg, session_id=sid, frame_id=2, timestamp=3)
                response = await stream([pb.VideoChunk(**fields), good])
                assert len(response) == 2
                assert response[0].error_code == expected and not response[0].data
                assert (response[0].frame_id, response[0].timestamp) == (-4, -9)
                check_frame(response[1], good)
                trace.append(["ProcessVideo", expected])
            for overrides in (
                {"session_id": " "},
                {"output_mode": 999},
                {"anonymization_mode": 999},
            ):
                fields = dict(data=jpeg, session_id=sid)
                fields.update(overrides)
                try:
                    await stream([pb.VideoChunk(**fields)])
                except grpc.aio.AioRpcError as e:
                    assert e.code() == grpc.StatusCode.INVALID_ARGUMENT
                    trace.append(["ProcessVideo", e.code().name, e.details()])
                else:
                    raise AssertionError("invalid stream accepted")
            for window in (1, 2, 5):
                async with VideoProcessorClient(target) as client:

                    async def frames():
                        for i in range(20):
                            yield VideoFrame(data=jpeg, frame_id=i, timestamp=i + 10)

                    results = [
                        r
                        async for r in client.process_video(
                            frames(), session_id=sid, window=window, timeout=60
                        )
                    ]
                    assert [r.response.frame_id for r in results] == list(range(20))
                    assert client.max_inflight_observed <= window
                    checks.append(f"python-client-window={window}")
            await expect_error(
                "GetWhitelistStatus",
                pb.GetWhitelistStatusRequest(session_id=" "),
                "INVALID_ARGUMENT",
            )
            await expect_error(
                "AddWhitelist", pb.FaceData(session_id=sid, data=b"bad"), "INVALID_ARGUMENT"
            )
            added = await stub.AddWhitelist(pb.FaceData(session_id=sid, data=jpeg), timeout=30)
            assert added.entry_count == 1
            exported = await stub.ExportWhitelist(
                pb.ExportWhitelistRequest(session_id=sid), timeout=10
            )
            replica = await create()
            imported = await stub.ImportWhitelist(
                pb.ImportWhitelistRequest(
                    session_id=replica,
                    embedding_model=exported.embedding_model,
                    entries=exported.entries,
                ),
                timeout=10,
            )
            assert imported.entry_count == 1
            status = await stub.GetWhitelistStatus(
                pb.GetWhitelistStatusRequest(session_id=replica), timeout=10
            )
            assert list(status.entry_ids) == [added.entry_id]
            removed = await stub.DeleteWhitelist(
                pb.DeleteWhitelistRequest(session_id=sid, entry_id=added.entry_id), timeout=10
            )
            assert removed.entry_count == 0
            checks.append("whitelist-add-export-import-delete")
            # Cancel an RPC that has acquired its tracker/session lease, then reconnect.
            call = stub.ProcessVideo(timeout=30)
            await call.write(pb.VideoChunk(data=jpeg, session_id=sid))
            assert (await call.read()).status_message == "success"
            await expect_error(
                "DeleteSession", pb.DeleteSessionRequest(session_id=sid), "FAILED_PRECONDITION"
            )
            call.cancel()
            await asyncio.sleep(0.1)
            response = await stream([pb.VideoChunk(data=jpeg, session_id=sid)])
            assert response[0].stats.tracker_frame == 1
            checks.append("cancel-reconnect-reset")
        finally:
            for sid in sessions:
                for attempt in range(20):
                    try:
                        await stub.DeleteSession(
                            pb.DeleteSessionRequest(session_id=sid), timeout=10
                        )
                        break
                    except grpc.aio.AioRpcError as e:
                        if e.code() != grpc.StatusCode.FAILED_PRECONDITION or attempt == 19:
                            raise
                        await asyncio.sleep(0.1)
            listed = await stub.ListSessions(pb.ListSessionsRequest(), timeout=10)
            assert not set(sessions) & {r.session_id for r in listed.sessions}
    return {
        "target": target,
        "passed": True,
        "checks": checks,
        "error_contract": trace,
        "temporary_sessions_deleted": len(sessions),
        "input_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
        "protobuf_sha256": hashlib.sha256(pb.DESCRIPTOR.serialized_pb).hexdigest(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = asyncio.run(verify(args.target, args.image))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
