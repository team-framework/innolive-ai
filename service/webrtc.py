"""Bounded WebRTC video bridge to the existing raw-YUV ProcessVideo contract."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from contextlib import suppress
from fractions import Fraction
from uuid import uuid4

import av
import numpy as np
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCIceServer,
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
)
from aiortc.mediastreams import MediaStreamError
from fastapi import Request
from fastapi.responses import JSONResponse, Response

from grpc_client import VideoClientError, VideoFrame, VideoRpcError
from service.recognition import validate_session_id
from service.webrtc_encoder import configure_encoder

LOGGER = logging.getLogger("innolive.webrtc")
VIDEO_TIME_BASE = Fraction(1, 90_000)
_END = object()


class LatestFrameSlot:
    """Keep one pending frame; replace stale input instead of accumulating latency."""

    def __init__(self):
        self.queue = asyncio.Queue(maxsize=1)
        self.received = self.dropped = 0
        self.closed = False

    def put(self, frame):
        if self.closed:
            return
        self.received += 1
        if self.queue.full():
            self.queue.get_nowait()
            self.dropped += 1
        self.queue.put_nowait(frame)

    async def get(self):
        frame = await self.queue.get()
        if frame is _END:
            raise MediaStreamError
        return frame

    def close(self):
        self.closed = True
        if self.queue.full():
            self.queue.get_nowait()
        self.queue.put_nowait(_END)


def pack_frame(frame, frame_id: int, previous_pts: int, mode: str) -> VideoFrame:
    scale = min(1, 640 / max(frame.width, frame.height))
    width, height = (int(value * scale) // 2 * 2 for value in (frame.width, frame.height))
    if min(width, height) < 32:
        raise ValueError("video aspect ratio is outside the supported limits")
    pts = (
        int(frame.pts * frame.time_base / VIDEO_TIME_BASE)
        if frame.pts is not None
        else previous_pts + 3000
    )
    pts = max(previous_pts + 1, pts)
    yuv = frame.reformat(width=width, height=height, format="yuv420p")
    return VideoFrame(
        yuv.to_ndarray().tobytes(),
        pts,
        frame_id,
        anonymization_mode=mode,
        width=width,
        height=height,
        pix_fmt="yuv420p",
    )


class ProcessedVideoTrack(MediaStreamTrack):
    kind = "video"

    def __init__(self, source, client, session_id, mode, report, fail, fps=30):
        super().__init__()
        self.source, self.client, self.session_id = source, client, session_id
        self.mode, self.report, self.fail, self.fps = mode, report, fail, fps
        self.slot = LatestFrameSlot()
        self.processed = 0
        self.last_received = time.monotonic()
        self.drain_task = asyncio.create_task(self._drain())
        self.results = client.process_video(self._frames(), session_id=session_id, window=1)
        self.closed = False

    async def _drain(self):
        try:
            while True:
                frame = await self.source.recv()
                self.last_received = time.monotonic()
                self.slot.put(frame)
        except (MediaStreamError, asyncio.CancelledError):
            pass
        except Exception:
            LOGGER.exception("WebRTC incoming video failed")
        finally:
            self.slot.close()

    async def _frames(self):
        seq, pts, last = 0, -1, 0.0
        while not self.closed:
            await asyncio.sleep(max(0, last + 1 / self.fps - time.monotonic()))
            frame = await asyncio.wait_for(self.slot.get(), 15)
            last = time.monotonic()
            seq += 1
            request = await asyncio.to_thread(pack_frame, frame, seq, pts, self.mode)
            pts = request.timestamp
            yield request

    async def recv(self):
        try:
            result = await asyncio.wait_for(anext(self.results), 60)
            response = result.response
            array = np.frombuffer(result.processed_data, np.uint8).reshape(
                response.height * 3 // 2, response.width
            )
            frame = av.VideoFrame.from_ndarray(array, format="yuv420p")
            frame.pts, frame.time_base = response.timestamp, VIDEO_TIME_BASE
            self.processed += 1
            self.report(
                {
                    "type": "frame",
                    "seq": response.frame_id,
                    "server_ms": round(response.timing.server_total_ms, 2),
                    "received": self.slot.received,
                    "processed": self.processed,
                    "dropped": self.slot.dropped,
                    "faces": dict(
                        Counter(
                            f.anonymization.status for f in response.faces if f.anonymization.status
                        )
                    ),
                }
            )
            return frame
        except asyncio.CancelledError:
            raise
        except Exception:
            self.report(
                {"type": "error", "message": "영상 처리 연결이 종료됐습니다. 다시 시작해 주세요."}
            )
            self.fail()
            raise MediaStreamError from None

    async def close(self):
        if self.closed:
            return
        self.closed = True
        self.stop()
        self.source.stop()
        self.slot.close()
        self.drain_task.cancel()
        await asyncio.gather(self.drain_task, return_exceptions=True)
        await self.results.aclose()


class WebRtcGateway:
    def __init__(self, settings):
        self.settings = settings
        self.peers = {}
        self.encoder = None
        self.sweeper = None
        self.stopping = False
        self.close_tasks = set()

    async def start(self):
        self.encoder = await asyncio.to_thread(configure_encoder, self.settings.webrtc_encoder)
        self.sweeper = asyncio.create_task(self._sweep())

    def config(self):
        return {
            "enabled": True,
            "codec": "H264",
            "encoder": self.encoder,
            "max_long_edge": 640,
            "target_fps": min(self.settings.target_fps or 30, 30),
            "max_bitrate": 1_500_000,
            "ice_servers": list(self.settings.webrtc_ice_servers),
            "active_peers": len(self.peers),
            "max_peers": self.settings.webrtc_max_peers,
        }

    async def capabilities(self, request: Request):
        return JSONResponse(self.config(), headers={"Cache-Control": "no-store"})

    async def offer(self, request: Request):
        if self.stopping or not request.app.state.ready:
            return self.error(503, "서버가 준비되지 않았습니다.")
        if len(self.peers) >= self.settings.webrtc_max_peers:
            return self.error(429, "현재 영상 연결이 가득 찼습니다.")
        try:
            raw = bytearray()
            async for chunk in request.stream():
                raw.extend(chunk)
                if len(raw) > 65_536:
                    return self.error(413, "연결 요청이 너무 큽니다.")
            body = json.loads(raw)
            if not isinstance(body, dict) or body.get("type") != "offer":
                raise ValueError("expected an SDP offer")
            sid = validate_session_id(body.get("session_id", ""))
            if not isinstance(body.get("sdp"), str) or not body["sdp"].startswith("v=0"):
                raise ValueError("invalid SDP")
            client = request.app.state.grpc_client
            await client.get_whitelist_status(sid)
        except VideoRpcError as error:
            return self.error(
                404 if error.code.name == "NOT_FOUND" else 503, "세션을 확인할 수 없습니다."
            )
        except (TypeError, ValueError, VideoClientError):
            return self.error(400, "잘못된 영상 연결 요청입니다.")
        # No await between this second capacity check and reservation.
        if len(self.peers) >= self.settings.webrtc_max_peers:
            return self.error(429, "현재 영상 연결이 가득 찼습니다.")
        pc = RTCPeerConnection(
            RTCConfiguration(
                iceServers=[RTCIceServer(**s) for s in self.settings.webrtc_ice_servers]
            )
        )
        peer_id = uuid4().hex
        state = {"pc": pc, "tracks": [], "data": None, "created": time.monotonic()}
        self.peers[peer_id] = state

        def report(payload):
            channel = state["data"]
            if channel and channel.readyState == "open" and channel.bufferedAmount < 16_384:
                channel.send(json.dumps(payload, separators=(",", ":")))

        def fail():
            task = asyncio.create_task(self.close_peer(peer_id))
            self.close_tasks.add(task)
            task.add_done_callback(self.close_tasks.discard)

        @pc.on("datachannel")
        def datachannel(channel):
            if channel.label == "stats" and state["data"] is None:
                state["data"] = channel
            else:
                channel.close()

        @pc.on("track")
        def track(source):
            if source.kind != "video" or state["tracks"]:
                source.stop()
                return
            output = ProcessedVideoTrack(
                source,
                client,
                sid,
                self.settings.anonymization_mode,
                report,
                fail,
                fps=min(self.settings.target_fps or 30, 30),
            )
            state["tracks"].append(output)
            pc.addTrack(output)

        @pc.on("connectionstatechange")
        async def connectionstatechange():
            if pc.connectionState in {"failed", "closed"}:
                await self.close_peer(peer_id)

        try:
            async with asyncio.timeout(20):
                await pc.setRemoteDescription(RTCSessionDescription(sdp=body["sdp"], type="offer"))
                transceivers = pc.getTransceivers()
                if len(transceivers) != 1 or not state["tracks"]:
                    raise ValueError("exactly one video track is required")
                codecs = [
                    c
                    for c in RTCRtpSender.getCapabilities("video").codecs
                    if c.mimeType.lower() == "video/h264"
                ]
                transceivers[0].setCodecPreferences(codecs)
                await pc.setLocalDescription(await pc.createAnswer())
            return JSONResponse(
                {
                    "sdp": pc.localDescription.sdp,
                    "type": pc.localDescription.type,
                    "peer_id": peer_id,
                }
            )
        except asyncio.CancelledError:
            await self.close_peer(peer_id)
            raise
        except Exception:
            await self.close_peer(peer_id)
            LOGGER.warning("WebRTC negotiation rejected", exc_info=True)
            return self.error(400, "H.264 영상 연결을 만들 수 없습니다.")

    async def delete(self, request: Request):
        await self.close_peer(request.path_params["peer_id"])
        return Response(status_code=204)

    async def close_peer(self, peer_id):
        state = self.peers.pop(peer_id, None)
        if state is None:
            return
        await state["pc"].close()
        for track in state["tracks"]:
            with suppress(Exception):
                await track.close()

    async def _sweep(self):
        while True:
            await asyncio.sleep(5)
            now = time.monotonic()
            stale = [
                key
                for key, state in self.peers.items()
                if now - max([state["created"], *[t.last_received for t in state["tracks"]]]) > 20
            ]
            await asyncio.gather(*(self.close_peer(key) for key in stale))

    async def close(self):
        self.stopping = True
        if self.sweeper:
            self.sweeper.cancel()
            await asyncio.gather(self.sweeper, return_exceptions=True)
        await asyncio.gather(*(self.close_peer(key) for key in list(self.peers)))
        await asyncio.gather(*self.close_tasks, return_exceptions=True)

    @staticmethod
    def error(status, message):
        return JSONResponse(
            {"error": {"code": "WEBRTC_ERROR", "message": message}}, status_code=status
        )
