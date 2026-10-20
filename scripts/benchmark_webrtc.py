#!/usr/bin/env python3
"""Compare JPEG/WebSocket and H.264/WebRTC using the same paced moving portrait.

Latency runs from frame creation to decoded return on one client clock. A visible
16-bit frame marker survives both H.264 encodes; server time is reported separately.
Use a synthetic portrait, not a live camera, for this reproducible transport test.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import av
import cv2
import httpx
import numpy as np
import websockets
from aiortc import (
    MediaStreamTrack,
    RTCConfiguration,
    RTCPeerConnection,
    RTCRtpSender,
    RTCSessionDescription,
)

from service.protocol import decode_result, encode_request


class MovingPortrait(MediaStreamTrack):
    kind = "video"

    def __init__(self, source, faces=1):
        super().__init__()
        self.source = cv2.resize(source, (640, 640))
        self.faces = faces
        self.started = None
        self.seq = 0
        self.sent = {}

    async def pixels(self):
        if self.started is None:
            self.started = time.perf_counter()
        await asyncio.sleep(max(0, self.started + self.seq / 30 - time.perf_counter()))
        self.seq = max(self.seq + 1, int((time.perf_counter() - self.started) * 30))
        matrix = cv2.getRotationMatrix2D((320, 320), 5 * math.sin(self.seq / 45), 0.8)
        matrix[1, 2] -= 80
        pixels = cv2.warpAffine(self.source, matrix, (640, 480), borderValue=(100, 100, 100))
        if self.faces == 2:
            portrait = cv2.resize(pixels, (320, 240))
            pixels[:] = 100
            pixels[100:340, :320] = portrait
            pixels[100:340, 320:] = portrait
        for bit in range(16):
            pixels[:16, bit * 16 : (bit + 1) * 16] = 255 if self.seq >> bit & 1 else 0
        self.sent[self.seq] = time.perf_counter()
        return self.seq, pixels

    async def recv(self):
        seq, pixels = await self.pixels()
        frame = av.VideoFrame.from_ndarray(pixels, format="bgr24")
        frame.pts, frame.time_base = seq * 3000, Fraction(1, 90_000)
        return frame


class Measurement:
    def __init__(self, source, output, warmup):
        self.source, self.output, self.warmup = source, output, warmup
        self.times, self.latencies, self.rows = [], [], []

    def record(self, pixels):
        now = time.perf_counter()
        seq = sum(int(pixels[4:12, b * 16 + 4 : b * 16 + 12].mean() > 128) << b for b in range(16))
        if seq not in self.source.sent or now - self.source.started < self.warmup:
            return
        self.times.append(now)
        self.latencies.append(1000 * (now - self.source.sent[seq]))
        self.rows.append({"seq": seq, "latency_ms": self.latencies[-1]})
        if len(self.times) == 20:
            cv2.imwrite(str(self.output), pixels)

    def summary(self):
        if len(self.times) < 10:
            raise RuntimeError("insufficient returned frames for comparison")
        return {
            "frames": len(self.times),
            "fps": (len(self.times) - 1) / (self.times[-1] - self.times[0]),
            "latency_p50_ms": float(np.median(self.latencies)),
            "latency_p95_ms": float(np.percentile(self.latencies, 95)),
            "latency_samples": self.rows,
        }


async def jpeg_run(args, http, source, sid):
    meter = Measurement(source, args.output / "jpeg.jpg", args.warmup)
    url = args.url.replace("https:", "wss:").replace("http:", "ws:")
    meta, sizes = [], []
    async with websockets.connect(url + "/ws?session_id=" + sid, max_size=8 * 1024 * 1024) as ws:
        slots = asyncio.Semaphore(2)

        async def send():
            while True:
                await slots.acquire()
                seq, pixels = await source.pixels()
                ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 90])
                assert ok
                packet = encode_request(seq, encoded.tobytes())
                sizes.append(len(packet))
                await ws.send(packet)

        sender = asyncio.create_task(send())
        try:
            start = time.perf_counter()
            while time.perf_counter() - start < args.seconds + args.warmup:
                packet = await asyncio.wait_for(ws.recv(), 60)
                result, jpeg = decode_result(packet)
                meter.record(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR))
                sizes.append(len(packet))
                meta.append(result)
                slots.release()
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
    rows = meta[5:]
    return meter.summary() | {
        "server_p50_ms": float(np.median([r["timing_ms"]["server_total"] for r in rows])),
        "swapped_frames": sum(
            any(o.get("anonymization", {}).get("status") == "swapped" for o in r["objects"])
            for r in rows
        ),
        "processed_frames": len(rows),
        "payload_bytes": sum(sizes),
    }


async def webrtc_run(args, http, source, sid):
    config = (await http.get(args.url + "/api/webrtc/config")).json()
    from aiortc import RTCIceServer

    pc = RTCPeerConnection(
        RTCConfiguration(iceServers=[RTCIceServer(**s) for s in config["ice_servers"]])
    )
    pc.addTrack(source)
    reports = []
    data = pc.createDataChannel("stats", ordered=False, maxRetransmits=0)

    @data.on("message")
    def message(value):
        report = json.loads(value)
        if report["type"] == "frame":
            reports.append(report)

    incoming = asyncio.Queue()

    @pc.on("track")
    def track(value):
        incoming.put_nowait(value)

    for transceiver in pc.getTransceivers():
        transceiver.setCodecPreferences(
            [
                c
                for c in RTCRtpSender.getCapabilities("video").codecs
                if c.mimeType.lower() == "video/h264"
            ]
        )
    peer_id = None
    meter = Measurement(source, args.output / "webrtc.jpg", args.warmup)
    try:
        await pc.setLocalDescription(await pc.createOffer())
        answer = await http.post(
            args.url + "/api/webrtc/offer",
            json={"sdp": pc.localDescription.sdp, "type": "offer", "session_id": sid},
        )
        answer.raise_for_status()
        answer = answer.json()
        peer_id = answer.pop("peer_id")
        await pc.setRemoteDescription(RTCSessionDescription(**answer))
        incoming_track = await asyncio.wait_for(incoming.get(), 30)
        start = time.perf_counter()
        while time.perf_counter() - start < args.seconds + args.warmup:
            frame = await asyncio.wait_for(incoming_track.recv(), 60)
            meter.record(frame.to_ndarray(format="bgr24"))
        transports = [s for s in (await pc.getStats()).values() if s.type == "transport"]
        rows = reports[5:]
        return meter.summary() | {
            "server_p50_ms": float(np.median([r["server_ms"] for r in rows])),
            "swapped_frames": sum(r["faces"].get("swapped", 0) > 0 for r in rows),
            "processed_frames": len(rows),
            "transport_bytes": sum(s.bytesSent + s.bytesReceived for s in transports),
            "encoder": config["encoder"],
        }
    finally:
        await pc.close()
        if peer_id:
            await http.delete(args.url + "/api/webrtc/" + peer_id)


async def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    source = cv2.imread(str(args.source))
    if source is None:
        raise ValueError("source image could not be read")
    summary = {}
    async with httpx.AsyncClient(timeout=90) as http:
        for name, run in (("jpeg", jpeg_run), ("webrtc", webrtc_run)):
            response = await http.post(args.url + "/api/sessions")
            response.raise_for_status()
            sid = response.json()["session_id"]
            try:
                summary[name] = await run(args, http, MovingPortrait(source, args.faces), sid)
                print(
                    name,
                    {k: v for k, v in summary[name].items() if k != "latency_samples"},
                    flush=True,
                )
            finally:
                await http.delete(args.url + "/api/sessions/" + sid)
    (args.output / "benchmark.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/webrtc"))
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--warmup", type=float, default=5)
    parser.add_argument("--faces", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    if not 5 <= args.seconds <= 120 or not 0 <= args.warmup <= 30:
        parser.error("seconds must be 5..120 and warmup 0..30")
    args.url = args.url.rstrip("/")
    asyncio.run(main(args))
