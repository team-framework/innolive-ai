"""Optional, version-pinned H.264 encoder selection for the WebRTC gateway."""

from __future__ import annotations

import logging
from fractions import Fraction

import aiortc
import av
import numpy as np
from aiortc import rtcrtpsender
from aiortc.codecs import get_encoder, h264
from aiortc.codecs.h264 import H264Encoder

LOGGER = logging.getLogger("innolive.webrtc")


def create_codec(name: str, width: int, height: int, bitrate: int):
    codec = av.CodecContext.create(name, "w")
    codec.width, codec.height = width, height
    codec.pix_fmt = "yuv420p"
    codec.bit_rate = bitrate
    codec.framerate = Fraction(30, 1)
    # Browser capture timestamps are not exact multiples of 1/30. Quantizing
    # them can merge adjacent encoded frames into one RTP timestamp and stall decoding.
    codec.time_base = Fraction(1, 90_000)
    codec.gop_size = 60
    codec.max_b_frames = 0
    codec.profile = "Baseline"
    if name == "h264_nvenc":
        codec.options = {
            "preset": "p1",
            "tune": "ull",
            "rc": "cbr",
            "zerolatency": "1",
            "rc-lookahead": "0",
            "delay": "0",
            "level": "31",
        }
    else:
        codec.thread_count = 2
        codec.options = {"preset": "veryfast", "tune": "zerolatency", "level": "31"}
    return codec


def probe_encoder(name: str) -> None:
    codec = create_codec(name, 640, 480, 1_500_000)
    frame = av.VideoFrame.from_ndarray(np.zeros((720, 640), np.uint8), format="yuv420p")
    frame.pts, frame.time_base = 0, Fraction(1, 30)
    if not codec.encode(frame):
        raise RuntimeError(f"{name} buffers the first frame")


class RealtimeH264Encoder(H264Encoder):
    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.target_bitrate = 1_500_000

    def _encode_frame(self, frame, force_keyframe):
        # Configure the codec before aiortc packetizes it. Keep aiortc's RTP,
        # keyframe requests, and REMB bitrate adaptation intact.
        if self.codec is None or (frame.width, frame.height) != (
            self.codec.width,
            self.codec.height,
        ):
            self.buffer_data = b""
            self.buffer_pts = None
            self.codec = create_codec(self.name, frame.width, frame.height, self.target_bitrate)
        elif self.name == "h264_nvenc" and (
            abs(self.target_bitrate - self.codec.bit_rate) / self.codec.bit_rate > 0.1
        ):
            # FFmpeg reconfigures NVENC bitrate in-place. Recreating the CUDA
            # encoder on every REMB update stalls video and forces keyframes.
            self.codec.bit_rate = self.target_bitrate
        elif abs(self.target_bitrate - self.codec.bit_rate) / self.codec.bit_rate > 0.1:
            self.buffer_data = b""
            self.buffer_pts = None
            self.codec = create_codec(self.name, frame.width, frame.height, self.target_bitrate)
        yield from super()._encode_frame(frame, force_keyframe)


def configure_encoder(preference: str) -> str:
    # The factory hook is deliberately limited to the tested aiortc release.
    if aiortc.__version__ != "1.15.0":
        raise RuntimeError("WebRTC requires aiortc==1.15.0; install requirements-webrtc.txt")
    if preference not in {"auto", "nvenc", "x264"}:
        raise ValueError("WebRTC encoder must be auto, nvenc or x264")
    # Fit RTP extensions, SRTP authentication, UDP and IPv6 inside a 1280-byte
    # tunnel MTU. aiortc's 1300-byte payload alone exceeds Tailscale's MTU.
    h264.PACKET_MAX = 1150
    selected = "libx264"
    if preference != "x264":
        try:
            probe_encoder("h264_nvenc")
            selected = "h264_nvenc"
        except Exception:
            if preference == "nvenc":
                raise
            LOGGER.warning("NVENC initialization failed; using low-latency libx264")
    if selected == "libx264":
        probe_encoder(selected)

    def factory(codec):
        if codec.mimeType.lower() == "video/h264":
            return RealtimeH264Encoder(selected)
        return get_encoder(codec)

    rtcrtpsender.get_encoder = factory
    return selected
