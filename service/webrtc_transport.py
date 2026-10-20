"""H.264 frame completion without waiting for the next camera frame."""

from aiortc.jitterbuffer import JitterBuffer, JitterFrame


class MarkerJitterBuffer(JitterBuffer):
    """Retain aiortc's loss/reordering handling, flush contiguous marked frames.

    H.264 RTP sets marker on the last packet of an access unit. aiortc 1.15.0
    otherwise waits for a different timestamp, adding a camera-frame interval.
    Missing packets still block delivery; a marker never bypasses a sequence gap.
    """

    def _remove_frame(self, sequence_number):
        frame = super()._remove_frame(sequence_number)
        if frame is not None:
            return frame
        packets = []
        timestamp = None
        for offset in range(self.capacity):
            packet = self._packets[(self._origin + offset) % self.capacity]
            if packet is None:
                return None
            if timestamp is None:
                timestamp = packet.timestamp
            if packet.timestamp != timestamp:
                return None
            packets.append(packet._data)
            if packet.marker:
                self.remove(offset + 1)
                return JitterFrame(b"".join(packets), timestamp)
        return None


def configure_receiver(receiver):
    # Private layout is covered by the gateway's aiortc==1.15.0 guard and tests.
    receiver._RTCRtpReceiver__jitter_buffer = MarkerJitterBuffer(capacity=128, is_video=True)
