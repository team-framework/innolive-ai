import unittest
from pathlib import Path

import numpy as np

from service.face_processor import BATCH_SIZE, Face, FaceProcessorPool, Settings


class RecordingSegmenter:
    def __init__(self):
        self.batch_sizes = []

    def predict(self, images):
        self.batch_sizes.append(len(images))
        return [
            [
                Face(
                    (index, 0, index + 1, 1),
                    0.9,
                    ((index, 0), (index + 1, 0), (index, 1)),
                )
            ]
            for index in range(len(images))
        ]


class OrderedTracker:
    def __init__(self):
        self.next_id = 1

    def update(self, faces, image):
        tracked = tuple(
            Face(face.bbox, face.confidence, face.polygon, self.next_id)
            for face in faces
        )
        self.next_id += 1
        return tracked


class FaceProcessorTest(unittest.TestCase):
    def setUp(self):
        self.segmenter = RecordingSegmenter()
        settings = Settings(Path("unused.pt"), ("cpu",), decode_workers=4)
        self.pool = FaceProcessorPool(
            settings,
            segmenters=[self.segmenter],
            decoder=lambda jpeg: np.full((2, 2, 3), jpeg[0], dtype=np.uint8),
            tracker_factory=OrderedTracker,
        )

    def tearDown(self):
        self.pool.close()

    def test_partial_input_is_padded_for_one_fixed_b4_inference(self):
        results = self.pool.open_stream().process([b"\x01", b"\x02", b"\x03"])

        self.assertEqual(self.segmenter.batch_sizes, [BATCH_SIZE])
        self.assertEqual(len(results), 3)
        self.assertEqual([result.faces[0].track_id for result in results], [1, 2, 3])
        self.assertEqual([int(result.image[0, 0, 0]) for result in results], [1, 2, 3])

    def test_tracking_state_is_isolated_by_stream(self):
        first_stream = self.pool.open_stream()
        second_stream = self.pool.open_stream()

        first = first_stream.process([b"\x01"])[0]
        second = second_stream.process([b"\x02"])[0]

        self.assertEqual(first.faces[0].track_id, 1)
        self.assertEqual(second.faces[0].track_id, 1)

    def test_frames_from_multiple_streams_share_one_gpu_batch(self):
        streams = [self.pool.open_stream() for _ in range(BATCH_SIZE)]
        pending = [
            stream.submit(bytes([index + 1])) for index, stream in enumerate(streams)
        ]
        results = [result.result() for result in pending]

        self.assertEqual(self.segmenter.batch_sizes, [BATCH_SIZE])
        self.assertEqual([frame.faces[0].track_id for frame in results], [1] * 4)
        self.assertEqual(
            [int(frame.image[0, 0, 0]) for frame in results],
            [1, 2, 3, 4],
        )

    def test_tracking_order_is_preserved_across_pipelined_batches(self):
        stream = self.pool.open_stream()
        pending = [stream.submit(bytes([index + 1])) for index in range(8)]
        results = [result.result() for result in pending]

        self.assertEqual(self.segmenter.batch_sizes, [BATCH_SIZE, BATCH_SIZE])
        self.assertEqual(
            [frame.faces[0].track_id for frame in results],
            list(range(1, 9)),
        )

    def test_single_frame_uses_the_b1_segmenter(self):
        batch_segmenter = RecordingSegmenter()
        single_segmenter = RecordingSegmenter()
        settings = Settings(Path("unused.pt"), ("cpu",), decode_workers=1)
        pool = FaceProcessorPool(
            settings,
            segmenters=[batch_segmenter],
            single_segmenters=[single_segmenter],
            decoder=lambda jpeg: np.zeros((2, 2, 3), dtype=np.uint8),
            tracker_factory=OrderedTracker,
        )
        try:
            frame = pool.open_stream().process([b"\x01"])[0]
        finally:
            pool.close()

        self.assertEqual(single_segmenter.batch_sizes, [1])
        self.assertEqual(batch_segmenter.batch_sizes, [])
        self.assertEqual(frame.timing.inference_batch_size, 1)


if __name__ == "__main__":
    unittest.main()
