import unittest

import cv2
import numpy as np

from service.face_processor import Face, ProcessedFrame
from service.transform_image import blur_faces, encode_blurred_jpeg


class TransformImageTest(unittest.TestCase):
    def test_separable_blur_matches_gaussian_blur_without_touching_background(self):
        image = np.random.default_rng(7).integers(
            0,
            256,
            size=(300, 300, 3),
            dtype=np.uint8,
        )
        polygon = ((100, 100), (200, 100), (200, 200), (100, 200))
        frame = ProcessedFrame(image, (Face((100, 100, 200, 200), 0.9, polygon),))

        expected = image.copy()
        gaussian = cv2.GaussianBlur(image, (0, 0), sigmaX=24, sigmaY=24)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 255)
        cv2.copyTo(gaussian, mask, expected)

        difference = np.abs(blur_faces(frame).astype(int) - expected.astype(int))
        self.assertLessEqual(difference.max(), 1)
        self.assertTrue(np.array_equal(blur_faces(ProcessedFrame(image, ())), image))

    def test_no_face_frame_keeps_original_jpeg(self):
        source = b"original-jpeg"
        frame = ProcessedFrame(
            np.zeros((2, 2, 3), dtype=np.uint8),
            (),
            source_jpeg=source,
        )

        self.assertIs(encode_blurred_jpeg(frame), source)


if __name__ == "__main__":
    unittest.main()
