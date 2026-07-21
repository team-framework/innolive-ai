from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from service.face_processor import ProcessedFrame


@lru_cache(maxsize=8)
def _gaussian_kernel(sigma: float) -> np.ndarray:
    import cv2

    kernel_size = round(sigma * 6 + 1) | 1
    return cv2.getGaussianKernel(kernel_size, sigma, cv2.CV_32F)


def blur_faces(frame: ProcessedFrame, sigma: float = 24.0) -> np.ndarray:
    import cv2

    if not frame.faces:
        return frame.image

    output = frame.image.copy()
    height, width = frame.image.shape[:2]
    padding = math.ceil(sigma * 3)
    kernel = _gaussian_kernel(sigma)

    for face in frame.faces:
        box_x1, box_y1, box_x2, box_y2 = face.bbox
        x1 = max(0, math.floor(box_x1) - padding)
        y1 = max(0, math.floor(box_y1) - padding)
        x2 = min(width, math.ceil(box_x2) + padding)
        y2 = min(height, math.ceil(box_y2) + padding)
        if x1 >= x2 or y1 >= y2:
            continue

        source = frame.image[y1:y2, x1:x2]
        blurred = cv2.sepFilter2D(source, -1, kernel, kernel)
        polygon = np.asarray(face.polygon, dtype=np.int32) - (x1, y1)
        mask = np.zeros(source.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [polygon], 255)
        cv2.copyTo(blurred, mask, output[y1:y2, x1:x2])
    return output


def encode_blurred_jpeg(frame: ProcessedFrame, quality: int = 85) -> bytes:
    if not frame.faces and frame.source_jpeg is not None:
        return frame.source_jpeg

    import cv2

    success, buffer = cv2.imencode(
        ".jpg",
        blur_faces(frame),
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )
    if not success:
        raise ValueError("failed to encode processed frame")
    return buffer.tobytes()
