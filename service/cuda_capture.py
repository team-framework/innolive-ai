"""Coordinate rare graph capture with libraries that synchronize the whole GPU.

Steady-state graph replay does not acquire this lock. YOLO already has one
inference lane; this only pauses it while a new identity's graph is captured.
"""

from threading import RLock

CUDA_CAPTURE_LOCK = RLock()
