from __future__ import annotations

import os
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Protocol, Sequence

import numpy as np


BATCH_SIZE = 4
PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True, slots=True)
class Settings:
    model_path: Path
    devices: tuple[str, ...]
    single_model_path: Path | None = None
    confidence: float = 0.25
    image_size: int = 640
    decode_workers: int = min(32, (os.cpu_count() or 4) + 4)
    track_workers: int = min(16, os.cpu_count() or 4)
    # Keep the scheduler queue short enough that overload cannot turn into
    # hundreds of milliseconds of stale video.  The stream-level semaphore
    # applies the same bound on the gRPC side; this is the cross-stream cap.
    batch_wait_ms: float = 0.5
    batch_queue_size: int = 32

    @classmethod
    def from_env(cls) -> Settings:
        configured_model = os.getenv("AI_MODEL_PATH")
        engine = PROJECT_ROOT / "models" / "yolo.engine"
        model_path = Path(configured_model) if configured_model else engine
        if not model_path.exists() and not configured_model:
            model_path = PROJECT_ROOT / "models" / "yolo.pt"

        devices = tuple(
            device.strip()
            for device in os.getenv("AI_DEVICES", "0").split(",")
            if device.strip()
        )
        if not devices:
            raise ValueError("AI_DEVICES must contain at least one device")

        configured_single_model = os.getenv("AI_SINGLE_MODEL_PATH")
        default_single_model = PROJECT_ROOT / "models" / "yolo_b1.engine"
        single_model_path = (
            Path(configured_single_model)
            if configured_single_model
            else default_single_model if default_single_model.exists() else None
        )

        return cls(
            model_path=model_path,
            devices=devices,
            single_model_path=single_model_path,
            confidence=float(os.getenv("AI_CONFIDENCE", "0.25")),
            image_size=int(os.getenv("AI_IMAGE_SIZE", "640")),
            decode_workers=int(
                os.getenv("AI_DECODE_WORKERS", str(min(32, (os.cpu_count() or 4) + 4)))
            ),
            track_workers=int(
                os.getenv("AI_TRACK_WORKERS", str(min(16, os.cpu_count() or 4)))
            ),
            batch_wait_ms=float(os.getenv("AI_BATCH_WAIT_MS", "0.5")),
            batch_queue_size=int(os.getenv("AI_BATCH_QUEUE_SIZE", "32")),
        )


@dataclass(frozen=True, slots=True)
class Face:
    bbox: tuple[float, float, float, float]
    confidence: float
    polygon: tuple[tuple[float, float], ...]
    track_id: int | None = None
    metadata_polygon: tuple[tuple[float, float], ...] = ()

    @property
    def transport_polygon(self) -> tuple[tuple[float, float], ...]:
        return self.metadata_polygon or self.polygon

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox,
            "confidence": round(self.confidence, 4),
            "polygon": self.transport_polygon,
            "trackId": self.track_id,
        }


@dataclass(frozen=True, slots=True)
class ProcessingTiming:
    queue_ms: float = 0.0
    decode_ms: float = 0.0
    inference_ms: float = 0.0
    tracking_ms: float = 0.0
    inference_batch_size: int = 0


@dataclass(frozen=True, slots=True)
class ProcessedFrame:
    image: np.ndarray
    faces: tuple[Face, ...]
    timing: ProcessingTiming = ProcessingTiming()
    source_jpeg: bytes | None = None

    @property
    def height(self) -> int:
        return self.image.shape[0]

    @property
    def width(self) -> int:
        return self.image.shape[1]


class Segmenter(Protocol):
    def predict(self, images: Sequence[np.ndarray]) -> list[list[Face]]: ...


class Tracker(Protocol):
    def update(self, faces: Sequence[Face], image: np.ndarray) -> tuple[Face, ...]: ...


class YoloFaceSegmenter:
    def __init__(
        self,
        settings: Settings,
        device: str,
        model_path: Path,
        batch_size: int,
    ):
        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")

        from ultralytics import YOLO

        self._model = YOLO(str(model_path), task="segment")
        self._device = device
        self._batch_size = batch_size
        self._confidence = settings.confidence
        self._image_size = settings.image_size
        self._half = device != "cpu" and model_path.suffix != ".engine"
        self._lock = threading.Lock()
        self._warm_up()

    def predict(self, images: Sequence[np.ndarray]) -> list[list[Face]]:
        if len(images) != self._batch_size:
            raise ValueError(f"inference requires a fixed B{self._batch_size} input")

        with self._lock:
            options = {
                "source": list(images),
                "batch": self._batch_size,
                "classes": [0],
                "conf": self._confidence,
                "device": self._device,
                "imgsz": self._image_size,
                "retina_masks": True,
                "verbose": False,
            }
            if self._half:
                options["half"] = True
            results = self._model.predict(**options)
        return [self._extract_faces(result) for result in results]

    def _warm_up(self) -> None:
        image = np.zeros((self._image_size, self._image_size, 3), dtype=np.uint8)
        self.predict([image] * self._batch_size)
        self.predict([image] * self._batch_size)

    @staticmethod
    def _extract_faces(result) -> list[Face]:
        if result.boxes is None or len(result.boxes) == 0:
            return []

        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        polygons = result.masks.xy if result.masks is not None else [None] * len(boxes)

        faces = []
        for box, score, polygon in zip(boxes, scores, polygons):
            rounded_box = tuple(round(float(value), 1) for value in box)
            if polygon is None or len(polygon) < 3:
                x1, y1, x2, y2 = rounded_box
                points = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
                metadata_points = points
            else:
                points = tuple(
                    (round(float(x), 1), round(float(y), 1)) for x, y in polygon
                )
                metadata_points = tuple(
                    (round(float(x), 1), round(float(y), 1))
                    for x, y in YoloFaceSegmenter._simplify_polygon(polygon)
                )
            faces.append(
                Face(
                    rounded_box,
                    float(score),
                    points,
                    metadata_polygon=metadata_points,
                )
            )
        return faces

    @staticmethod
    def _simplify_polygon(polygon: np.ndarray) -> np.ndarray:
        import cv2

        contour = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
        epsilon = max(1.0, cv2.arcLength(contour, True) * 0.002)
        simplified = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        return simplified if len(simplified) >= 3 else np.asarray(polygon)


class DetectionSet:
    def __init__(self, faces: Sequence[Face]):
        self.xyxy = np.asarray([face.bbox for face in faces], dtype=np.float32).reshape(
            -1, 4
        )
        self.conf = np.asarray([face.confidence for face in faces], dtype=np.float32)
        self.cls = np.zeros(len(faces), dtype=np.float32)

    @property
    def xywh(self) -> np.ndarray:
        boxes = self.xyxy.copy()
        boxes[:, 2:] -= boxes[:, :2]
        boxes[:, :2] += boxes[:, 2:] / 2
        return boxes

    def __getitem__(self, index) -> DetectionSet:
        selected = object.__new__(DetectionSet)
        selected.xyxy = self.xyxy[index].reshape(-1, 4)
        selected.conf = self.conf[index].reshape(-1)
        selected.cls = self.cls[index].reshape(-1)
        return selected

    def __len__(self) -> int:
        return len(self.conf)


class BoTSortTracker:
    def __init__(self):
        from ultralytics.trackers.bot_sort import BOTSORT

        class StreamBOTSORT(BOTSORT):
            @staticmethod
            def reset_id() -> None:
                return None

        args = SimpleNamespace(
            track_high_thresh=0.25,
            track_low_thresh=0.1,
            new_track_thresh=0.25,
            track_buffer=30,
            match_thresh=0.8,
            fuse_score=True,
            gmc_method="sparseOptFlow",
            proximity_thresh=0.5,
            appearance_thresh=0.8,
            with_reid=False,
            model="auto",
        )
        self._tracker = StreamBOTSORT(args)
        self._local_ids: dict[int, int] = {}

    def update(self, faces: Sequence[Face], image: np.ndarray) -> tuple[Face, ...]:
        tracks = self._tracker.update(DetectionSet(faces), image)
        ids_by_detection: dict[int, int] = {}

        for track in tracks:
            raw_track_id = int(track[4])
            detection_index = int(track[-1])
            local_id = self._local_ids.setdefault(
                raw_track_id, len(self._local_ids) + 1
            )
            ids_by_detection[detection_index] = local_id

        return tuple(
            replace(face, track_id=ids_by_detection.get(index))
            for index, face in enumerate(faces)
        )


@dataclass(frozen=True, slots=True)
class _BatchItem:
    jpeg: bytes
    tracker: Tracker
    result: Future[ProcessedFrame]
    submitted_at: float


_STOP = object()


class _InferenceBatchWorker:
    def __init__(
        self,
        batch_segmenter: Segmenter,
        single_segmenter: Segmenter | None,
        decoder: Callable[[bytes], np.ndarray],
        decode_pool: ThreadPoolExecutor,
        track_pool: ThreadPoolExecutor,
        wait_ms: float,
        queue_size: int,
        worker_id: int,
    ):
        self._batch_segmenter = batch_segmenter
        self._single_segmenter = single_segmenter
        self._decoder = decoder
        self._decode_pool = decode_pool
        self._track_pool = track_pool
        self._wait_seconds = max(0.0, wait_ms / 1_000)
        self._tracker_tails: dict[int, Future] = {}
        self._tracker_lock = threading.Lock()
        self._queue: queue.Queue[_BatchItem | object] = queue.Queue(queue_size)
        self._thread = threading.Thread(
            target=self._run,
            name=f"inference-{worker_id}",
            daemon=True,
        )
        self._thread.start()

    def submit(self, jpeg: bytes, tracker: Tracker) -> Future[ProcessedFrame]:
        result: Future[ProcessedFrame] = Future()
        try:
            self._queue.put_nowait(
                _BatchItem(jpeg, tracker, result, time.perf_counter())
            )
        except queue.Full:
            result.set_exception(RuntimeError("inference queue is full"))
        return result

    def close(self) -> None:
        self._queue.put(_STOP)
        self._thread.join()

    def _run(self) -> None:
        while True:
            first = self._queue.get()
            if first is _STOP:
                return

            items = [first]
            deadline = time.perf_counter() + self._wait_seconds
            stopping = False
            while len(items) < BATCH_SIZE:
                timeout = deadline - time.perf_counter()
                if timeout <= 0:
                    break
                try:
                    item = self._queue.get(timeout=timeout)
                except queue.Empty:
                    break
                if item is _STOP:
                    stopping = True
                    break
                items.append(item)

            self._process(items)
            if stopping:
                return

    def _process(self, items: Sequence[_BatchItem]) -> None:
        try:
            processing_started = time.perf_counter()
            images = list(
                self._decode_pool.map(self._decoder, (item.jpeg for item in items))
            )
            decoded_at = time.perf_counter()
            if len(images) == 1 and self._single_segmenter is not None:
                detections = self._single_segmenter.predict(images)
                inference_batch_size = 1
            else:
                padded = images + [images[-1]] * (BATCH_SIZE - len(images))
                detections = self._batch_segmenter.predict(padded)
                inference_batch_size = BATCH_SIZE
            inferred_at = time.perf_counter()
        except Exception as error:
            for item in items:
                if item.result.set_running_or_notify_cancel():
                    item.result.set_exception(error)
            return

        decode_ms = (decoded_at - processing_started) * 1_000
        inference_ms = (inferred_at - decoded_at) * 1_000
        groups: dict[
            int,
            list[tuple[_BatchItem, np.ndarray, list[Face], ProcessingTiming]],
        ] = {}
        for item, image, faces in zip(items, images, detections):
            if item.result.set_running_or_notify_cancel():
                timing = ProcessingTiming(
                    queue_ms=(processing_started - item.submitted_at) * 1_000,
                    decode_ms=decode_ms,
                    inference_ms=inference_ms,
                    inference_batch_size=inference_batch_size,
                )
                groups.setdefault(id(item.tracker), []).append(
                    (item, image, faces, timing)
                )

        for tracker_id, group in groups.items():
            with self._tracker_lock:
                previous = self._tracker_tails.get(tracker_id)
                tracking_submitted_at = time.perf_counter()
                tracked = self._track_pool.submit(
                    self._track,
                    previous,
                    group,
                    tracking_submitted_at,
                )
                self._tracker_tails[tracker_id] = tracked
            tracked.add_done_callback(
                lambda result, key=tracker_id, items=group: self._finish_tracking(
                    key, items, result
                )
            )

    @staticmethod
    def _track(
        previous: Future | None,
        group: Sequence[
            tuple[_BatchItem, np.ndarray, list[Face], ProcessingTiming]
        ],
        submitted_at: float,
    ) -> list[tuple[_BatchItem, ProcessedFrame]]:
        if previous is not None:
            previous.result()
        tracked = []
        for item, image, faces, timing in group:
            tracked_faces = item.tracker.update(faces, image)
            tracked.append(
                (
                    item,
                    ProcessedFrame(
                        image,
                        tracked_faces,
                        replace(
                            timing,
                            tracking_ms=(time.perf_counter() - submitted_at) * 1_000,
                        ),
                        item.jpeg,
                    ),
                )
            )
        return tracked

    def _finish_tracking(
        self,
        tracker_id: int,
        group: Sequence[
            tuple[_BatchItem, np.ndarray, list[Face], ProcessingTiming]
        ],
        tracked: Future,
    ) -> None:
        try:
            for item, frame in tracked.result():
                item.result.set_result(frame)
        except Exception as error:
            for item, _, _, _ in group:
                if not item.result.done():
                    item.result.set_exception(error)
        finally:
            with self._tracker_lock:
                if self._tracker_tails.get(tracker_id) is tracked:
                    self._tracker_tails.pop(tracker_id)


class FaceProcessorPool:
    def __init__(
        self,
        settings: Settings,
        segmenters: Sequence[Segmenter] | None = None,
        single_segmenters: Sequence[Segmenter | None] | None = None,
        decoder: Callable[[bytes], np.ndarray] | None = None,
        tracker_factory: Callable[[], Tracker] = BoTSortTracker,
    ):
        self.settings = settings
        if segmenters is None:
            self._segmenters, self._single_segmenters = self._load_segmenters()
        else:
            self._segmenters = tuple(segmenters)
            self._single_segmenters = tuple(
                single_segmenters or [None] * len(self._segmenters)
            )
        if len(self._segmenters) != len(self._single_segmenters):
            raise ValueError("batch and single segmenter counts must match")
        self._decoder = decoder or self._decode_jpeg
        self._tracker_factory = tracker_factory
        self._decode_pool = ThreadPoolExecutor(
            max_workers=settings.decode_workers,
            thread_name_prefix="jpeg-decode",
        )
        self._track_pool = ThreadPoolExecutor(
            max_workers=settings.track_workers,
            thread_name_prefix="stream-tracker",
        )
        self._workers = tuple(
            _InferenceBatchWorker(
                segmenter,
                single_segmenter,
                self._decoder,
                self._decode_pool,
                self._track_pool,
                settings.batch_wait_ms,
                settings.batch_queue_size,
                worker_id,
            )
            for worker_id, (segmenter, single_segmenter) in enumerate(
                zip(self._segmenters, self._single_segmenters)
            )
        )
        self._next_worker = 0
        self._stream_lock = threading.Lock()

    def open_stream(self) -> FaceStream:
        with self._stream_lock:
            worker = self._workers[self._next_worker]
            self._next_worker = (self._next_worker + 1) % len(self._workers)
        return FaceStream(worker, self._tracker_factory())

    def close(self) -> None:
        for worker in self._workers:
            worker.close()
        self._track_pool.shutdown(wait=True)
        self._decode_pool.shutdown(wait=True, cancel_futures=True)

    def _load_segmenters(
        self,
    ) -> tuple[tuple[Segmenter, ...], tuple[Segmenter | None, ...]]:
        batch_segmenters = tuple(
            YoloFaceSegmenter(
                self.settings,
                device,
                self.settings.model_path,
                BATCH_SIZE,
            )
            for device in self.settings.devices
        )
        single_segmenters = tuple(
            YoloFaceSegmenter(
                self.settings,
                device,
                self.settings.single_model_path,
                1,
            )
            if self.settings.single_model_path is not None
            else None
            for device in self.settings.devices
        )
        return batch_segmenters, single_segmenters

    @staticmethod
    def _decode_jpeg(jpeg: bytes) -> np.ndarray:
        import cv2

        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("invalid JPEG frame")
        return image


class FaceStream:
    def __init__(self, worker: _InferenceBatchWorker, tracker: Tracker):
        self._worker = worker
        self._tracker = tracker

    def submit(self, jpeg: bytes) -> Future[ProcessedFrame]:
        return self._worker.submit(jpeg, self._tracker)

    def process(self, jpegs: Sequence[bytes]) -> list[ProcessedFrame]:
        if not 0 < len(jpegs) <= BATCH_SIZE:
            raise ValueError(f"a batch must contain 1 to {BATCH_SIZE} frames")
        results = [self.submit(jpeg) for jpeg in jpegs]
        return [result.result() for result in results]
