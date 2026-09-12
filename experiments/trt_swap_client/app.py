#!/usr/bin/env python3
"""WebSocket test client for batched YOLO face swap with fail-closed blur."""

from __future__ import annotations

import argparse
import asyncio
import base64
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from experiments.trt_swap_client.video_io import (
    NvdecReader,
    NvencHlsWriter,
    probe_video,
    require_nvcodec_ffmpeg,
)
from service.adaface_model import AdaFaceConfig, AdaFaceRuntime
from service.detection import is_face_object, is_number_plate_object
from service.mosaic import DEFAULT_BLUR_RADIUS, DEFAULT_PIXEL_SIZE, MASK_FEATHER_RADIUS
from service.recognition import RecognitionConfig, SessionRegistry, StreamRecognition
from service.runtime import IMAGE_SIZE, MAX_DETECTIONS, MAX_POLYGON_POINTS
from service.tracking import DETECTOR_CONFIDENCE, StreamTracker

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENGINE = ROOT / "models" / "best_swap_b4.engine"
DEFAULT_SOURCE = Path.home() / "Documents" / "input.png"
DEFAULT_SWAPPER = ROOT / "models" / "face_swap" / "inswapper_128.onnx"
DEFAULT_HLS_DIR = ROOT / "face_swap_lab_output" / "trt_hls"


@dataclass(frozen=True, slots=True)
class Settings:
    detector: Path
    swapper: Path
    source: Path
    device: str
    max_batch: int
    batch_wait_ms: float
    max_queue: int
    swap_ort_mem_gib: float
    input_video: Path | None
    hls_dir: Path


@dataclass(slots=True)
class StreamState:
    session_id: str
    tracker: StreamTracker
    recognition: StreamRecognition
    sequence: int = 0


@dataclass(slots=True)
class FrameJob:
    frame: np.ndarray
    stream: StreamState
    future: asyncio.Future[tuple[np.ndarray, dict[str, Any]]]


def _swap_providers(device: str, memory_gib: float) -> list[Any]:
    """Bound ONNX Runtime's per-session CUDA arena without changing model outputs."""

    return [
        (
            "CUDAExecutionProvider",
            {
                "device_id": int(device),
                "gpu_mem_limit": int(memory_gib * 1024**3),
                "arena_extend_strategy": "kSameAsRequested",
                "do_copy_in_default_stream": True,
            },
        ),
        "CPUExecutionProvider",
    ]


class InSwapper:
    """Keep the established generator/face-analysis behavior, isolated from server code."""

    def __init__(self, source_path: Path, model_path: Path, providers: list[Any]):
        if not model_path.is_file():
            raise FileNotFoundError(f"InSwapper model is missing: {model_path}")
        from insightface import model_zoo
        from insightface.app import FaceAnalysis

        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if source is None:
            raise ValueError(f"could not read swap source: {source_path}")
        self.analysis = FaceAnalysis(
            name="buffalo_l",
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        self.analysis.prepare(ctx_id=0, det_size=(640, 640))
        self.model = model_zoo.get_model(str(model_path), providers=providers)
        faces = self.analysis.get(source)
        if not faces:
            raise ValueError("no source face found")
        self.source_face = max(
            faces, key=lambda face: float(np.prod(face.bbox[2:] - face.bbox[:2]))
        )

    def apply(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray:
        """Swap only a class-0 YOLO ROI; detector results are never passed here."""

        faces = self.analysis.get(frame)
        if not faces:
            raise RuntimeError("swapper could not align detected face")
        target = max(faces, key=lambda face: _iou(face.bbox, bbox))
        if _iou(target.bbox, bbox) < 0.2:
            raise RuntimeError("swapper alignment does not match YOLO face")
        return self.model.get(frame, target, self.source_face, paste_back=True)


class SwapLab:
    def __init__(self, settings: Settings):
        from ultralytics import YOLO

        self.settings = settings
        self.model = YOLO(str(settings.detector), task="segment")
        self.names = {int(key): str(value) for key, value in self.model.names.items()}
        if self.names != {0: "face", 1: "number_plate"}:
            raise RuntimeError(f"expected class 0=face and 1=number_plate, got {self.names}")
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider is required for this GPU test client")
        providers = _swap_providers(settings.device, settings.swap_ort_mem_gib)
        self.swapper = InSwapper(settings.source, settings.swapper, providers)
        self.sessions = SessionRegistry()
        self.adaface = AdaFaceRuntime(
            AdaFaceConfig(device=f"cuda:{settings.device}", queue_capacity=32),
            fallback_device=settings.device,
        )
        self.queue: asyncio.Queue[FrameJob] = asyncio.Queue(maxsize=settings.max_queue)
        self.worker: asyncio.Task[None] | None = None
        self.frames = 0
        self.latencies: deque[float] = deque(maxlen=300)

    async def start(self) -> None:
        if not self.adaface.ready:
            raise RuntimeError(f"AdaFace unavailable: {self.adaface.load_error}")
        self.worker = asyncio.create_task(self._batch_loop(), name="yolo-swap-batcher")

    async def close(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            await asyncio.gather(self.worker, return_exceptions=True)
        await asyncio.to_thread(self.adaface.close)

    def create_stream(self, session_id: str) -> StreamState:
        self.sessions.get_or_create(session_id)
        return StreamState(
            session_id=session_id,
            tracker=StreamTracker(device=self.settings.device),
            recognition=StreamRecognition(self.adaface, RecognitionConfig(), owner=session_id),
        )

    async def submit(
        self, frame: np.ndarray, stream: StreamState
    ) -> tuple[np.ndarray, dict[str, Any]]:
        loop = asyncio.get_running_loop()
        future: asyncio.Future[tuple[np.ndarray, dict[str, Any]]] = loop.create_future()
        try:
            self.queue.put_nowait(FrameJob(frame, stream, future))
        except asyncio.QueueFull as error:
            raise RuntimeError("detector queue is saturated; drop this frame") from error
        return await future

    async def enroll(self, session_id: str, image: np.ndarray) -> dict[str, Any]:
        future = self.adaface.submit_enrollment(image, owner=f"enroll:{session_id}")
        if future is None:
            raise HTTPException(503, "AdaFace queue is full")
        embedding = await future
        entry, count, version = self.sessions.append(session_id, embedding)
        return {"entry_id": entry.entry_id, "entry_count": count, "whitelist_version": version}

    async def _batch_loop(self) -> None:
        while True:
            first = await self.queue.get()
            jobs = [first]
            deadline = asyncio.get_running_loop().time() + self.settings.batch_wait_ms / 1000
            while len(jobs) < self.settings.max_batch:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    break
                try:
                    jobs.append(await asyncio.wait_for(self.queue.get(), remaining))
                except TimeoutError:
                    break
            try:
                results = await asyncio.to_thread(self._predict, [job.frame for job in jobs])
                for job, prediction in zip(jobs, results, strict=True):
                    output, meta = await self._compose(job.frame, prediction, job.stream, len(jobs))
                    job.future.set_result((output, meta))
            except Exception as error:
                for job in jobs:
                    if not job.future.done():
                        job.future.set_exception(error)
            finally:
                for _ in jobs:
                    self.queue.task_done()

    def _predict(self, frames: list[np.ndarray]) -> list[Any]:
        return list(
            self.model.predict(
                source=frames,
                batch=len(frames),
                imgsz=IMAGE_SIZE,
                conf=DETECTOR_CONFIDENCE,
                iou=0.70,
                classes=[0, 1],
                max_det=MAX_DETECTIONS,
                retina_masks=True,
                device=self.settings.device,
                half=True,
                verbose=False,
            )
        )

    async def _compose(
        self, frame: np.ndarray, prediction: Any, stream: StreamState, batch_size: int
    ) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.perf_counter()
        boxes = prediction.boxes.cpu().numpy()
        tracks = stream.tracker.update(boxes, frame)
        objects = _objects(prediction, tracks, self.names, frame.shape[1], frame.shape[0])
        objects, temporal = stream.tracker.stabilize(objects, frame.shape[1], frame.shape[0])
        stream.sequence += 1
        recognition = stream.recognition.process(
            frame, objects, self.sessions.snapshot(stream.session_id), stream.sequence
        )
        output = frame.copy()
        swapped = 0
        fallback = 0
        fallback_objects: list[dict[str, Any]] = []
        for item in objects:
            if is_number_plate_object(item):
                fallback_objects.append(item)
                fallback += 1
                continue
            if not is_face_object(item) or item.get("whitelisted") is True:
                continue
            # Class-0 faces alone can reach the generator. Any failure stays protected.
            try:
                output = self.swapper.apply(output, item["bbox"])
                swapped += 1
            except Exception:
                fallback_objects.append(item)
                fallback += 1
        if fallback_objects:
            output = _blur_objects(output, fallback_objects)
        elapsed = (time.perf_counter() - started) * 1_000
        self.frames += 1
        self.latencies.append(elapsed)
        return output, {
            "detections": len(objects),
            "swap_faces": swapped,
            "fallback_blurs": fallback,
            "yolo_batch": batch_size,
            "adaface": recognition,
            "tracking": temporal,
            "total_ms": round(elapsed, 2),
        }


def _objects(prediction: Any, tracks: np.ndarray, names: dict[int, str], width: int, height: int):
    polygons = prediction.masks.xy if prediction.masks is not None else []
    objects: list[dict[str, Any]] = []
    for row in tracks:
        index = int(row[-1])
        if not 0 <= index < len(prediction.boxes):
            continue
        try:
            polygon = (
                np.asarray(polygons[index], dtype=np.float32).reshape((-1, 2))
                if index < len(polygons)
                else np.empty((0, 2), dtype=np.float32)
            )
        except (TypeError, ValueError):
            polygon = np.empty((0, 2), dtype=np.float32)
        if len(polygon) < 3 or not np.isfinite(polygon).all():
            x1, y1, x2, y2 = row[:4]
            polygon = np.asarray(((x1, y1), (x2, y1), (x2, y2), (x1, y2)), dtype=np.float32)
        stride = max(1, int(np.ceil(len(polygon) / MAX_POLYGON_POINTS)))
        polygon = np.ascontiguousarray(polygon[::stride][:MAX_POLYGON_POINTS], dtype=np.float32)
        class_id = int(row[6])
        objects.append(
            {
                "track_id": int(row[4]),
                "class_id": class_id,
                "class_name": names[class_id],
                "confidence": float(row[5]),
                "bbox": [float(value) for value in row[:4]],
                "mask_polygon": polygon.tolist(),
                "mask_area_px": _polygon_area(polygon),
            }
        )
    return objects


def _polygon_area(polygon: np.ndarray) -> float:
    """Compute a finite contour area without requiring an OpenCV contour layout."""

    points = np.asarray(polygon, dtype=np.float32).reshape((-1, 2))
    if len(points) < 3 or not np.isfinite(points).all():
        return 0.0
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))) * 0.5)


def _blur_objects(image: np.ndarray, items: list[dict[str, Any]]) -> np.ndarray:
    """Apply the production fallback algorithm once, after all swap attempts."""

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for item in items:
        polygon = np.rint(np.asarray(item["mask_polygon"], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(mask, [polygon], 255)
    if not np.any(mask):
        return image
    if MASK_FEATHER_RADIUS > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (MASK_FEATHER_RADIUS * 2 + 1, MASK_FEATHER_RADIUS * 2 + 1)
        )
        expanded = cv2.dilate(mask, kernel)
        distance = cv2.distanceTransform(expanded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        alpha_mask = np.minimum(distance / MASK_FEATHER_RADIUS, 1.0) * 255.0
        alpha_mask = np.rint(alpha_mask).astype(np.uint8)
        alpha_mask[mask != 0] = 255
    else:
        alpha_mask = mask
    rows, columns = np.nonzero(alpha_mask)
    padding = int(np.ceil(DEFAULT_BLUR_RADIUS * 3))
    top = max(0, int(rows.min()) - padding)
    bottom = min(image.shape[0], int(rows.max()) + padding + 1)
    left = max(0, int(columns.min()) - padding)
    right = min(image.shape[1], int(columns.max()) + padding + 1)
    region = image[top:bottom, left:right]
    reduced_size = (
        max(1, int(np.ceil(region.shape[1] / DEFAULT_PIXEL_SIZE))),
        max(1, int(np.ceil(region.shape[0] / DEFAULT_PIXEL_SIZE))),
    )
    reduced = cv2.resize(region, reduced_size, interpolation=cv2.INTER_AREA)
    blurred = cv2.resize(
        cv2.GaussianBlur(reduced, (0, 0), DEFAULT_BLUR_RADIUS / DEFAULT_PIXEL_SIZE),
        (region.shape[1], region.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    output = image.copy()
    alpha = alpha_mask[top:bottom, left:right, None].astype(np.uint32)
    output[top:bottom, left:right] = (
        region.astype(np.uint32) * (255 - alpha) + blurred.astype(np.uint32) * alpha + 127
    ) // 255
    return output


def _iou(left: Any, right: list[float]) -> float:
    x1, y1, x2, y2 = (float(value) for value in left[:4])
    a1, b1, a2, b2 = right
    intersection = max(0.0, min(x2, a2) - max(x1, a1)) * max(0.0, min(y2, b2) - max(y1, b1))
    union = (x2 - x1) * (y2 - y1) + (a2 - a1) * (b2 - b1) - intersection
    return intersection / union if union > 0 else 0.0


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="InnoLive TensorRT Face Swap Lab")
    lab = SwapLab(settings)
    file_task: asyncio.Task[None] | None = None
    settings.hls_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/hls", StaticFiles(directory=str(settings.hls_dir)), name="hls")

    @app.on_event("startup")
    async def startup() -> None:
        nonlocal file_task
        await lab.start()
        if settings.input_video is not None:
            file_task = asyncio.create_task(
                _run_nvcodec_file(lab, settings.input_video, settings.hls_dir),
                name="nvdec-nvenc-file-swap",
            )

    @app.on_event("shutdown")
    async def shutdown() -> None:
        if file_task is not None:
            file_task.cancel()
            await asyncio.gather(file_task, return_exceptions=True)
        await lab.close()

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return _HTML

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "ready": True,
            "frames": lab.frames,
            "queue": lab.queue.qsize(),
            "p50_ms": float(np.percentile(lab.latencies, 50)) if lab.latencies else None,
        }

    @app.post("/api/enroll/{session_id}")
    async def enroll(session_id: str, payload: dict[str, str]) -> dict[str, Any]:
        raw = base64.b64decode(payload.get("jpeg", ""), validate=True)
        image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise HTTPException(400, "invalid jpeg")
        return await lab.enroll(session_id, image)

    @app.websocket("/ws/{session_id}")
    async def stream(websocket: WebSocket, session_id: str) -> None:
        await websocket.accept()
        state = lab.create_stream(session_id)
        try:
            while True:
                payload = await websocket.receive_bytes()
                frame = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue
                try:
                    output, metadata = await lab.submit(frame, state)
                except RuntimeError as error:
                    await websocket.send_json({"error": str(error), "dropped": True})
                    continue
                ok, encoded = cv2.imencode(".jpg", output, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    await websocket.send_json(metadata)
                    await websocket.send_bytes(encoded.tobytes())
        except WebSocketDisconnect:
            state.recognition.close()
            state.tracker.reset()

    return app


async def _run_nvcodec_file(lab: SwapLab, input_video: Path, output_dir: Path) -> None:
    """Run an optional hardware decode/encode stream through the same batcher."""

    ffmpeg = await asyncio.to_thread(require_nvcodec_ffmpeg)
    spec = await asyncio.to_thread(probe_video, input_video)
    reader = NvdecReader(input_video, spec, ffmpeg=ffmpeg)
    writer = NvencHlsWriter(output_dir, spec, ffmpeg=ffmpeg)
    state = lab.create_stream("nvcodec-file")
    try:
        while frame := await asyncio.to_thread(reader.read):
            output, _ = await lab.submit(frame, state)
            await asyncio.to_thread(writer.write, output)
    finally:
        state.recognition.close()
        state.tracker.reset()
        await asyncio.to_thread(reader.close)
        await asyncio.to_thread(writer.close)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--detector", type=Path, default=DEFAULT_ENGINE)
    parser.add_argument("--swapper", type=Path, default=DEFAULT_SWAPPER)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-batch", type=int, default=4)
    parser.add_argument("--batch-wait-ms", type=float, default=3.0)
    parser.add_argument("--max-queue", type=int, default=16)
    parser.add_argument(
        "--swap-ort-mem-gib",
        type=float,
        default=2.0,
        help="per-session ONNX Runtime CUDA arena limit for InSwapper (default: 2 GiB)",
    )
    parser.add_argument(
        "--input-video",
        type=Path,
        help="optional compressed input decoded by NVDEC and written as /hls/live.m3u8 via NVENC",
    )
    parser.add_argument("--hls-dir", type=Path, default=DEFAULT_HLS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.max_batch < 1
        or args.max_queue < args.max_batch
        or args.batch_wait_ms < 0
        or not 0 < args.swap_ort_mem_gib <= 8
    ):
        raise SystemExit(
            "max-batch >= 1, max-queue >= max-batch, batch-wait-ms >= 0, "
            "and swap-ort-mem-gib in (0, 8] are required"
        )
    input_video = args.input_video.expanduser().resolve() if args.input_video else None
    if input_video is not None and not input_video.is_file():
        raise SystemExit(f"input video not found: {input_video}")
    settings = Settings(
        args.detector.expanduser().resolve(),
        args.swapper.expanduser().resolve(),
        args.source.expanduser().resolve(),
        args.device,
        args.max_batch,
        args.batch_wait_ms,
        args.max_queue,
        args.swap_ort_mem_gib,
        input_video,
        args.hls_dir.expanduser().resolve(),
    )
    import uvicorn

    uvicorn.run(create_app(settings), host=args.host, port=args.port)


_HTML = """<!doctype html><meta charset=utf-8><title>TensorRT Swap Lab</title><style>body{font:16px system-ui;background:#111;color:#eee;margin:2rem}video,img{width:min(48%,720px);background:#222}pre{background:#222;padding:1rem}</style><h1>TensorRT Face Swap Lab</h1><p>Browser webcam → batched YOLO → class-0 swap / protected fallback</p><video id=v autoplay muted playsinline></video><img id=o><pre id=m>starting…</pre><script>const id=crypto.randomUUID(),ws=new WebSocket(`${location.protocol==='https:'?'wss':'ws'}://${location.host}/ws/${id}`),v=document.querySelector('#v'),o=document.querySelector('#o'),m=document.querySelector('#m'),c=document.createElement('canvas');let busy=false;navigator.mediaDevices.getUserMedia({video:{width:1920,height:1080},audio:false}).then(s=>v.srcObject=s);ws.onmessage=e=>{if(typeof e.data==='string'){m.textContent=e.data;return}o.src=URL.createObjectURL(e.data);busy=false};setInterval(()=>{if(busy||!v.videoWidth||ws.readyState!==1)return;busy=true;c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d').drawImage(v,0,0);c.toBlob(b=>{if(b)ws.send(b);else busy=false},'image/jpeg',.9)},33)</script>"""


if __name__ == "__main__":
    main()
