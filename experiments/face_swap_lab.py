#!/usr/bin/env python3
"""Local macOS webcam face-swap quality and synthetic-session load laboratory.

This program intentionally has no dependency on the gRPC server. It displays a
webcam target beside its processed result and repeats the exact same captured
frame for N independent synthetic sessions to expose compute contention. It is
not a network or video-codec capacity benchmark.

Use only source images and target video for which you have permission.
"""

from __future__ import annotations

import argparse
import statistics
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import tkinter as tk
    from tkinter import filedialog, ttk
except ImportError as error:  # pragma: no cover - platform-specific failure
    raise SystemExit(
        "Tk is required. Install a Python distribution that includes tkinter."
    ) from error


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path.home() / "Documents" / "input.png"
DEFAULT_MODEL = ROOT / "models" / "face_swap" / "inswapper_128.onnx"
MODEL_GEOMETRIC = "Geometric preview (no AI)"
MODEL_INSWAPPER = "InSwapper 128 (CoreML/CPU)"
MODEL_CHOICES = (MODEL_GEOMETRIC, MODEL_INSWAPPER)
SESSION_CHOICES = (1, 4, 16)
MAX_SAMPLE_COUNT = 180


def percentile(values: deque[float] | list[float], percent: float) -> float:
    """Return a linearly interpolated percentile without an extra dependency."""

    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * (percent / 100.0)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def coreml_providers() -> tuple[list[str], str]:
    """Prefer CoreML for ONNX; fall back explicitly instead of silently assuming GPU use."""

    try:
        import onnxruntime as ort
    except ImportError:
        return ["CPUExecutionProvider"], "ONNX Runtime is unavailable; CPU only"

    available = set(ort.get_available_providers())
    if "CoreMLExecutionProvider" in available:
        return ["CoreMLExecutionProvider", "CPUExecutionProvider"], "CoreML preferred"
    return ["CPUExecutionProvider"], "CoreML unavailable; CPU fallback"


def mps_status() -> str:
    """Report MPS availability for future Torch adapters without claiming ONNX uses it."""

    try:
        import torch
    except ImportError:
        return "PyTorch unavailable"
    return "MPS available" if torch.backends.mps.is_available() else "MPS unavailable"


def read_bgr(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"could not read source image: {path}")
    return image


def largest_face(rectangles: np.ndarray | tuple[Any, ...]) -> tuple[int, int, int, int] | None:
    if len(rectangles) == 0:
        return None
    return max(
        (tuple(int(value) for value in rect) for rect in rectangles),
        key=lambda rect: rect[2] * rect[3],
    )


def preview_caption(model: str, multiplier: int) -> str:
    return f"{model} · same frame processed by {multiplier} independent synthetic session(s)"


class GeometricPreviewSwapper:
    """Always-available non-AI baseline for webcam, masking, and load-path validation."""

    name = MODEL_GEOMETRIC

    def __init__(self, source: np.ndarray):
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self.detector = cv2.CascadeClassifier(str(cascade_path))
        if self.detector.empty():
            raise RuntimeError("OpenCV Haar face detector is unavailable")
        source_rect = self._detect(source)
        if source_rect is None:
            raise ValueError("no frontal face found in source image")
        x, y, width, height = source_rect
        self.source_face = source[y : y + height, x : x + width].copy()

    def _detect(self, frame: np.ndarray) -> tuple[int, int, int, int] | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rectangles = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.12,
            minNeighbors=5,
            minSize=(48, 48),
        )
        return largest_face(rectangles)

    def swap(self, frame: np.ndarray, *, max_faces: int) -> tuple[np.ndarray, int]:
        target_rect = self._detect(frame)
        if target_rect is None:
            return frame.copy(), 0
        x, y, width, height = target_rect
        if x < 0 or y < 0 or x + width > frame.shape[1] or y + height > frame.shape[0]:
            return frame.copy(), 0
        resized = cv2.resize(self.source_face, (width, height), interpolation=cv2.INTER_LANCZOS4)
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, (width // 2, height // 2), (width // 2, height // 2), 0, 0, 360, 255, -1)
        output = frame.copy()
        center = (x + width // 2, y + height // 2)
        try:
            output = cv2.seamlessClone(resized, output, mask, center, cv2.NORMAL_CLONE)
        except cv2.error:
            alpha = (mask.astype(np.float32) / 255.0)[..., None]
            region = output[y : y + height, x : x + width]
            output[y : y + height, x : x + width] = (
                resized * alpha + region * (1.0 - alpha)
            ).astype(np.uint8)
        return output, min(max_faces, 1)


class InSwapper128:
    """InsightFace adapter with a cached source embedding and CoreML-preferred ONNX sessions."""

    name = MODEL_INSWAPPER

    def __init__(self, source: np.ndarray, model_path: Path):
        if not model_path.is_file():
            raise FileNotFoundError(
                "InSwapper model is missing. Place a licensed model at "
                f"{model_path} or choose it with the UI."
            )
        try:
            from insightface import model_zoo
            from insightface.app import FaceAnalysis
        except ImportError as error:
            raise RuntimeError("install requirements-face-swap-lab.txt to use InSwapper") from error

        providers, self.provider_status = coreml_providers()
        self.analysis = FaceAnalysis(name="buffalo_l", providers=providers)
        self.analysis.prepare(ctx_id=0, det_size=(640, 640))
        self.swapper = model_zoo.get_model(str(model_path), providers=providers)
        source_faces = self.analysis.get(source)
        if not source_faces:
            raise ValueError("no face found in source image")
        self.source_face = max(source_faces, key=lambda face: self._area(face.bbox))

    @staticmethod
    def _area(bbox: Any) -> float:
        return float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))

    def swap(self, frame: np.ndarray, *, max_faces: int) -> tuple[np.ndarray, int]:
        faces = sorted(
            self.analysis.get(frame), key=lambda face: self._area(face.bbox), reverse=True
        )
        output = frame.copy()
        for target_face in faces[:max_faces]:
            output = self.swapper.get(output, target_face, self.source_face, paste_back=True)
        return output, min(len(faces), max_faces)


def build_swapper(model_name: str, source_path: Path, model_path: Path):
    source = read_bgr(source_path)
    if model_name == MODEL_GEOMETRIC:
        return GeometricPreviewSwapper(source), "OpenCV CPU baseline"
    if model_name == MODEL_INSWAPPER:
        swapper = InSwapper128(source, model_path)
        return swapper, swapper.provider_status
    raise ValueError(f"unknown model: {model_name}")


@dataclass(slots=True)
class LoadStats:
    batch_ms: deque[float]
    per_session_ms: deque[float]
    faces: deque[int]

    @classmethod
    def create(cls) -> LoadStats:
        return cls(
            deque(maxlen=MAX_SAMPLE_COUNT),
            deque(maxlen=MAX_SAMPLE_COUNT),
            deque(maxlen=MAX_SAMPLE_COUNT),
        )

    def add(self, batch_ms: float, multiplier: int, faces: int) -> None:
        self.batch_ms.append(batch_ms)
        self.per_session_ms.append(batch_ms / multiplier)
        self.faces.append(faces)

    def summary(self, multiplier: int) -> str:
        if not self.batch_ms:
            return "측정 대기 중"
        batch_mean = statistics.fmean(self.batch_ms)
        total_fps = 1000.0 / batch_mean if batch_mean else 0.0
        session_fps = total_fps / multiplier
        return (
            f"synthetic aggregate {total_fps:.1f} fps | per-session {session_fps:.2f} fps\n"
            f"batch p50/p95 {percentile(self.batch_ms, 50):.1f} / "
            f"{percentile(self.batch_ms, 95):.1f} ms\n"
            f"one session p50 {percentile(self.per_session_ms, 50):.1f} ms | "
            f"detected faces {statistics.fmean(self.faces):.1f}"
        )


class FaceSwapLabApp:
    def __init__(self, root: tk.Tk, *, source_path: Path, model_path: Path, camera: int):
        self.root = root
        self.source_path = source_path
        self.model_path = model_path
        self.camera_index = camera
        self.camera: cv2.VideoCapture | None = None
        self.swapper: Any | None = None
        self.stats = LoadStats.create()
        self.running = False
        self.last_raw: np.ndarray | None = None
        self.last_output: np.ndarray | None = None
        self.raw_photo: ImageTk.PhotoImage | None = None
        self.output_photo: ImageTk.PhotoImage | None = None

        self.model_var = tk.StringVar(value=MODEL_GEOMETRIC)
        self.session_var = tk.StringVar(value="1")
        self.max_faces_var = tk.StringVar(value="1")
        self.camera_var = tk.StringVar(value=str(camera))
        self.resolution_var = tk.StringVar(value="1280x720")
        self.status_var = tk.StringVar(value="준비됨")
        self.provider_var = tk.StringVar(value=f"CoreML: {coreml_providers()[1]} | {mps_status()}")
        self.metrics_var = tk.StringVar(value="측정 대기 중")
        self.caption_var = tk.StringVar(value="")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self) -> None:
        self.root.title("Face Swap Lab — macOS prototype")
        self.root.minsize(1120, 760)
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        controls = ttk.Frame(self.root, padding=12)
        controls.grid(row=0, column=0, sticky="ew")
        for column in range(10):
            controls.columnconfigure(column, weight=1 if column in {1, 3, 5} else 0)

        ttk.Label(controls, text="Model").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.model_var,
            values=MODEL_CHOICES,
            state="readonly",
            width=27,
        ).grid(row=0, column=1, sticky="ew", padx=(4, 12))
        ttk.Label(controls, text="Synthetic sessions").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            controls,
            textvariable=self.session_var,
            values=[str(value) for value in SESSION_CHOICES],
            state="readonly",
            width=5,
        ).grid(row=0, column=3, sticky="ew", padx=(4, 12))
        ttk.Label(controls, text="Max faces").grid(row=0, column=4, sticky="w")
        ttk.Spinbox(controls, from_=1, to=4, textvariable=self.max_faces_var, width=5).grid(
            row=0, column=5, sticky="w", padx=(4, 12)
        )
        ttk.Button(controls, text="Start", command=self.start).grid(row=0, column=6, padx=4)
        ttk.Button(controls, text="Stop", command=self.stop).grid(row=0, column=7, padx=4)
        ttk.Button(controls, text="Save pair", command=self.save_pair).grid(row=0, column=8, padx=4)
        ttk.Button(controls, text="Source…", command=self.choose_source).grid(
            row=0, column=9, padx=4
        )

        secondary = ttk.Frame(self.root, padding=(12, 0, 12, 8))
        secondary.grid(row=1, column=0, sticky="new")
        ttk.Label(secondary, text="Camera").grid(row=0, column=0, sticky="w")
        ttk.Entry(secondary, textvariable=self.camera_var, width=5).grid(
            row=0, column=1, padx=(4, 12)
        )
        ttk.Label(secondary, text="Capture").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            secondary,
            textvariable=self.resolution_var,
            values=("640x480", "1280x720", "1920x1080"),
            state="readonly",
            width=11,
        ).grid(row=0, column=3, padx=(4, 12))
        ttk.Label(secondary, textvariable=self.status_var).grid(row=0, column=4, sticky="w")

        content = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        content.grid(row=2, column=0, sticky="nsew")
        content.columnconfigure(0, weight=1)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(1, weight=1)
        ttk.Label(content, text="Webcam target (raw)").grid(row=0, column=0, sticky="w")
        ttk.Label(content, text="Processed result").grid(row=0, column=1, sticky="w")
        self.raw_label = ttk.Label(content, anchor="center", text="camera stopped")
        self.raw_label.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
        self.output_label = ttk.Label(content, anchor="center", text="start a model to preview")
        self.output_label.grid(row=1, column=1, sticky="nsew", padx=(6, 0))

        footer = ttk.Frame(self.root, padding=(12, 0, 12, 12))
        footer.grid(row=3, column=0, sticky="ew")
        footer.columnconfigure(0, weight=1)
        ttk.Label(footer, textvariable=self.caption_var).grid(row=0, column=0, sticky="w")
        ttk.Label(footer, textvariable=self.provider_var).grid(row=1, column=0, sticky="w")
        ttk.Label(footer, textvariable=self.metrics_var, justify="left").grid(
            row=2, column=0, sticky="w"
        )

    def choose_source(self) -> None:
        selected = filedialog.askopenfilename(
            title="Choose face-swap source image",
            initialdir=str(self.source_path.parent),
            filetypes=(("Images", "*.png *.jpg *.jpeg *.webp"),),
        )
        if selected:
            self.source_path = Path(selected)
            self.status_var.set(f"source changed: {self.source_path.name}; restart to apply")

    def start(self) -> None:
        self.stop()
        try:
            self.camera_index = int(self.camera_var.get())
            max_faces = int(self.max_faces_var.get())
            if not 1 <= max_faces <= 4:
                raise ValueError("Max faces must be between 1 and 4")
            self.swapper, provider_status = build_swapper(
                self.model_var.get(), self.source_path, self.model_path
            )
            self.provider_var.set(f"{provider_status} | {mps_status()}")
            self.camera = cv2.VideoCapture(self.camera_index)
            width, height = (int(value) for value in self.resolution_var.get().split("x"))
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if not self.camera.isOpened():
                raise RuntimeError(f"could not open camera {self.camera_index}")
        except Exception as error:
            self.stop()
            self.status_var.set(str(error))
            return
        self.stats = LoadStats.create()
        self.running = True
        self.status_var.set(f"running with {self.source_path.name}")
        self._tick()

    def stop(self) -> None:
        self.running = False
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.swapper = None

    def _tick(self) -> None:
        if not self.running or self.camera is None or self.swapper is None:
            return
        ok, frame = self.camera.read()
        if not ok:
            self.status_var.set("camera frame read failed")
            self.stop()
            return
        multiplier = int(self.session_var.get())
        max_faces = int(self.max_faces_var.get())
        started = time.perf_counter()
        output = frame
        detected = 0
        try:
            for _ in range(multiplier):
                output, detected = self.swapper.swap(frame, max_faces=max_faces)
        except Exception as error:
            self.status_var.set(f"processing error: {error}")
            self.stop()
            return
        batch_ms = (time.perf_counter() - started) * 1_000
        self.stats.add(batch_ms, multiplier, detected)
        self.last_raw = frame
        self.last_output = output
        self._show_frame(self.raw_label, frame, output=False)
        self._show_frame(self.output_label, output, output=True)
        self.caption_var.set(preview_caption(self.model_var.get(), multiplier))
        self.metrics_var.set(self.stats.summary(multiplier))
        self.root.after(1, self._tick)

    def _show_frame(self, label: ttk.Label, frame: np.ndarray, *, output: bool) -> None:
        display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(display)
        image.thumbnail((620, 460), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo, text="")
        if output:
            self.output_photo = photo
        else:
            self.raw_photo = photo

    def save_pair(self) -> None:
        if self.last_raw is None or self.last_output is None:
            self.status_var.set("start the camera before saving")
            return
        output_dir = ROOT / "face_swap_lab_output"
        output_dir.mkdir(exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        raw_path = output_dir / f"{stamp}-raw.png"
        output_path = output_dir / f"{stamp}-processed.png"
        cv2.imwrite(str(raw_path), self.last_raw)
        cv2.imwrite(str(output_path), self.last_output)
        self.status_var.set(f"saved {raw_path.name} and {output_path.name}")

    def close(self) -> None:
        self.stop()
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--camera", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    FaceSwapLabApp(
        root,
        source_path=args.source.expanduser().resolve(),
        model_path=args.model_path.expanduser().resolve(),
        camera=args.camera,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
