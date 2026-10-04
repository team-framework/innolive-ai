#!/usr/bin/env python3
"""Compare serving tensors/logits with an explicitly supplied training common.py.

Cases JSON: {"cases": [{"id": "face-1", "image": "input.jpg", "bbox": [x1,y1,x2,y2]}]}.
Image paths are relative to the cases file. This CPU FP32 comparison does not measure accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from service.face_metadata import (
    CLASSES,
    DEFAULT_METADATA_MODEL,
    MetadataExtractor,
    crop_metadata_face,
)


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import reference: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def comparison(reference, candidate, *, atol: float, rtol: float) -> dict:
    heads = {}
    for name in CLASSES:
        left, right = reference[name], candidate[name]
        lp, rp = left.softmax(1), right.softmax(1)
        heads[name] = {
            "max_abs_logit_difference": float((left - right).abs().max()),
            "max_abs_probability_difference": float((lp - rp).abs().max()),
            "different_top1": int((lp.argmax(1) != rp.argmax(1)).sum()),
            "logits_close": bool(torch.allclose(left, right, atol=atol, rtol=rtol)),
        }
    return heads


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict:
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    reference = load_module(args.training_source, "metadata_training_reference")
    if tuple(reference.TARGETS) != tuple(CLASSES) or any(
        tuple(reference.CLASSES[name]) != values for name, values in CLASSES.items()
    ):
        raise ValueError("training class order does not match serving")
    serving = MetadataExtractor(args.checkpoint)
    if serving.input_size != reference.SIZE:
        raise ValueError("training input size does not match serving")
    trained = reference.MetadataLarge().to("cpu", memory_format=torch.channels_last).eval()
    trained.load_state_dict(serving.model.state_dict(), strict=True)
    baseline = None
    if args.baseline_source:
        old = load_module(args.baseline_source, "metadata_integration_baseline")
        baseline = old.MetadataExtractor(args.checkpoint, "cpu")
    cases = json.loads(args.cases.read_text())["cases"]
    if not cases:
        raise ValueError("comparison requires at least one face")
    logits = {name: [] for name in ("reference", "serving", "baseline", "head_only", "crop_only")}
    records = []
    tensor_error = 0.0
    for start in range(0, len(cases), args.batch_size):
        group = cases[start : start + args.batch_size]
        training_inputs, serving_crops, tight_crops = [], [], []
        for case in group:
            image_path = args.cases.parent / case["image"]
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"cannot read image: {image_path}")
            bbox = np.asarray(case["bbox"], dtype=float)
            expanded = crop_metadata_face(image, bbox)
            x1, y1, x2, y2 = bbox
            crop, _ = reference.crop_face(image, [x1, y1, x2 - x1, y2 - y1])
            rgb = np.ascontiguousarray(crop[:, :, ::-1].transpose(2, 0, 1))
            training_inputs.append(torch.from_numpy(rgb))
            serving_crops.append(expanded)
            if baseline:
                left, top = np.maximum(np.floor(bbox[:2]), 0).astype(int)
                right, bottom = np.minimum(
                    np.ceil(bbox[2:]), [image.shape[1], image.shape[0]]
                ).astype(int)
                tight_crops.append(image[top:bottom, left:right])
            records.append(
                {"id": case["id"], "image_sha256": digest(image_path), "bbox_xyxy": bbox.tolist()}
            )
        expected_inputs = reference.normalize(torch.stack(training_inputs), torch.device("cpu"))
        actual_inputs = serving.prepare_inputs(serving_crops)
        tensor_error = max(tensor_error, float((expected_inputs - actual_inputs).abs().max()))
        logits["reference"].append(trained(expected_inputs))
        logits["serving"].append(serving.model(actual_inputs))
        if baseline:
            captured = {}

            def capture_inputs(_model, inputs, state=captured):
                state["inputs"] = inputs[0]

            def capture_logits(_model, _inputs, outputs, state=captured):
                state["logits"] = outputs

            before = baseline.model.register_forward_pre_hook(capture_inputs)
            after = baseline.model.register_forward_hook(capture_logits)
            try:
                baseline.predict(tight_crops)
            finally:
                before.remove()
                after.remove()
            logits["baseline"].append(captured["logits"])
            # Separate the head mismatch from the crop/resize mismatch.
            logits["head_only"].append(baseline.model(expected_inputs))
            logits["crop_only"].append(trained(captured["inputs"]))
    combined = {
        variant: {name: torch.cat([batch[name] for batch in batches]) for name in CLASSES}
        for variant, batches in logits.items()
        if batches
    }
    comparisons = {
        variant: comparison(combined["reference"], values, atol=args.atol, rtol=args.rtol)
        for variant, values in combined.items()
        if variant != "reference"
    }
    for index, record in enumerate(records):
        record["predictions"] = {
            variant: {
                name: {
                    "label": CLASSES[name][int(values[name][index].argmax())],
                    "probabilities": values[name][index].softmax(0).tolist(),
                }
                for name in CLASSES
            }
            for variant, values in combined.items()
        }
    return {
        "schema_version": 1,
        "sample_count": len(cases),
        "device": "cpu",
        "precision": "float32",
        "training_source_sha256": digest(args.training_source),
        "checkpoint_sha256": digest(args.checkpoint),
        "baseline_source_sha256": digest(args.baseline_source) if args.baseline_source else None,
        "torch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "atol": args.atol,
        "rtol": args.rtol,
        "max_abs_input_tensor_difference": tensor_error,
        "comparisons": comparisons,
        "parity_passed": tensor_error <= args.atol
        and all(head["logits_close"] for head in comparisons["serving"].values()),
        "cases": records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training-source", type=Path, required=True, help="training snapshot common.py"
    )
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_METADATA_MODEL)
    parser.add_argument("--baseline-source", type=Path, help="optional pre-fix face_metadata.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.atol < 0 or args.rtol < 0:
        parser.error("batch-size must be positive; tolerances must be nonnegative")
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "cases"}, ensure_ascii=False
        )
    )
    return 0 if report["parity_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
