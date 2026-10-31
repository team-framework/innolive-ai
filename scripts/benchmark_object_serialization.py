#!/usr/bin/env python3
"""Compare CPU object serialization with a local Git revision, without loading a model."""

from __future__ import annotations

import argparse
import ast
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from service import runtime as runtime_module


def baseline_method(ref: str):
    revision = subprocess.run(
        ["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source = subprocess.run(
        ["git", "show", f"{revision}:service/runtime.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    manager = next(
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.ClassDef) and node.name == "RuntimeManager"
    )
    method = next(
        node
        for node in manager.body
        if isinstance(node, ast.FunctionDef) and node.name == "_objects"
    )
    namespace = vars(runtime_module).copy()
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), f"{revision}:_objects", "exec"),
        namespace,
    )
    return revision, namespace["_objects"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-ref", required=True)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if min(args.iterations, args.rounds) < 1:
        parser.error("iterations and rounds must be positive")
    revision, baseline = baseline_method(args.baseline_ref)
    manager = runtime_module.RuntimeManager.__new__(runtime_module.RuntimeManager)
    manager.names = {0: "face", 1: "number_plate"}
    candidate = runtime_module.RuntimeManager._objects
    cases = []
    for count in (1, 8, 32):
        angles = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        polygons = [
            np.column_stack((160 + 100 * np.cos(angles), 120 + 80 * np.sin(angles))).astype(
                np.float32
            )
            for _ in range(count)
        ]
        tracks = np.asarray(
            [[60, 40, 260, 200, index + 1, 0.95, index % 2, index] for index in range(count)],
            dtype=np.float32,
        )
        prediction = SimpleNamespace(masks=SimpleNamespace(xy=polygons), boxes=[None] * count)
        inputs = manager, prediction, tracks, 640, 480
        if baseline(*inputs) != candidate(*inputs):
            raise RuntimeError(f"output mismatch for {count} objects")
        samples = {"baseline": [], "candidate": []}
        functions = [("baseline", baseline), ("candidate", candidate)]
        for _, function in functions:
            for _ in range(25):
                function(*inputs)
        for round_index in range(args.rounds):
            for name, function in functions[:: 1 if round_index % 2 == 0 else -1]:
                started = time.perf_counter()
                for _ in range(args.iterations):
                    function(*inputs)
                samples[name].append((time.perf_counter() - started) * 1000 / args.iterations)
        old, new = (statistics.median(samples[name]) for name in ("baseline", "candidate"))
        cases.append(
            {
                "objects": count,
                "points_per_object": 64,
                "baseline_ms": old,
                "candidate_ms": new,
                "speedup": old / new,
                "rounds_ms": samples,
            }
        )
    print(
        json.dumps(
            {
                "scope": "CPU _objects only; no GPU, transport or rendering",
                "baseline": revision,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "platform": platform.platform(),
                "iterations": args.iterations,
                "rounds": args.rounds,
                "cases": cases,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
