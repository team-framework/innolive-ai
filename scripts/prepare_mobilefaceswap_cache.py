#!/usr/bin/env python3
"""Precompute exact FP32 preset-conditioned weights off the live video path."""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def initialize(model_dir, cache_dir):
    os.environ["OMP_NUM_THREADS"] = "1"
    import torch

    from service.mobilefaceswap import MobileFaceSwapRenderer
    from service.mobilefaceswap_torch import ConditionedModelCache

    global renderer, cache
    torch.set_num_threads(1)
    renderer = MobileFaceSwapRenderer(model_dir)
    cache = ConditionedModelCache(cache_dir, model_dir)


def prepare(preset):
    from service.mobilefaceswap_torch import from_paddle

    if cache.load(preset) is not None:
        return preset.key, "cached"
    cache.save(preset, from_paddle(renderer.source_model(preset)))
    if cache.load(preset) is None:
        raise RuntimeError(f"cache verification failed: {preset.key}")
    return preset.key, "prepared"


if __name__ == "__main__":
    from service.face_presets import DEFAULT_PRESET_MANIFEST, PresetCatalog

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, default=Path("models/mobilefaceswap"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_PRESET_MANIFEST)
    parser.add_argument("--output", type=Path, default=Path(".runtime/mobilefaceswap-cache"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1..8")
    presets = list(PresetCatalog(args.manifest).presets.values())
    started = time.monotonic()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        mp_context=multiprocessing.get_context("spawn"),
        initializer=initialize,
        initargs=(args.models.resolve(), args.output.resolve()),
    ) as pool:
        for index, (key, status) in enumerate(pool.map(prepare, presets), 1):
            if index % 20 == 0 or index == len(presets):
                print(f"{index}/{len(presets)} {status}: {key}", flush=True)
    print(f"Verified {len(presets)} presets in {time.monotonic() - started:.1f}s", flush=True)
