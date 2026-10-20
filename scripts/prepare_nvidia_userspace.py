#!/usr/bin/env python3
"""Extract matching NVIDIA libraries for this service without installing drivers.

Use when a package upgrade changed userspace while the running kernel module is
still older. Only official NVIDIA packages are downloaded; no sudo or reboot.
The CUDA launcher selects the directory matching the currently loaded module.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
from pathlib import Path

BASE = "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/"
PACKAGES = {"libnvidia-compute", "libnvidia-encode", "libnvidia-decode", "libnvidia-gpucomp"}


def prepare(root: Path) -> None:
    version = Path("/sys/module/nvidia/version").read_text().strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", version):
        raise ValueError("cannot identify the loaded NVIDIA module")
    destination = root / version
    if destination.exists():
        raise FileExistsError(
            f"already exists: {destination}; preserve or move it before preparing"
        )
    print(f"Preparing service-local NVIDIA {version} libraries", flush=True)
    with urllib.request.urlopen(BASE + "Packages.gz", timeout=60) as response:
        index = gzip.decompress(response.read()).decode()
    packages = {}
    for paragraph in index.split("\n\n"):
        row = dict(
            line.split(": ", 1)
            for line in paragraph.splitlines()
            if ": " in line and not line.startswith(" ")
        )
        if (
            row.get("Package") in PACKAGES
            and row.get("Version", "").startswith(version + "-")
            and row.get("Architecture") == "amd64"
        ):
            packages[row["Package"]] = row
    if packages.keys() != PACKAGES:
        raise RuntimeError(f"official packages unavailable for {version}; no files installed")
    root.mkdir(parents=True, exist_ok=True)
    records = []
    with tempfile.TemporaryDirectory(prefix="prepare-", dir=root) as temporary:
        stage = Path(temporary)
        unpacked = stage / "unpacked"
        unpacked.mkdir()
        for name, row in sorted(packages.items()):
            filename = Path(row["Filename"]).name
            target = stage / filename
            print(f"Downloading {name} ({int(row['Size']) / 1e6:.1f} MB)", flush=True)
            with (
                urllib.request.urlopen(BASE + filename, timeout=120) as response,
                target.open("wb") as output,
            ):
                shutil.copyfileobj(response, output)
            with target.open("rb") as downloaded:
                digest = hashlib.file_digest(downloaded, "sha256").hexdigest()
            if digest != row["SHA256"] or target.stat().st_size != int(row["Size"]):
                raise ValueError(f"checksum or size mismatch: {name}")
            subprocess.run(["dpkg-deb", "-x", str(target), str(unpacked)], check=True)
            records.append({"url": BASE + filename, "sha256": digest, "package": name})
        (unpacked / "manifest.json").write_text(
            json.dumps({"version": version, "packages": records}, indent=2) + "\n"
        )
        unpacked.rename(destination)
    print(f"Ready: {destination / 'usr/lib/x86_64-linux-gnu'}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".runtime/nvidia"))
    prepare(parser.parse_args().root.resolve())
