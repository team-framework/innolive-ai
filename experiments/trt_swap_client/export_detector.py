#!/usr/bin/env python3
"""Build the RF-DETR detector shared by the server and swap lab."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.export_tensorrt import main

if __name__ == "__main__":
    main()
