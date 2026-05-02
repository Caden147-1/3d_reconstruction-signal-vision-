"""Convenient launcher for the classroom raw data preprocessing pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

from preprocess_pipeline import main as run_pipeline


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_ARGS = [
    "--root",
    str(PROJECT_ROOT),
    "--output",
    "processed",
    "--pic-dir",
    "pic",
    "--images-per-wifi-point",
    "20",
    "--video-dir",
    "video",
    "--point-event",
    "CHECKPOINT_PAUSE",
]


def main() -> int:
    return run_pipeline([*DEFAULT_ARGS, *sys.argv[1:]])


if __name__ == "__main__":
    raise SystemExit(main())
