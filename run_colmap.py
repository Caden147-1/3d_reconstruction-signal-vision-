#!/usr/bin/env python3
"""Run the generated COLMAP workspace scripts from the project root."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


SCRIPT_BY_MODE = {
    "sparse": "run_colmap_sparse.bat",
    "dense": "run_colmap_dense.bat",
    "full": "run_colmap_full.bat",
    "automatic": "run_colmap_automatic.bat",
    "gui": "open_colmap_gui.bat",
    "reset": "reset_colmap_outputs.bat",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=SCRIPT_BY_MODE, nargs="?", default="sparse")
    parser.add_argument("--workspace", type=Path, default=Path("colmap_workspace"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    workspace = args.workspace if args.workspace.is_absolute() else root / args.workspace
    script = workspace / SCRIPT_BY_MODE[args.mode]
    if not script.exists():
        raise SystemExit(f"COLMAP script not found: {script}. Run `python main.py` first.")
    return subprocess.call(["cmd", "/c", str(script)], cwd=str(workspace))


if __name__ == "__main__":
    raise SystemExit(main())
