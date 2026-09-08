#!/usr/bin/env python3
"""Select the final dense PLY and remove intermediate dense artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Optional


FINAL_NAME = "final.ply"
SOURCE_PRIORITY = (
    ("poisson_mesh", "meshed-poisson.ply"),
    ("delaunay_mesh", "meshed-delaunay.ply"),
    ("filtered_point_cloud", "fused_filtered.ply"),
    ("raw_point_cloud", "fused.ply"),
    ("existing_final", FINAL_NAME),
)
AUXILIARY_ARTIFACTS = ("fused.ply.vis",)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--keep-intermediate",
        action="store_true",
        help="Keep source/intermediate PLY files after final.ply is selected.",
    )
    return parser.parse_args()


def read_ply_elements(path: Path) -> dict[str, int]:
    elements: dict[str, int] = {}
    try:
        with path.open("rb") as handle:
            header = handle.read(65536).decode("ascii", errors="ignore")
    except OSError:
        return elements

    for line in header.splitlines():
        parts = line.strip().split()
        if len(parts) == 3 and parts[0] == "element":
            try:
                elements[parts[1]] = int(parts[2])
            except ValueError:
                pass
        if line.strip() == "end_header":
            break
    return elements


def choose_source(dense_dir: Path) -> Optional[tuple[str, Path]]:
    for kind, name in SOURCE_PRIORITY:
        path = dense_dir / name
        if path.exists() and path.is_file():
            return kind, path
    return None


def write_summary(path: Optional[Path], payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cleanup_intermediate_files(dense_dir: Path, final_path: Path) -> list[str]:
    deleted: list[str] = []
    for path in sorted(dense_dir.glob("*.ply")):
        if path.resolve() == final_path.resolve():
            continue
        try:
            path.unlink()
        except OSError:
            continue
        deleted.append(path.name)

    for name in AUXILIARY_ARTIFACTS:
        path = dense_dir / name
        if not path.exists():
            continue
        try:
            path.unlink()
        except OSError:
            continue
        deleted.append(path.name)
    return deleted


def main() -> int:
    args = parse_args()
    dense_dir = args.dense_dir
    dense_dir.mkdir(parents=True, exist_ok=True)
    final_path = dense_dir / FINAL_NAME
    selected = choose_source(dense_dir)
    if selected is None:
        payload = {
            "status": "missing_source",
            "output": str(final_path),
            "explanation": "No dense PLY source was available to promote to final.ply.",
        }
        write_summary(args.summary, payload)
        print("[error] No dense PLY source was available for final.ply.")
        return 1

    source_kind, source_path = selected
    action = "kept"
    if source_path.resolve() != final_path.resolve():
        if final_path.exists():
            final_path.unlink()
        if args.keep_intermediate:
            shutil.copy2(source_path, final_path)
            action = "copied"
        else:
            shutil.move(str(source_path), str(final_path))
            action = "moved"

    deleted = [] if args.keep_intermediate else cleanup_intermediate_files(dense_dir, final_path)
    elements = read_ply_elements(final_path)
    payload = {
        "status": "ok",
        "output": str(final_path),
        "source": str(source_path),
        "source_kind": source_kind,
        "action": action,
        "keep_intermediate": bool(args.keep_intermediate),
        "deleted_intermediate_files": deleted,
        "vertex_count": elements.get("vertex"),
        "face_count": elements.get("face"),
        "explanation": "final.ply is the single dense model to open for normal inspection.",
    }
    write_summary(args.summary, payload)
    print(f"[done] Final PLY: {final_path}")
    if deleted:
        print("[done] Removed intermediate dense artifacts: " + ", ".join(deleted))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
