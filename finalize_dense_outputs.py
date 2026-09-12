#!/usr/bin/env python3
"""Select the final dense PLY and remove intermediate dense artifacts."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Optional

from ply_io import validate_ply
from point_cloud_filter import sampling_quality_warning


FINAL_NAME = "final.ply"
OBSERVED_SOURCE_PRIORITY = (
    ("filtered_point_cloud", "fused_filtered.ply"),
    ("raw_point_cloud", "fused.ply"),
    ("poisson_mesh", "meshed-poisson.ply"),
    ("delaunay_mesh", "meshed-delaunay.ply"),
    ("existing_final", FINAL_NAME),
)
MESH_SOURCE_PRIORITY = (
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
    parser.add_argument(
        "--prefer-mesh",
        action="store_true",
        help="Promote a Poisson/Delaunay mesh instead of the observed filtered point cloud.",
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


def choose_source(dense_dir: Path, prefer_mesh: bool = False, rejected: Optional[list[dict[str, str]]] = None) -> Optional[tuple[str, Path]]:
    priority = MESH_SOURCE_PRIORITY if prefer_mesh else OBSERVED_SOURCE_PRIORITY
    for kind, name in priority:
        path = dense_dir / name
        if path.exists() and path.is_file():
            try:
                elements = validate_ply(path)
                if kind.endswith("_mesh") and elements.get("face", 0) <= 0:
                    raise ValueError("Mesh has no faces")
                audit_path = dense_dir / "point_filter_summary.json"
                if kind == "filtered_point_cloud" and audit_path.exists():
                    audit = json.loads(audit_path.read_text(encoding="utf-8"))
                    if Path(audit.get("output", "")).resolve() == path.resolve() and audit.get("output_vertex_count") == elements["vertex"]:
                        warning = sampling_quality_warning(audit)
                        if warning:
                            raise ValueError(warning)
            except (OSError, ValueError) as exc:
                if rejected is not None:
                    rejected.append({"source": str(path), "reason": str(exc)})
                continue
            return kind, path
    return None


def write_summary(path: Optional[Path], payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def cleanup_intermediate_files(dense_dir: Path, final_path: Path) -> list[str]:
    deleted: list[str] = []
    for name in sorted({name for _kind, name in MESH_SOURCE_PRIORITY}):
        if name == "fused.ply":
            continue
        path = dense_dir / name
        if not path.is_file():
            continue
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
    rejected: list[dict[str, str]] = []
    selected = choose_source(dense_dir, prefer_mesh=args.prefer_mesh, rejected=rejected)
    if selected is None:
        payload = {
            "status": "missing_source",
            "output": str(final_path),
            "rejected_sources": rejected,
            "explanation": "No complete, non-empty dense PLY source was available; existing files were retained.",
        }
        write_summary(args.summary, payload)
        print("[error] No valid dense PLY source was available for final.ply; files were retained.")
        return 1

    source_kind, source_path = selected
    if source_kind == "existing_final" and rejected:
        payload = {
            "status": "no_valid_new_source",
            "output": str(final_path),
            "rejected_sources": rejected,
            "explanation": "All new dense PLY sources were invalid; the previous final and source files were retained without cleanup.",
        }
        write_summary(args.summary, payload)
        print("[error] New dense PLY sources were invalid; the previous final and source files were retained.")
        return 1
    action = "kept"
    elements = read_ply_elements(source_path)
    if source_path.resolve() != final_path.resolve():
        with tempfile.NamedTemporaryFile(dir=dense_dir, prefix="final-", suffix=".tmp", delete=False) as handle:
            staged = Path(handle.name)
        try:
            shutil.copy2(source_path, staged)
            elements = validate_ply(staged)
            staged.replace(final_path)
            action = "copied_and_replaced"
        except (OSError, ValueError) as exc:
            print(f"[error] Final PLY promotion failed; source files were retained: {exc}")
            return 1
        finally:
            staged.unlink(missing_ok=True)

    deleted = [] if args.keep_intermediate else cleanup_intermediate_files(dense_dir, final_path)
    payload = {
        "status": "ok",
        "output": str(final_path),
        "source": str(source_path),
        "source_kind": source_kind,
        "action": action,
        "keep_intermediate": bool(args.keep_intermediate),
        "raw_fused_preserved": (dense_dir / "fused.ply").is_file(),
        "prefer_mesh": bool(args.prefer_mesh),
        "rejected_sources": rejected,
        "deleted_intermediate_files": deleted,
        "vertex_count": elements.get("vertex"),
        "face_count": elements.get("face"),
        "explanation": (
            "final.ply defaults to the filtered observed point cloud; use --prefer-mesh "
            "only when a Poisson/Delaunay visualization mesh is explicitly desired."
        ),
    }
    write_summary(args.summary, payload)
    print(f"[done] Final PLY: {final_path}")
    if deleted:
        print("[done] Removed intermediate dense artifacts: " + ", ".join(deleted))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
