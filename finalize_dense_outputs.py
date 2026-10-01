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
from colmap_workspace_check import file_identity, write_provenance
from dense_cache import read_record, reconstruction_binding


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
    parser.add_argument(
        "--allow-raw-replacement",
        action="store_true",
        help="Explicitly allow raw fused.ply to replace an already valid final.ply.",
    )
    return parser.parse_args()


def validate_filter_audit(path: Path, vertex_count: int) -> None:
    """Reject malformed or poor-quality filter receipts before promotion."""
    audit_path = path.parent / "point_filter_summary.json"
    if not audit_path.exists():
        return
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if not isinstance(audit, dict):
        raise ValueError("Point filter summary must be a JSON object")
    output = audit.get("output", "")
    if not isinstance(output, str):
        raise ValueError("Point filter summary output must be a path string")
    if Path(output).resolve() != path.resolve() or audit.get("output_vertex_count") != vertex_count:
        return
    if not isinstance(audit.get("sampling", {}), dict):
        raise ValueError("Point filter sampling summary must be a JSON object")
    try:
        warning = sampling_quality_warning(audit)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"Invalid point filter sampling summary: {exc}") from exc
    if warning:
        raise ValueError(warning)


def choose_source(
    dense_dir: Path,
    prefer_mesh: bool = False,
    rejected: Optional[list[dict[str, str]]] = None,
    allow_raw_replacement: bool = False,
) -> Optional[tuple[str, Path]]:
    """Select the first valid source, protecting an existing final from raw fallback."""
    priority = MESH_SOURCE_PRIORITY if prefer_mesh else OBSERVED_SOURCE_PRIORITY
    for kind, name in priority:
        path = dense_dir / name
        if not path.is_file():
            continue
        try:
            elements = validate_ply(path)
            if kind.endswith("_mesh") and elements.get("face", 0) <= 0:
                raise ValueError("Mesh has no faces")
            if kind == "filtered_point_cloud":
                validate_filter_audit(path, elements["vertex"])
        except (OSError, ValueError) as exc:
            if rejected is not None:
                rejected.append({"source": str(path), "reason": str(exc)})
            continue
        if kind == "raw_point_cloud" and not allow_raw_replacement:
            final_path = dense_dir / FINAL_NAME
            if final_path.is_file():
                try:
                    validate_ply(final_path)
                except (OSError, ValueError):
                    pass
                else:
                    # Cleanup retains raw fusion; its presence alone does not
                    # mean a new reconstruction has produced a replacement.
                    return "existing_final", final_path
        return kind, path
    return None


def write_summary(path: Optional[Path], payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_provenance(path, payload)


def commit_final(source: Path, final: Path, summary: Path, payload: dict[str, Any]) -> None:
    """Commit the PLY and receipt together, retaining recovery files on rollback failure."""
    if summary.resolve() in {source.resolve(), final.resolve()}:
        raise ValueError("Final summary must not overwrite a PLY source or output")
    summary.parent.mkdir(parents=True, exist_ok=True)
    transaction = Path(tempfile.mkdtemp(dir=final.parent, prefix="final-transaction-"))
    with tempfile.NamedTemporaryFile(dir=summary.parent, prefix="final-summary-", suffix=".tmp", delete=False) as handle:
        staged_summary = Path(handle.name)
    with tempfile.NamedTemporaryFile(dir=summary.parent, prefix="final-summary-backup-", suffix=".tmp", delete=False) as handle:
        old_summary = Path(handle.name)
    staged, backup = transaction / "new.ply", transaction / "previous.ply"
    had_final, had_summary = final.exists(), summary.exists()
    replacing_final = replacing_summary = False
    keep_recovery = False
    try:
        shutil.copy2(source, staged)
        elements = validate_ply(staged)
        payload.update(vertex_count=elements["vertex"], face_count=elements.get("face"),
                       final_identity=file_identity(staged))
        if had_final:
            shutil.copy2(final, backup)
            if file_identity(backup) != file_identity(final):
                raise ValueError("Final backup changed during export")
        if had_summary:
            shutil.copy2(summary, old_summary)
        write_summary(staged_summary, payload)
        replacing_final = True
        staged.replace(final)
        replacing_summary = True
        staged_summary.replace(summary)
    except BaseException:
        try:
            if replacing_final:
                if had_final:
                    backup.replace(final)
                else:
                    final.unlink(missing_ok=True)
            if replacing_summary:
                if had_summary:
                    old_summary.replace(summary)
                else:
                    summary.unlink(missing_ok=True)
        except OSError as rollback_error:
            keep_recovery = True
            raise OSError(f"Final rollback failed; retain {transaction} and {old_summary} for recovery") from rollback_error
        raise
    finally:
        if not keep_recovery:
            shutil.rmtree(transaction)
            staged_summary.unlink(missing_ok=True)
            old_summary.unlink(missing_ok=True)


def cleanup_intermediate_files(dense_dir: Path, final_path: Path) -> list[str]:
    """Remove intermediate PLYs, retaining raw fusion and its visibility evidence."""
    deleted: list[str] = []
    final_path = final_path.resolve()
    for name in sorted({name for _kind, name in MESH_SOURCE_PRIORITY}):
        if name == "fused.ply":
            continue
        path = dense_dir / name
        if not path.is_file():
            continue
        if path.resolve() == final_path:
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
    args.summary = args.summary or dense_dir / "final_output_summary.json"
    if args.summary.resolve() in {(dense_dir / name).resolve() for _, name in MESH_SOURCE_PRIORITY}:
        print("[error] Final summary must not overwrite a PLY source or output")
        return 1
    final_path = dense_dir / FINAL_NAME
    previous = read_record(args.summary)
    rejected: list[dict[str, str]] = []
    selected = choose_source(
        dense_dir, prefer_mesh=args.prefer_mesh, rejected=rejected,
        allow_raw_replacement=args.allow_raw_replacement,
    )
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
        write_summary(args.summary, {**previous, "last_finalization_error": payload} if previous else payload)
        print("[error] New dense PLY sources were invalid; the previous final and source files were retained.")
        return 1
    try:
        binding = reconstruction_binding(dense_dir) if source_kind != "existing_final" else None
    except (OSError, ValueError) as exc:
        print(f"[error] Cannot bind final output to its reconstruction: {exc}")
        return 1
    if previous and final_path.is_file():
        identity = file_identity(final_path)
        unchanged = previous.get("final_identity") == identity
        same_observed_source = (
            previous.get("source_kind") == "videoscene_completed_point_cloud" and
            source_kind == "filtered_point_cloud" and binding and binding == previous.get("reconstruction_binding") and
            file_identity(source_path) == previous.get("observed_identity")
        )
        if source_kind == "existing_final" or (unchanged and same_observed_source):
            if previous.get("final_identity") and not unchanged:
                print("[error] Existing final changed since its provenance record; refusing to relabel it as observed geometry.")
                return 1
            if validate_ply(final_path)["vertex"] != previous.get("vertex_count"):
                print("[error] Existing final vertex count differs from its source record.")
                return 1
            write_summary(args.summary, {**previous, "action": "kept"})
            print(f"[done] Existing final and its source provenance retained: {final_path}")
            return 0
    replacing = source_path.resolve() != final_path.resolve()
    cleanup = not args.keep_intermediate and source_kind != "existing_final"
    payload = {
        "status": "ok",
        "output": str(final_path),
        "source": str(source_path),
        "source_kind": source_kind,
        "action": "copied_and_replaced" if replacing else "kept",
        "keep_intermediate": bool(args.keep_intermediate),
        "raw_fused_preserved": (dense_dir / "fused.ply").is_file(),
        "prefer_mesh": bool(args.prefer_mesh),
        "rejected_sources": rejected,
        "deleted_intermediate_files": [],
        "cleanup_status": "pending" if cleanup else "retained",
        "reconstruction_binding": binding,
        "explanation": (
            "final.ply defaults to the filtered observed point cloud; use --prefer-mesh "
            "only when a Poisson/Delaunay visualization mesh is explicitly desired. "
            "An existing valid final is preserved instead of replacing it with raw fused.ply "
            "unless --allow-raw-replacement is explicitly supplied."
        ),
    }
    try:
        if replacing:
            commit_final(source_path, final_path, args.summary, payload)
        else:
            elements = validate_ply(final_path)
            payload.update(vertex_count=elements["vertex"], face_count=elements.get("face"),
                           final_identity=file_identity(final_path))
            write_summary(args.summary, payload)
    except (OSError, ValueError) as exc:
        print(f"[error] Final export failed; intermediate sources were retained: {exc}")
        return 1
    # Cleanup is optional maintenance after both durable outputs are committed.
    deleted = cleanup_intermediate_files(dense_dir, final_path) if cleanup else []
    if cleanup:
        payload.update(deleted_intermediate_files=deleted, cleanup_status="finished")
        try:
            write_summary(args.summary, payload)
        except OSError as exc:
            print(f"[warn] Final and provenance committed; cleanup receipt remains pending: {exc}")
    print(f"[done] Final PLY: {final_path}")
    if deleted:
        print("[done] Removed intermediate dense artifacts: " + ", ".join(deleted))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
