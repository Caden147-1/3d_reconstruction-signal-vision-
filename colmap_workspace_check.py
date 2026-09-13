#!/usr/bin/env python3
"""Validate that a COLMAP workspace still matches its current input manifest.

The check is intentionally read-only. It prevents a dense run from silently
using a database or sparse model created before new photo/video inputs were
prepared.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sqlite3
import struct
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set


PROVENANCE_NAME = "sparse_run_provenance.json"
PENDING_NAME = "sparse_run_inputs.json"
SPARSE_RUNNERS = {"run_colmap_sparse.bat", "run_colmap_automatic.bat"}


def file_identity(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"File changed during provenance inspection: {path}")
    return {"size": after.st_size, "sha256": digest.hexdigest()}


def contained_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"Input path escapes its root: {relative}")
    return path


def sparse_input_identity(workspace: Path, runner: str) -> Dict[str, Any]:
    if runner not in SPARSE_RUNNERS:
        raise ValueError("Unrecognized sparse runner")
    names = read_image_list(workspace / "image_list.txt")
    if not names or len(names) != len(set(names)) or not (workspace / runner).is_file():
        raise ValueError("Input list or sparse runner is missing or invalid")
    files: Dict[str, Any] = {}
    for name in sorted(names):
        identity = file_identity(contained_path(workspace / "images", name))
        if identity is None:
            raise ValueError(f"Missing input image: {name}")
        files[f"images/{name}"] = identity
    for name in ("image_list.txt", "match_pairs.txt", "pdr_alignment_refs.txt", "metadata/images_manifest.csv", runner):
        files[name] = file_identity(workspace / name)
    summary_path = workspace / "colmap_workspace_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    mask = summary.get("colmap_mask", {})
    mask_root = Path(mask["path"]) if mask.get("enabled") and mask.get("path") else None
    if mask_root is not None and not mask_root.is_dir():
        raise ValueError("Configured mask directory is missing")
    return {
        "runner": runner, "files": files,
        "mask_root": str(mask_root.resolve()) if mask_root is not None else None,
        "masks": {
            name: file_identity(contained_path(mask_root, name + ".png"))
            for name in sorted(names)
        } if mask_root is not None else {},
    }


def sparse_output_identity(workspace: Path) -> Dict[str, Any]:
    files = {name: file_identity(workspace / name) for name in ("database.db", "database.db-wal")}
    for folder in ("sparse", "sparse_pdr_aligned", "sparse_pose_prior"):
        base = workspace / folder
        if base.is_dir():
            for path in sorted(base.rglob("*")):
                if path.is_file() and path.suffix in {".bin", ".txt"}:
                    files[path.relative_to(workspace).as_posix()] = file_identity(path)
    selected = dense_input_model(workspace)
    return {"files": files, "dense_input": selected.relative_to(workspace).as_posix() if selected else None}


def inspect_provenance(workspace: Path) -> Dict[str, Any]:
    receipt = workspace / PROVENANCE_NAME
    if (workspace / PENDING_NAME).exists():
        return {"status": "incomplete_run", "error": "Sparse input snapshot exists without successful completion."}
    if not receipt.is_file():
        return {"status": "unverified", "error": "No run-time input provenance was recorded; filenames alone cannot verify old models."}
    try:
        recorded = json.loads(receipt.read_text(encoding="utf-8"))
        if recorded["schema_version"] != 1 or recorded["status"] != "complete":
            raise ValueError("Unsupported or incomplete provenance record")
        current_inputs = sparse_input_identity(workspace, recorded["inputs"]["runner"])
        if current_inputs != recorded["inputs"]:
            return {"status": "input_mismatch", "error": "Image bytes, masks, matching pairs, pose references, manifest or sparse runner changed."}
        if sparse_output_identity(workspace) != recorded["outputs"]:
            return {"status": "output_mismatch", "error": "The sparse model or database no longer matches the recorded run."}
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        return {"status": "invalid_record", "error": str(exc)}
    return {"status": "verified", "error": "", "record": str(receipt)}


def write_provenance(path: Path, payload: Dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="provenance-", suffix=".tmp", delete=False) as handle:
        staged = Path(handle.name)
    try:
        staged.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        staged.replace(path)
    finally:
        staged.unlink(missing_ok=True)


def begin_sparse_run(workspace: Path, runner: str = "run_colmap_sparse.bat") -> None:
    report = inspect_workspace(workspace)
    failures = validation_failures(report, require_clean_database=True)
    if any((workspace / name).exists() for name in ("sparse", "sparse_pdr_aligned", "sparse_pose_prior")):
        failures.append("Existing sparse directories must be reset before recording a new run")
    if any((workspace / ("database.db" + suffix)).exists() for suffix in ("-wal", "-shm", "-journal")):
        failures.append("Old SQLite sidecar files must be reset before recording a new run")
    if failures:
        raise ValueError("; ".join(failures))
    inputs = sparse_input_identity(workspace, runner)
    write_provenance(workspace / PENDING_NAME, {"schema_version": 1, "inputs": inputs})


def complete_sparse_run(workspace: Path) -> None:
    pending = json.loads((workspace / PENDING_NAME).read_text(encoding="utf-8"))
    if pending["schema_version"] != 1 or sparse_input_identity(workspace, pending["inputs"]["runner"]) != pending["inputs"]:
        raise ValueError("Sparse inputs changed during reconstruction; refusing to certify the result")
    failures = validation_failures(inspect_workspace(workspace), require_database_coverage=True,
                                   require_sparse_model=True, require_provenance=False)
    if failures:
        raise ValueError("; ".join(failures))
    write_provenance(workspace / PROVENANCE_NAME, {
        "schema_version": 1, "status": "complete", "inputs": pending["inputs"],
        "outputs": sparse_output_identity(workspace),
    })
    (workspace / PENDING_NAME).unlink()


def normalize_name(value: str) -> str:
    return value.strip().replace("\\", "/").lstrip("./")


def read_image_list(path: Path) -> List[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return [normalize_name(line) for line in handle if line.strip()]


def read_manifest_sources(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8", errors="replace") as handle:
        return {
            normalize_name(row.get("image_file", "")): row.get("source_type", "unknown") or "unknown"
            for row in csv.DictReader(handle)
            if row.get("image_file")
        }


def read_database_names(path: Path) -> tuple[Set[str], str]:
    if not path.exists():
        return set(), ""
    try:
        connection = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        try:
            rows = connection.execute("SELECT name FROM images").fetchall()
        finally:
            connection.close()
    except sqlite3.Error as exc:
        return set(), str(exc)
    return {normalize_name(str(row[0])) for row in rows}, ""


def read_exact(handle: Any, size: int) -> bytes:
    if size < 0:
        raise ValueError("Invalid COLMAP record length")
    raw = handle.read(size)
    if len(raw) != size:
        raise ValueError("Truncated COLMAP model")
    return raw


def read_binary_image_names(path: Path, camera_ids: Set[int] | None = None) -> Set[str]:
    names: Set[str] = set()
    ids: Set[int] = set()
    with path.open("rb") as handle:
        file_size = path.stat().st_size
        count = struct.unpack("<Q", read_exact(handle, 8))[0]
        for _ in range(count):
            record = struct.unpack("<idddddddi", read_exact(handle, 64))
            image_id, camera_id = record[0], record[-1]
            if image_id in ids or image_id <= 0 or camera_id <= 0 or not all(math.isfinite(v) for v in record[1:8]) or sum(v*v for v in record[1:5]) < 1e-12:
                raise ValueError("Invalid COLMAP image pose or ID")
            ids.add(image_id)
            if camera_ids is not None:
                camera_ids.add(camera_id)
            name = bytearray()
            while True:
                byte = read_exact(handle, 1)
                if byte == b"\0":
                    break
                name.extend(byte)
            decoded = normalize_name(name.decode("utf-8"))
            if not decoded or decoded in names:
                raise ValueError("Empty or duplicate COLMAP image name")
            names.add(decoded)
            points = struct.unpack("<Q", read_exact(handle, 8))[0]
            if points * 24 > file_size - handle.tell():
                raise ValueError("Truncated COLMAP image observations")
            handle.seek(points * 24, 1)
        if handle.read(1):
            raise ValueError("Extra data in COLMAP images.bin")
    return names


def read_text_image_names(path: Path, camera_ids: Set[int] | None = None) -> Set[str]:
    names: Set[str] = set()
    ids: Set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            parts = text.split(maxsplit=9)
            if len(parts) != 10:
                raise ValueError("Invalid COLMAP image header")
            image_id, camera_id = int(parts[0]), int(parts[8])
            pose = [float(value) for value in parts[1:8]]
            name = normalize_name(parts[9])
            if image_id in ids or image_id <= 0 or camera_id <= 0 or not name or name in names or not all(math.isfinite(v) for v in pose) or sum(v*v for v in pose[:4]) < 1e-12:
                raise ValueError("Invalid COLMAP image pose or name")
            ids.add(image_id)
            names.add(name)
            if camera_ids is not None:
                camera_ids.add(camera_id)
            observations = handle.readline()
            if not observations:
                raise ValueError("Missing COLMAP image observation row")
            fields = observations.split()
            if len(fields) % 3:
                raise ValueError("Invalid COLMAP image observations")
            for index in range(0, len(fields), 3):
                if not math.isfinite(float(fields[index])) or not math.isfinite(float(fields[index + 1])):
                    raise ValueError("Non-finite COLMAP image observation")
                int(fields[index + 2])
    return names


def model_has_images(model_dir: Path) -> bool:
    return (model_dir / "images.bin").exists() or (model_dir / "images.txt").exists()


def dense_input_model(workspace: Path) -> Path | None:
    aligned = workspace / "sparse_pdr_aligned" / "0"
    if (aligned / "cameras.bin").exists() or (aligned / "cameras.txt").exists():
        return aligned
    raw = workspace / "sparse" / "0"
    return raw if raw.exists() else None


# Camera layouts follow COLMAP's documented model IDs and binary format.
CAMERA_MODELS = [
    ("SIMPLE_PINHOLE", 3), ("PINHOLE", 4), ("SIMPLE_RADIAL", 4), ("RADIAL", 5),
    ("OPENCV", 8), ("OPENCV_FISHEYE", 8), ("FULL_OPENCV", 12), ("FOV", 5),
    ("SIMPLE_RADIAL_FISHEYE", 4), ("RADIAL_FISHEYE", 5), ("THIN_PRISM_FISHEYE", 12),
]


def inspect_sparse_model(model: Path) -> Dict[str, Any]:
    """Require complete image, camera and point records before dense execution."""
    binary = (model / "cameras.bin").exists()
    camera_ids: Set[int] = set()
    names = read_binary_image_names(model / "images.bin", camera_ids) if binary else read_text_image_names(model / "images.txt", camera_ids)
    cameras: Set[int] = set()

    def add_camera(camera_id: int, width: int, height: int, params: Sequence[float]) -> None:
        if camera_id <= 0 or camera_id in cameras or width <= 0 or height <= 0 or not params or params[0] <= 0 or not all(math.isfinite(v) for v in params):
            raise ValueError("Invalid COLMAP camera")
        cameras.add(camera_id)

    if binary:
        with (model / "cameras.bin").open("rb") as handle:
            count = struct.unpack("<Q", read_exact(handle, 8))[0]
            for _ in range(count):
                camera_id, model_id, width, height = struct.unpack("<iiQQ", read_exact(handle, 24))
                if not 0 <= model_id < len(CAMERA_MODELS):
                    raise ValueError(f"Unsupported COLMAP camera model ID: {model_id}")
                num_params = CAMERA_MODELS[model_id][1]
                add_camera(camera_id, width, height, struct.unpack("<" + "d" * num_params, read_exact(handle, 8 * num_params)))
            if handle.read(1):
                raise ValueError("Extra data in COLMAP cameras.bin")
        points_path = model / "points3D.bin"
        with points_path.open("rb") as handle:
            point_count = struct.unpack("<Q", read_exact(handle, 8))[0]
            file_size = points_path.stat().st_size
            for _ in range(point_count):
                record = struct.unpack("<QdddBBBd", read_exact(handle, 43))
                if not all(math.isfinite(v) for v in (*record[1:4], record[-1])):
                    raise ValueError("Non-finite COLMAP sparse point")
                tracks = struct.unpack("<Q", read_exact(handle, 8))[0]
                if tracks * 8 > file_size - handle.tell():
                    raise ValueError("Truncated COLMAP point tracks")
                handle.seek(tracks * 8, 1)
            if handle.read(1):
                raise ValueError("Extra data in COLMAP points3D.bin")
    else:
        camera_sizes = dict(CAMERA_MODELS)
        with (model / "cameras.txt").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 4 or parts[1] not in camera_sizes or len(parts) != 4 + camera_sizes[parts[1]]:
                    raise ValueError("Invalid COLMAP camera record")
                add_camera(int(parts[0]), int(parts[2]), int(parts[3]), [float(v) for v in parts[4:]])
        point_count = 0
        with (model / "points3D.txt").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 8 or (len(parts) - 8) % 2:
                    raise ValueError("Invalid COLMAP sparse point record")
                int(parts[0])
                if not all(math.isfinite(float(parts[i])) for i in (1, 2, 3, 7)):
                    raise ValueError("Non-finite COLMAP sparse point")
                for token in parts[4:7] + parts[8:]:
                    int(token)
                point_count += 1
    if not camera_ids <= cameras:
        raise ValueError("Sparse images reference missing cameras")
    if len(names) < 2 or not point_count:
        raise ValueError("Dense reconstruction requires registered views and sparse points")
    return {"names": names, "camera_count": len(cameras), "point_count": point_count}


def read_model_names(model_dir: Path | None) -> Set[str]:
    if model_dir is None:
        return set()
    binary = model_dir / "images.bin"
    return read_binary_image_names(binary) if binary.exists() else read_text_image_names(model_dir / "images.txt")


def source_coverage(
    expected_names: Iterable[str],
    source_by_name: Dict[str, str],
    observed_names: Set[str],
) -> Dict[str, Dict[str, Any]]:
    expected_by_source: Dict[str, Set[str]] = {}
    for name in expected_names:
        expected_by_source.setdefault(source_by_name.get(name, "unknown"), set()).add(name)
    result: Dict[str, Dict[str, Any]] = {}
    for source, names in sorted(expected_by_source.items()):
        missing = sorted(names - observed_names)
        result[source] = {
            "expected_count": len(names),
            "present_count": len(names) - len(missing),
            "missing_count": len(missing),
            "missing_examples": missing[:10],
        }
    return result


def inspect_workspace(workspace: Path) -> Dict[str, Any]:
    workspace = workspace.resolve()
    image_list = read_image_list(workspace / "image_list.txt")
    expected = set(image_list)
    images_dir = workspace / "images"
    actual = {
        normalize_name(str(path.relative_to(images_dir)))
        for path in images_dir.rglob("*")
        if path.is_file()
    } if images_dir.exists() else set()
    source_by_name = read_manifest_sources(workspace / "metadata" / "images_manifest.csv")
    database_names, database_error = read_database_names(workspace / "database.db")
    selected_model = dense_input_model(workspace)
    sparse_names: Set[str] = set()
    sparse_error = ""
    if selected_model is not None:
        try:
            sparse_names = inspect_sparse_model(selected_model)["names"]
        except (OSError, ValueError, struct.error) as exc:
            sparse_error = str(exc)
    return {
        "workspace": str(workspace),
        "input": {
            "listed_count": len(image_list),
            "unique_listed_count": len(expected),
            "directory_file_count": len(actual),
            "missing_files": sorted(expected - actual)[:20],
            "extra_files": sorted(actual - expected)[:20],
            "duplicate_list_entries": len(image_list) - len(expected),
            "source_coverage": source_coverage(expected, source_by_name, actual),
        },
        "database": {
            "exists": (workspace / "database.db").exists(),
            "error": database_error,
            "image_count": len(database_names),
            "missing_names": sorted(expected - database_names)[:20],
            "extra_names": sorted(database_names - expected)[:20],
            "source_coverage": source_coverage(expected, source_by_name, database_names),
        },
        "dense_input_sparse": {
            "path": "" if selected_model is None else str(selected_model),
            "exists": selected_model is not None,
            "error": sparse_error,
            "registered_image_count": len(sparse_names),
            "missing_names": sorted(expected - sparse_names)[:20],
            "extra_names": sorted(sparse_names - expected)[:20],
            "source_coverage": source_coverage(expected, source_by_name, sparse_names),
        },
        "provenance": inspect_provenance(workspace),
    }


def validation_failures(
    report: Dict[str, Any],
    *,
    require_clean_database: bool = False,
    require_database_coverage: bool = False,
    require_sparse_model: bool = False,
    require_provenance: bool = True,
) -> List[str]:
    failures: List[str] = []
    input_info = report["input"]
    database = report["database"]
    sparse = report["dense_input_sparse"]
    if not input_info["listed_count"]:
        failures.append("image_list.txt is empty or missing")
    if input_info["missing_files"]:
        failures.append("some image_list entries are missing from workspace/images")
    if input_info["duplicate_list_entries"]:
        failures.append("image_list.txt contains duplicate names")
    if require_clean_database and database["exists"]:
        failures.append("database.db already exists; run the launcher reset before sparse reconstruction")
    if require_database_coverage:
        if not database["exists"]:
            failures.append("database.db is missing; run sparse reconstruction first")
        elif database["error"]:
            failures.append(f"database.db could not be read: {database['error']}")
        elif database["missing_names"] or database["extra_names"]:
            failures.append("database image names do not match the current image_list.txt")
    if require_sparse_model:
        if not sparse["exists"]:
            failures.append("the dense input sparse model is missing")
        elif sparse.get("error"):
            failures.append(f"the dense input sparse model is invalid: {sparse['error']}")
        elif sparse.get("registered_image_count", 0) < 2:
            failures.append("the dense input sparse model has fewer than two registered images")
        elif sparse.get("extra_names"):
            failures.append("the dense input sparse model contains images outside the current image list")
        if require_provenance and report.get("provenance", {}).get("status") != "verified":
            failures.append("sparse input provenance is not verified: " + report.get("provenance", {}).get("error", "missing run record"))
    return failures


def print_summary(report: Dict[str, Any]) -> None:
    input_info = report["input"]
    database = report["database"]
    sparse = report["dense_input_sparse"]
    print(f"[provenance] {report.get('provenance', {}).get('status', 'unverified')}", flush=True)
    print(
        "[coverage] input: "
        f"listed={input_info['listed_count']}, files={input_info['directory_file_count']}, "
        f"missing={len(input_info['missing_files'])}, extra={len(input_info['extra_files'])}",
        flush=True,
    )
    database_state = "missing" if not database["exists"] else str(database["image_count"])
    print(
        "[coverage] database: "
        f"images={database_state}, missing={len(database['missing_names'])}, extra={len(database['extra_names'])}",
        flush=True,
    )
    if sparse["exists"]:
        print(
            "[coverage] dense sparse input: "
            f"registered={sparse['registered_image_count']}, path={sparse['path']}",
            flush=True,
        )
    for source, values in database["source_coverage"].items():
        print(
            f"[coverage] database {source}: {values['present_count']}/{values['expected_count']}",
            flush=True,
        )
    for source, values in sparse["source_coverage"].items():
        if values["expected_count"] and values["present_count"] < values["expected_count"]:
            print(
                f"[warn] sparse registered {source}: {values['present_count']}/{values['expected_count']}",
                flush=True,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--require-clean-database", action="store_true")
    parser.add_argument("--require-database-coverage", action="store_true")
    parser.add_argument("--require-sparse-model", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print the complete inspection report as JSON.")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--record-sparse-start", action="store_true", help="Snapshot inputs before feature extraction, requiring a clean workspace.")
    action.add_argument("--record-sparse-complete", action="store_true", help="Record completed output provenance only if the pre-run inputs are unchanged.")
    parser.add_argument("--runner", choices=sorted(SPARSE_RUNNERS), default="run_colmap_sparse.bat")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.record_sparse_start or args.record_sparse_complete:
        try:
            if args.record_sparse_start:
                begin_sparse_run(args.workspace, args.runner)
            else:
                complete_sparse_run(args.workspace)
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
            print(f"[error] Sparse provenance: {exc}", flush=True)
            return 2
        print("[provenance] " + ("inputs recorded" if args.record_sparse_start else "completed run recorded"), flush=True)
        return 0
    report = inspect_workspace(args.workspace)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_summary(report)
    failures = validation_failures(
        report,
        require_clean_database=args.require_clean_database,
        require_database_coverage=args.require_database_coverage,
        require_sparse_model=args.require_sparse_model,
    )
    if failures:
        for failure in failures:
            print(f"[error] Workspace preflight: {failure}", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
