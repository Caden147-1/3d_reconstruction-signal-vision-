"""Read feature coverage and numerical warnings without changing reconstruction.

A log is evidence for a model only when its prefix was recorded during that
sparse run and the run's output provenance still verifies. Never select a log
by modification time: a newer failed run may belong to different inputs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import statistics
import time


NATIVE_PREFIX = re.compile(r"^[IWEF]\d{8}\s+\S+\s+\d+\s+[^\]]+\]\s*")
RUN_ID = re.compile(r"^[A-Za-z0-9_-]+$")


def parse_sparse_log(text: str) -> dict:
    """Count original native messages, not the console's repeated-warning summary."""
    failures = positive = cholesky = 0
    zero_features = []
    examples = []
    image_name = None
    phase = "unknown"
    registered_frames = None
    contexts = {}
    for line_number, raw in enumerate(text.splitlines(), 1):
        line = NATIVE_PREFIX.sub("", raw.strip())
        if line.startswith("Registering initial image pair"):
            registered_frames = 2
            phase = "initialization"
        match = re.search(r"num_reg_(?:frames|images)=(\d+)", line)
        if match:
            registered_frames = int(match.group(1))
            phase = "registration"
        if "Global bundle adjustment" in line:
            phase = "global_bundle_adjustment"
        if "Linear solver failure" in line:
            failures += 1
            positive += "not positive definite" in line
            cholesky += "Cholesky factorization" in line
            context = (phase, registered_frames)
            contexts[context] = contexts.get(context, 0) + 1
            if len(examples) < 5:
                examples.append({"line": line_number, "message": line})
        match = re.match(r"Name:\s+(.+)$", line)
        if match:
            image_name = match.group(1).strip()
        match = re.match(r"Features:\s+(\d+)\s+\(SIFT\)", line)
        if match:
            if int(match.group(1)) == 0 and image_name:
                zero_features.append(image_name)
            image_name = None
    return {
        "linear_solver_failure_count": failures,
        "not_positive_definite_count": positive,
        "dense_cholesky_failure_count": cholesky,
        "failure_contexts": [
            {"phase": phase, "last_reported_registered_frames": frames, "count": count}
            for (phase, frames), count in contexts.items()
        ],
        "examples": examples,
        "zero_feature_images": sorted(set(zero_features)),
        "interpretation": (
            "Counts describe failed solver steps, not a proven failed final model or "
            "a diagnosed root cause. Registration context is log context, not a "
            "certified sparse component assignment. Exit code 0 does not certify "
            "numerical convergence or geometry accuracy."
        ),
    }


def _log_path(workspace: Path, record: dict) -> Path:
    run_id = record.get("run_id", "")
    if not isinstance(run_id, str) or not RUN_ID.fullmatch(run_id):
        raise ValueError("Invalid sparse log run ID")
    path = Path(record.get("path", ""))
    path = (path if path.is_absolute() else workspace / path).resolve()
    if path.parent != (workspace / "logs").resolve() or not path.name.endswith(f"_{run_id}.log"):
        raise ValueError("Sparse log is not the recorded run's workspace log")
    return path


def begin_run_log(workspace: Path) -> dict:
    """Capture the launcher's explicit identity; missing logging remains usable."""
    record = {"run_id": os.environ.get("FYP_COLMAP_RUN_ID", ""),
              "path": os.environ.get("FYP_COLMAP_LOG_PATH", "")}
    if not all(record.values()):
        return {"status": "unavailable", "reason": "launcher_log_identity_missing"}
    try:
        path = _log_path(Path(workspace), record)
        record["path"] = path.relative_to(Path(workspace).resolve()).as_posix()
    except (OSError, ValueError, TypeError) as exc:
        return {"status": "unavailable", "reason": str(exc)}
    return {"status": "pending", **record}


def complete_run_log(workspace: Path, pending: dict, *, wait_seconds: float = 2.0) -> dict:
    """Seal completed mapper output while later quality output may append safely."""
    if not isinstance(pending, dict):
        return {"status": "unavailable", "reason": "invalid_pending_log_record"}
    if pending.get("status") != "pending":
        return pending or {"status": "unavailable", "reason": "launcher_log_identity_missing"}
    try:
        path = _log_path(Path(workspace), pending)
        sentinel = f"[sparse-log] mapper complete run={pending['run_id']}".encode("ascii")
        deadline = time.monotonic() + wait_seconds
        while True:
            with path.open("rb") as handle:
                data = handle.read(path.stat().st_size)
            # The parent's logging thread may not have drained the child pipe
            # yet. This same-pipe marker certifies that all mapper lines arrived.
            if sentinel in data.splitlines():
                break
            if time.monotonic() >= deadline:
                raise ValueError("Completed mapper log marker was not flushed; numerical diagnostics unavailable")
            time.sleep(0.05)
        # Only complete lines are part of this immutable prefix.
        data = data[:data.rfind(b"\n") + 1]
        if not data:
            raise ValueError("Recorded sparse log has no completed lines")
        return {"status": "recorded", "run_id": pending["run_id"], "path": pending["path"],
                "byte_count": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    except (OSError, ValueError, TypeError) as exc:
        return {"status": "unavailable", "reason": str(exc)}


def recorded_log_diagnostics(workspace: Path) -> dict:
    """Verify only the recorded log prefix; caller also checks model provenance."""
    try:
        receipt = json.loads((Path(workspace) / "sparse_run_provenance.json").read_text(encoding="utf-8"))
        record = receipt.get("run_log", {})
        if record.get("status") != "recorded":
            return {"status": "unverified", "reason": "no_run_bound_log_receipt"}
        path = _log_path(Path(workspace), record)
        length = record.get("byte_count")
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise ValueError("Invalid sparse log prefix size")
        if length > path.stat().st_size:
            raise ValueError("Recorded sparse log prefix changed or is truncated")
        with path.open("rb") as handle:
            data = handle.read(length)
        if len(data) != length or hashlib.sha256(data).hexdigest() != record.get("sha256"):
            raise ValueError("Recorded sparse log prefix changed or is truncated")
        sentinel = f"[sparse-log] mapper complete run={record['run_id']}".encode("ascii")
        if sentinel not in data.splitlines():
            raise ValueError("Recorded log does not contain its completed mapper marker")
        return {"status": "verified_prefix", "run_id": record["run_id"],
                "path": str(path), "byte_count": length, "sha256": record["sha256"],
                **parse_sparse_log(data.decode("utf-8", errors="replace"))}
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        return {"status": "unverified", "reason": str(exc)}


def sparse_run_diagnostics(workspace: Path, provenance_status: str) -> dict:
    if provenance_status != "verified":
        return {"status": "unverified", "reason": "sparse_outputs_not_verified",
                "provenance_status": provenance_status}
    result = recorded_log_diagnostics(workspace)
    if result["status"] == "verified_prefix":
        result["status"] = "verified"
    return result


def database_feature_diagnostics(workspace: Path) -> dict:
    """Query the current database read-only; distinguish zero from absent rows."""
    from colmap_workspace_check import open_readonly_database
    path = Path(workspace) / "database.db"
    if not path.is_file():
        return {"status": "unavailable", "reason": "database_missing"}
    try:
        connection = open_readonly_database(path)
        try:
            rows = connection.execute(
                'SELECT i.name, k.rows FROM images i LEFT JOIN keypoints k '
                'ON i.image_id = k.image_id ORDER BY i.name').fetchall()
        finally:
            connection.close()
        if any(count is not None and (not isinstance(count, int) or count < 0) for _, count in rows):
            raise ValueError("Invalid keypoint row count")
        counts = [count for _, count in rows if count is not None]
        zero = [name for name, count in rows if count == 0]
        missing = [name for name, count in rows if count is None]
        return {"status": "available", "image_count": len(rows),
                "zero_feature_count": len(zero), "zero_feature_images": zero,
                "missing_feature_count": len(missing), "missing_feature_images": missing,
                "keypoint_count": {"min": min(counts) if counts else None,
                                   "median": statistics.median(counts) if counts else None,
                                   "max": max(counts) if counts else None},
                "interpretation": "Zero keypoints prevent feature matching for that image. "
                                  "This does not identify why extraction returned zero; "
                                  "inspect the original image and masks before changing inputs."}
    except (OSError, ValueError, sqlite3.Error) as exc:
        return {"status": "unavailable", "reason": str(exc)}
