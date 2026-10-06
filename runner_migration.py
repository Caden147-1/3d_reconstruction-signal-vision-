"""Remove retired runner throttles without rewriting reconstruction history.

Only the exact, audited text transformation below is compatible with an old
runner identity. Every other script edit still invalidates provenance/cache.
Historical scripts are retained as non-executable text, never run again.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import tempfile

from workspace_lock import locked_workspace, WorkspaceBusyError


RECEIPT = "runner_migration.json"
TRANSFORMATION = "remove_resource_throttles_v1"
RUNNERS = ("run_colmap_sparse.bat", "run_colmap_dense.bat", "run_colmap_automatic.bat")
_ENV = re.compile(r'set "(?:OMP_NUM_THREADS|OPENBLAS_NUM_THREADS|MKL_NUM_THREADS|NUMEXPR_NUM_THREADS|OPENCV_FOR_THREADS_NUM)=[1-9][0-9]*"')
_CAP = re.compile(r"echo \[load\] Low-peak thread cap: [1-9][0-9]*")
_THREAD = re.compile(r"  --(?:FeatureExtraction|FeatureMatching|SiftExtraction|SiftMatching|Mapper|StereoFusion|PoissonMeshing)\.num_threads [1-9][0-9]*( \^)?")
_WAIT = re.compile(r"echo \[cooldown\] Waiting ([1-9][0-9]*)s after [A-Za-z -]+ to reduce sustained peak load\.\.\.")


def bytes_identity(data: bytes) -> dict:
    return {"size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def remove_runner_throttles(data: bytes) -> bytes:
    """Delete generated throttle lines and repair only their BAT continuation."""
    lines = data.decode("utf-8").splitlines(keepends=True)
    if not any(_CAP.fullmatch(line.rstrip("\r\n")) or _WAIT.fullmatch(line.rstrip("\r\n"))
               for line in lines):
        # Explicit thread settings in a newly generated runner remain valid.
        return data
    result = []
    index = 0
    while index < len(lines):
        line = lines[index]
        text = line.rstrip("\r\n")
        index += 1
        if _ENV.fullmatch(text) or _CAP.fullmatch(text):
            continue
        wait = _WAIT.fullmatch(text)
        if wait:
            expected = f"timeout /t {wait[1]} /nobreak >nul"
            if index >= len(lines) or lines[index].rstrip("\r\n") != expected:
                raise ValueError("Unrecognized stage wait; refusing partial runner migration")
            index += 1
            continue
        thread = _THREAD.fullmatch(text)
        if thread:
            if not result or not result[-1].rstrip("\r\n").endswith(" ^"):
                raise ValueError("Thread option is not in a generated continued command")
            if not thread[1]:
                previous = result[-1]
                body = previous.rstrip("\r\n")
                result[-1] = body[:-2] + previous[len(body):]
            continue
        result.append(line)
    return "".join(result).encode("utf-8")


def _read_receipt(workspace: Path, identify=None) -> dict:
    try:
        path = workspace / RECEIPT
        expected = identify(path) if identify is not None else None
        data = path.read_bytes()
        if identify is not None and bytes_identity(data) != expected:
            return {}
        receipt = json.loads(data)
        if receipt.get("schema_version") == 1 and receipt.get("transformation") == TRANSFORMATION:
            return receipt
    except (OSError, ValueError, AttributeError):
        pass
    return {}


def _verified_entry(workspace: Path, runner: str, identify) -> dict | None:
    """A receipt alone cannot authorize changes: rederive exact bytes each time."""
    from colmap_workspace_check import contained_path

    if runner not in RUNNERS:
        return None
    receipt = _read_receipt(workspace, identify)
    try:
        entry = receipt["runners"][runner]
        history = contained_path(workspace, entry["history"])
        if history.parent != (workspace / "runner_history").resolve() or history.suffix != ".txt":
            return None
        before = identify(history)
        current_path = workspace / runner
        after = identify(current_path)
        if before != entry["before"] or after != entry["after"] or before == after:
            return None
        original, current = history.read_bytes(), current_path.read_bytes()
        if bytes_identity(original) != before or bytes_identity(current) != after:
            return None
        if remove_runner_throttles(original) != current:
            return None
        return entry
    except (OSError, ValueError, KeyError, TypeError, UnicodeError):
        return None


def compatible_runner_identity(workspace: Path, runner: str, expected: dict | None, *, identify=None):
    """Use the old identity only when its exact performance-only edit is proved."""
    from colmap_workspace_check import file_identity

    identify = identify or file_identity
    actual = identify(workspace / runner)
    if actual == expected:
        return actual
    entry = _verified_entry(workspace, runner, identify)
    return expected if entry is not None and expected == entry["before"] else actual


def migration_dependency_paths(workspace: Path) -> list[Path]:
    """Include absent receipt too, so creation during evaluation is detected."""
    from colmap_workspace_check import contained_path

    paths = [workspace / RECEIPT]
    for name, entry in _read_receipt(workspace).get("runners", {}).items():
        try:
            path = contained_path(workspace, entry["history"])
            if name in RUNNERS and path.parent == (workspace / "runner_history").resolve() and path.suffix == ".txt":
                paths.extend((workspace / name, path))
        except (ValueError, KeyError, TypeError):
            continue
    return list(dict.fromkeys(paths))


def effective_execution_settings(workspace: Path, summary: dict) -> dict:
    """Keep historical metadata intact; migrated runners no longer inherit caps."""
    from colmap_workspace_check import file_identity

    result = dict(summary)
    receipt = _read_receipt(workspace)
    if (not receipt.get("runners") or
            not all(_verified_entry(workspace, name, file_identity) for name in receipt["runners"])):
        return result
    old = receipt.get("previous_execution_settings", {})
    for key in ("colmap_num_threads", "stage_cooldown_s"):
        if key in old and summary.get(key) == old[key]:
            result[key] = 0
    return result


def _write_bytes(path: Path, data: bytes) -> None:
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="runner-", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(data)
    try:
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


@locked_workspace("workspace")
def migrate_workspace_runners(workspace: Path, *, apply: bool = False) -> dict:
    """Migrate idle, verified workspaces; never rewrite model or result receipts."""
    from colmap_workspace_check import file_identity, inspect_provenance, write_provenance
    from dense_cache import CACHE_RECORD, cache_identity, read_record
    from reconstruction_stages import verified_context, verified_final

    workspace = workspace.resolve()
    existing = _read_receipt(workspace)
    if existing:
        if not existing.get("runners") or any(_verified_entry(workspace, name, file_identity) is None
                                              for name in existing["runners"]):
            raise ValueError("Existing runner migration cannot be verified")
        return {"status": "already_migrated", "runners": sorted(existing["runners"])}
    changes = {}
    for name in RUNNERS:
        path = workspace / name
        if path.is_file():
            before = path.read_bytes()
            after = remove_runner_throttles(before)
            if before != after:
                changes[name] = (before, after)
    if not changes:
        return {"status": "unchanged", "runners": []}
    proof = inspect_provenance(workspace)
    if proof.get("status") != "verified":
        raise ValueError("Runner migration requires verified sparse provenance: " + proof.get("status", "missing"))
    cache = read_record(workspace / "dense" / CACHE_RECORD)
    if cache and (cache.get("status") != "complete" or cache.get("identity") != cache_identity(workspace)):
        raise ValueError("Runner migration requires an idle, completed, matching dense cache")
    binding = verified_context(workspace) if cache else None
    final_receipt = verified_final(workspace, binding) if binding else None
    if not apply:
        return {"status": "ready", "runners": sorted(changes), "verified_final": bool(final_receipt)}
    summary = read_record(workspace / "colmap_workspace_summary.json")
    history = workspace / "runner_history"
    if history.resolve() != history:
        raise ValueError("Runner history must not redirect outside its workspace")
    history.mkdir(exist_ok=True)
    receipt = {"schema_version": 1, "transformation": TRANSFORMATION,
               "created_at": datetime.now(timezone.utc).isoformat(),
               "previous_execution_settings": {key: summary[key] for key in ("colmap_num_threads", "stage_cooldown_s") if key in summary},
               "runners": {},
               "note": "Model, cache, fusion and final receipts retain their original historical runner hashes. Only execution throttles were removed."}
    for name, (before, after) in changes.items():
        archive = history / (name.removesuffix(".bat") + "_" + bytes_identity(before)["sha256"] + ".txt")
        if archive.exists() and archive.read_bytes() != before:
            raise ValueError("Existing runner history does not match")
        _write_bytes(archive, before)
        receipt["runners"][name] = {"before": bytes_identity(before), "after": bytes_identity(after),
                                     "history": archive.relative_to(workspace).as_posix()}
    try:
        # Publish the proof first; it only takes effect once exact new bytes exist.
        write_provenance(workspace / RECEIPT, receipt)
        for name, (before, after) in changes.items():
            if (workspace / name).read_bytes() != before:
                raise ValueError("Runner changed while preparing migration")
            _write_bytes(workspace / name, after)
        if inspect_provenance(workspace).get("status") != "verified":
            raise ValueError("Sparse provenance changed during migration")
        if binding and verified_context(workspace) != binding:
            raise ValueError("Dense reconstruction binding changed during migration")
        if binding:
            verified_final(workspace, binding)
    except BaseException:
        for name, (before, after) in changes.items():
            if (workspace / name).read_bytes() == after:
                _write_bytes(workspace / name, before)
        (workspace / RECEIPT).unlink(missing_ok=True)
        raise
    return {"status": "migrated", "runners": sorted(changes), "verified_final": bool(final_receipt)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Apply the verified migration; default is a read-only check.")
    args = parser.parse_args()
    try:
        result = migrate_workspace_runners(args.workspace, apply=args.apply)
    except (OSError, ValueError, WorkspaceBusyError) as exc:
        print(f"[error] Runner migration: {exc}", flush=True)
        return 1
    print("[runner migration] " + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
