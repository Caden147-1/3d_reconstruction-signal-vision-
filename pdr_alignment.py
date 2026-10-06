"""Record visual model selection without inferring a metric pose from sensor timing.

The filename remains callable by existing generated sparse runners.
"""
import argparse
import json
from pathlib import Path

from colmap_workspace_check import (
    ALIGNMENT_RECORD, PENDING_NAME, PROVENANCE_NAME, file_identity,
    primary_sparse_model, sparse_input_identity, sparse_model_identity, write_provenance,
)
from workspace_lock import WorkspaceBusyError, locked_workspace


@locked_workspace("workspace")
def select_visual_model(workspace):
    workspace = Path(workspace).resolve()
    if (workspace / PROVENANCE_NAME).exists():
        raise ValueError("Existing certified geometry must not be changed; resume its dense stage")
    pending = json.loads((workspace / PENDING_NAME).read_text(encoding="utf-8"))
    if pending.get("schema_version") != 1 or sparse_input_identity(workspace, pending["inputs"]["runner"]) != pending["inputs"]:
        raise ValueError("Sparse inputs changed before model selection")
    if any((workspace / name).exists() for name in (
            "sparse_pdr_aligned", "sparse_pdr_aligned.pending", ALIGNMENT_RECORD,
            "pdr_alignment_reviewed_refs.txt")):
        raise ValueError("Previous model-selection artifacts exist; reset before a new sparse run")
    model = primary_sparse_model(workspace)
    if model is None:
        raise ValueError("No valid visual sparse component is available")
    name = model.relative_to(workspace).as_posix()
    decision = {"schema_version": 1, "status": "withheld",
                "reason": "video_sensor_assistance_is_not_a_metric_camera_transform",
                "selected_model": name, "source_model": name,
                "source_model_identity": sparse_model_identity(model),
                "manifest_identity": file_identity(workspace / "metadata/images_manifest.csv"),
                "policy": {"capture_mode": "off", "method": "visual_geometry_with_soft_sensor_matching"}}
    write_provenance(workspace / ALIGNMENT_RECORD, decision)
    print(f"[model] Selected {name}; WiFi/IMU assist matching without moving camera poses.", flush=True)
    return decision


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    # Existing video runners supply these flags; there is no alignment branch.
    parser.add_argument("--colmap-exe", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--disabled", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--capture-mode", choices=("off",), default="off", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        select_visual_model(args.workspace)
        return 0
    except (OSError, ValueError, KeyError, TypeError, WorkspaceBusyError) as exc:
        print(f"[model error] {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
