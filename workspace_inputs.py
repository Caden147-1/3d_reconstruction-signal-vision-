"""Protect the inputs of existing reconstructions from preprocessing writes."""

from pathlib import Path


RECONSTRUCTION_OUTPUTS = (
    "database.db", "database.db-wal", "database.db-shm", "database.db-journal",
    "sparse_run_inputs.json", "sparse_run_provenance.json",
    "pdr_alignment_decision.json", "pdr_alignment_reviewed_refs.txt",
    "sparse_pdr_aligned.pending",
    "sparse", "sparse_pdr_aligned", "sparse_pose_prior", "automatic",
    "sparse_backbone", "sparse_backbone_images.txt", "sparse_optimization.json",
    "sparse_recovery", "sparse_recovery_summary.json",
    "dense", "stereo", "final.ply", "final_output_summary.json",
    "meshed-poisson.ply", "meshed-delaunay.ply", "fused_filtered.ply",
    "point_filter_summary.json",
)


class WorkspaceInputsProtectedError(ValueError):
    """Existing reconstruction inputs can only change after an explicit reset."""


def existing_reconstruction_outputs(workspace: Path) -> list[str]:
    return [name for name in RECONSTRUCTION_OUTPUTS
            if (workspace / name).exists() or (workspace / name).is_symlink()]


def require_fresh_workspace(workspace: Path) -> None:
    existing = existing_reconstruction_outputs(workspace)
    if existing:
        raise WorkspaceInputsProtectedError(
            f"Existing reconstruction inputs are protected in {workspace}: "
            + ", ".join(existing)
            + ". Use run_colmap.py resume to continue, select a new --colmap-workspace, "
              "or explicitly reset/rebuild before preprocessing."
        )
