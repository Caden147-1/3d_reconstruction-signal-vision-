"""Check native COLMAP build/CLI compatibility before resetting run outputs.

These read-only help probes verify the commands used by the generated pipeline.
CUDA build support is separate from device/driver health and from GPU BA.
"""
from __future__ import annotations

from pathlib import Path
import re
import subprocess


FEATURE_OPTIONS = {
    "database_path", "image_path", "image_list_path",
    "ImageReader.camera_model", "ImageReader.single_camera",
    "ImageReader.single_camera_per_folder", "ImageReader.mask_path",
    "FeatureExtraction.use_gpu", "FeatureExtraction.num_threads",
    "SiftExtraction.max_image_size", "SiftExtraction.max_num_features",
}
MATCHING_OPTIONS = {
    "database_path", "FeatureMatching.use_gpu", "FeatureMatching.num_threads",
    "FeatureMatching.guided_matching", "FeatureMatching.max_num_matches",
}
MAPPER_OPTIONS = {
    'database_path', 'image_path', 'input_path', 'output_path',
    'Mapper.image_list_path', 'Mapper.multiple_models', 'Mapper.num_threads',
    'Mapper.snapshot_path', 'Mapper.snapshot_frames_freq', 'Mapper.ba_use_gpu',
    'Mapper.ba_refine_focal_length', 'Mapper.ba_refine_extra_params',
    'Mapper.ba_refine_principal_point',
    'Mapper.ba_global_frames_ratio', 'Mapper.ba_global_points_ratio',
    'Mapper.max_runtime_seconds', 'Mapper.max_reg_trials',
    'Mapper.ba_global_max_num_iterations', 'Mapper.ba_global_max_refinements',
}
BUNDLE_OPTIONS = {
    'input_path', 'output_path', 'BundleAdjustment.max_num_iterations',
    'BundleAdjustment.refine_focal_length', 'BundleAdjustment.refine_extra_params',
    'BundleAdjustment.refine_principal_point', 'BundleAdjustment.use_gpu',
}
DENSE_OPTIONS = {
    "image_undistorter": {"image_path", "input_path", "output_path", "output_type", "max_image_size"},
    "patch_match_stereo": {"workspace_path", "workspace_format", "PatchMatchStereo.geom_consistency",
                           "PatchMatchStereo.max_image_size"},
    "stereo_fusion": {"workspace_path", "workspace_format", "input_type", "output_path",
                      "StereoFusion.max_image_size", "StereoFusion.num_threads", "StereoFusion.mask_path"},
}


def _read_help(executable: Path, command: str | None = None) -> str:
    from process_control import managed_process
    arguments = [str(executable), *([command] if command else []), "-h"]
    label = command or "version/build"
    try:
        with managed_process(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as process:
            output, _ = process.communicate(timeout=20)
            text = output.decode("utf-8", errors="replace")
            if process.returncode:
                raise ValueError(f"COLMAP {label} probe failed (exit {process.returncode}): {text[-800:].strip()}")
            return text
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ValueError(f"Cannot execute COLMAP {label} probe at {executable}: {exc}") from exc


def validate_colmap_capabilities(executable: Path, *, matcher: str = "constrained",
                                 require_dense: bool = True) -> dict:
    """Fail before preprocessing if the selected native executable cannot fit it."""
    executable = Path(executable).resolve()
    if not executable.is_file():
        raise ValueError(f"COLMAP executable is missing: {executable}")
    matching_commands = {
        "constrained": ("matches_importer",),
        "hybrid": ("matches_importer", "exhaustive_matcher"),
        "exhaustive": ("exhaustive_matcher",),
        "sequential": ("sequential_matcher",),
    }
    if matcher not in matching_commands:
        raise ValueError(f"Unknown COLMAP matcher: {matcher}")
    banner = _read_help(executable)
    version = re.search(r"\bCOLMAP\s+([0-9]+(?:\.[0-9]+)+[^\s]*)", banner)
    if not version:
        raise ValueError("The selected executable did not identify itself as COLMAP")
    cuda = bool(re.search(r"\bwith CUDA\b", banner, flags=re.IGNORECASE))
    if require_dense and not cuda:
        raise ValueError("COLMAP does not report a CUDA-enabled build. This full pipeline requires CUDA "
                         "PatchMatch stereo; select a compatible CUDA COLMAP executable with --colmap-exe. "
                         "A working nvidia-smi alone does not establish COLMAP CUDA support.")
    required = {"feature_extractor": FEATURE_OPTIONS}
    for command in matching_commands[matcher]:
        required[command] = MATCHING_OPTIONS | ({"match_list_path", "match_type"} if command == "matches_importer" else set())
    required['mapper'] = MAPPER_OPTIONS
    required['bundle_adjuster'] = BUNDLE_OPTIONS
    if require_dense:
        required.update(DENSE_OPTIONS)
    for command, options in required.items():
        available = set(re.findall(r"--([A-Za-z_][A-Za-z_0-9.]*)", _read_help(executable, command)))
        missing = sorted(options - available)
        if missing:
            raise ValueError(f"COLMAP {version[1]} has an incompatible {command} CLI; missing: "
                             + ", ".join("--" + option for option in missing)
                             + ". Select a build with the FeatureExtraction/FeatureMatching interface "
                               "used by this pipeline (validated against COLMAP 3.13).")
    return {"version": version[1], "cuda_build": cuda, "commands_checked": list(required),
            "executable": str(executable)}
