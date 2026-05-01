"""Prepare a COLMAP workspace from the exported reconstruction dataset."""

from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


METADATA_FILES = [
    "images_manifest.csv",
    "camera_pose_seed.csv",
    "wifi_fingerprint_library.csv",
    "wifi_boundary_seed.csv",
    "wifi_localization_model.json",
    "wifi_localization_eval.csv",
    "wifi_positioning_summary.json",
    "wifi_signal_boundary.csv",
    "wifi_signal_grid.csv",
    "export_summary.json",
]


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def windows_path(path: Path) -> str:
    return str(path.resolve())


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def find_colmap_exe(root: Path, colmap_dir: Optional[Path] = None, colmap_exe: Optional[Path] = None) -> Path:
    candidates: List[Path] = []
    if colmap_exe is not None:
        candidates.append(colmap_exe if colmap_exe.is_absolute() else root / colmap_exe)
    if colmap_dir is not None:
        resolved_dir = colmap_dir if colmap_dir.is_absolute() else root / colmap_dir
        candidates.extend([resolved_dir / "bin" / "colmap.exe", resolved_dir / "COLMAP.bat"])
    candidates.extend(
        [
            root / "colmap-x64-windows-cuda" / "bin" / "colmap.exe",
            root / "colmap-x64-windows-cuda" / "COLMAP.bat",
            root / "colmap" / "bin" / "colmap.exe",
            root / "colmap.exe",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve() if candidates else (root / "colmap.exe").resolve()


def copy_images(
    reconstruction_dir: Path,
    workspace_dir: Path,
    manifest_rows: Sequence[Dict[str, str]],
) -> Tuple[int, List[str], List[str]]:
    source_images_dir = reconstruction_dir / "images"
    target_images_dir = workspace_dir / "images"
    target_images_dir.mkdir(parents=True, exist_ok=True)

    image_names: List[str] = []
    missing: List[str] = []
    copied = 0
    for row in manifest_rows:
        image_file = row.get("image_file", "")
        if not image_file:
            continue
        source = source_images_dir / image_file
        target = target_images_dir / image_file
        image_names.append(image_file)
        if not source.exists():
            missing.append(str(source))
            continue
        shutil.copy2(source, target)
        copied += 1
    return copied, image_names, missing


def copy_metadata(reconstruction_dir: Path, workspace_dir: Path) -> List[str]:
    metadata_dir = workspace_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for name in METADATA_FILES:
        source = reconstruction_dir / name
        if not source.exists():
            continue
        shutil.copy2(source, metadata_dir / name)
        copied.append(name)
    return copied


def write_image_list(workspace_dir: Path, image_names: Sequence[str]) -> None:
    with (workspace_dir / "image_list.txt").open("w", encoding="utf-8") as handle:
        for image_name in image_names:
            handle.write(f"{image_name}\n")


def circular_slot_distance(slot_a: int, slot_b: int, slot_count: int) -> int:
    diff = abs(slot_a - slot_b)
    return min(diff, max(0, slot_count - diff))


def point_distance(row_a: Dict[str, str], row_b: Dict[str, str]) -> float:
    ax = safe_float(row_a.get("pdr_x"))
    ay = safe_float(row_a.get("pdr_y"))
    bx = safe_float(row_b.get("pdr_x"))
    by = safe_float(row_b.get("pdr_y"))
    if any(math.isnan(value) for value in (ax, ay, bx, by)):
        return math.inf
    return math.hypot(ax - bx, ay - by)


def build_constrained_match_pairs(
    manifest_rows: Sequence[Dict[str, str]],
    point_window: int,
    slot_window: int,
    max_pdr_distance_m: float,
) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    rows = [
        row
        for row in manifest_rows
        if row.get("image_file") and row.get("global_point_index") and row.get("image_slot")
    ]
    rows.sort(key=lambda row: (safe_int(row.get("global_point_index")), safe_int(row.get("image_slot"))))

    by_point: Dict[int, List[Dict[str, str]]] = {}
    for row in rows:
        by_point.setdefault(safe_int(row.get("global_point_index")), []).append(row)
    for point_rows in by_point.values():
        point_rows.sort(key=lambda row: safe_int(row.get("image_slot")))

    pairs: Dict[Tuple[str, str], str] = {}

    def add_pair(row_a: Dict[str, str], row_b: Dict[str, str], reason: str) -> None:
        image_a = row_a.get("image_file", "")
        image_b = row_b.get("image_file", "")
        if not image_a or not image_b or image_a == image_b:
            return
        key = tuple(sorted((image_a, image_b)))
        pairs.setdefault(key, reason)

    for point, point_rows in by_point.items():
        slot_count = max((safe_int(row.get("image_slot")) for row in point_rows), default=len(point_rows))
        for idx, row_a in enumerate(point_rows):
            for row_b in point_rows[idx + 1 :]:
                distance = circular_slot_distance(
                    safe_int(row_a.get("image_slot")),
                    safe_int(row_b.get("image_slot")),
                    slot_count,
                )
                if distance <= slot_window:
                    add_pair(row_a, row_b, "same_wifi_point_nearby_slots")

    points = sorted(by_point)
    point_row_by_id = {point: by_point[point][0] for point in points if by_point[point]}
    for left_index, point_a in enumerate(points):
        for point_b in points[left_index + 1 :]:
            point_gap = point_b - point_a
            pdr_gap = point_distance(point_row_by_id[point_a], point_row_by_id[point_b])
            if point_gap > point_window and pdr_gap > max_pdr_distance_m:
                continue
            rows_a = by_point[point_a]
            rows_b = by_point[point_b]
            slot_count = max(
                [safe_int(row.get("image_slot")) for row in rows_a + rows_b] or [len(rows_a), len(rows_b)]
            )
            for row_a in rows_a:
                for row_b in rows_b:
                    slot_gap = circular_slot_distance(
                        safe_int(row_a.get("image_slot")),
                        safe_int(row_b.get("image_slot")),
                        slot_count,
                    )
                    if slot_gap <= slot_window:
                        reason = "nearby_wifi_points" if point_gap <= point_window else "nearby_pdr_points"
                        add_pair(row_a, row_b, reason)

    sorted_pairs = sorted(pairs)
    summary = {
        "point_count": len(by_point),
        "image_count": len(rows),
        "pair_count": len(sorted_pairs),
        "point_window": point_window,
        "slot_window": slot_window,
        "max_pdr_distance_m": max_pdr_distance_m,
        "reason_counts": {},
    }
    for reason in pairs.values():
        summary["reason_counts"][reason] = summary["reason_counts"].get(reason, 0) + 1
    return sorted_pairs, summary


def write_match_pairs(workspace_dir: Path, pairs: Sequence[Tuple[str, str]]) -> None:
    with (workspace_dir / "match_pairs.txt").open("w", encoding="utf-8") as handle:
        for image_a, image_b in pairs:
            handle.write(f"{image_a} {image_b}\n")


def write_pdr_alignment_refs(workspace_dir: Path, manifest_rows: Sequence[Dict[str, str]]) -> int:
    written = 0
    seen: set[str] = set()
    with (workspace_dir / "pdr_alignment_refs.txt").open("w", encoding="utf-8") as handle:
        for row in sorted(
            manifest_rows,
            key=lambda item: (safe_int(item.get("global_point_index")), safe_int(item.get("image_slot"))),
        ):
            image_file = row.get("image_file", "")
            if not image_file or image_file in seen:
                continue
            x = safe_float(row.get("pdr_x"))
            y = safe_float(row.get("pdr_y"))
            z = safe_float(row.get("pdr_z"), 1.5)
            if any(math.isnan(value) for value in (x, y, z)):
                continue
            handle.write(f"{image_file} {x:.6f} {y:.6f} {z:.6f}\n")
            seen.add(image_file)
            written += 1
    return written


def batch_header(root: Path, workspace_dir: Path, colmap_exe: Path) -> str:
    return f"""@echo off
setlocal
set "ROOT={windows_path(root)}"
set "WORKSPACE={windows_path(workspace_dir)}"
set "COLMAP_EXE={windows_path(colmap_exe)}"
set "IMAGE_PATH=%WORKSPACE%\\images"
set "IMAGE_LIST=%WORKSPACE%\\image_list.txt"
set "DATABASE=%WORKSPACE%\\database.db"

if not exist "%COLMAP_EXE%" (
  echo [error] COLMAP not found: %COLMAP_EXE%
  exit /b 1
)
"""


def write_sparse_batch(
    root: Path,
    workspace_dir: Path,
    colmap_exe: Path,
    camera_model: str,
    single_camera: bool,
    use_gpu: bool,
    matcher: str,
) -> None:
    matcher_command = "sequential_matcher" if matcher == "sequential" else "exhaustive_matcher"
    single_camera_flag = "1" if single_camera else "0"
    gpu_flag = "1" if use_gpu else "0"
    if matcher == "constrained":
        matching_block = f"""echo [2/4] Matching constrained image pairs from WiFi/PDR/order priors...
if not exist "%WORKSPACE%\\match_pairs.txt" (
  echo [error] Missing constrained pair list: %WORKSPACE%\\match_pairs.txt
  exit /b 1
)
"%COLMAP_EXE%" matches_importer ^
  --database_path "%DATABASE%" ^
  --match_list_path "%WORKSPACE%\\match_pairs.txt" ^
  --match_type pairs ^
  --FeatureMatching.use_gpu {gpu_flag} ^
  --FeatureMatching.guided_matching 1 ^
  --FeatureMatching.max_num_matches 65536
if errorlevel 1 exit /b %errorlevel%
"""
    else:
        matching_block = f"""echo [2/4] Matching features with {matcher_command}...
"%COLMAP_EXE%" {matcher_command} ^
  --database_path "%DATABASE%" ^
  --FeatureMatching.use_gpu {gpu_flag} ^
  --FeatureMatching.guided_matching 1 ^
  --FeatureMatching.max_num_matches 65536
if errorlevel 1 exit /b %errorlevel%
"""
    content = batch_header(root, workspace_dir, colmap_exe) + f"""
if exist "%DATABASE%" (
  echo [warn] Existing database found: %DATABASE%
  echo [warn] If you changed matching constraints, run reset_colmap_outputs.bat first.
)
if not exist "%WORKSPACE%\\sparse" mkdir "%WORKSPACE%\\sparse"

echo [1/4] Extracting features...
"%COLMAP_EXE%" feature_extractor ^
  --database_path "%DATABASE%" ^
  --image_path "%IMAGE_PATH%" ^
  --image_list_path "%IMAGE_LIST%" ^
  --ImageReader.camera_model {camera_model} ^
  --ImageReader.single_camera {single_camera_flag} ^
  --FeatureExtraction.use_gpu {gpu_flag}
if errorlevel 1 exit /b %errorlevel%

{matching_block}

echo [3/4] Building sparse model...
"%COLMAP_EXE%" mapper ^
  --database_path "%DATABASE%" ^
  --image_path "%IMAGE_PATH%" ^
  --output_path "%WORKSPACE%\\sparse" ^
  --Mapper.multiple_models 0
if errorlevel 1 exit /b %errorlevel%

echo [4/4] Aligning sparse model to PDR coordinate seeds...
if exist "%WORKSPACE%\\pdr_alignment_refs.txt" (
  if exist "%WORKSPACE%\\sparse\\0" (
    if not exist "%WORKSPACE%\\sparse_pdr_aligned" mkdir "%WORKSPACE%\\sparse_pdr_aligned"
    "%COLMAP_EXE%" model_aligner ^
      --input_path "%WORKSPACE%\\sparse\\0" ^
      --output_path "%WORKSPACE%\\sparse_pdr_aligned\\0" ^
      --ref_images_path "%WORKSPACE%\\pdr_alignment_refs.txt" ^
      --ref_is_gps 0 ^
      --alignment_type custom ^
      --min_common_images 3 ^
      --alignment_max_error 10
    if errorlevel 1 (
      echo [warn] PDR alignment failed; dense step will use raw sparse model.
    ) else (
      echo [done] PDR-aligned sparse model: %WORKSPACE%\\sparse_pdr_aligned\\0
    )
  )
)

echo [done] Sparse model written under %WORKSPACE%\\sparse
"""
    write_text(workspace_dir / "run_colmap_sparse.bat", content)


def write_dense_batch(root: Path, workspace_dir: Path, colmap_exe: Path, dense_max_image_size: int) -> None:
    content = batch_header(root, workspace_dir, colmap_exe) + f"""
set "SPARSE_MODEL=%WORKSPACE%\\sparse_pdr_aligned\\0"
if not exist "%SPARSE_MODEL%" set "SPARSE_MODEL=%WORKSPACE%\\sparse\\0"
if not exist "%SPARSE_MODEL%" (
  echo [error] Sparse model not found: %SPARSE_MODEL%
  echo Run run_colmap_sparse.bat first, or check which sparse model folder COLMAP created.
  exit /b 1
)
if not exist "%WORKSPACE%\\dense" mkdir "%WORKSPACE%\\dense"

echo [1/3] Undistorting images...
"%COLMAP_EXE%" image_undistorter ^
  --image_path "%IMAGE_PATH%" ^
  --input_path "%SPARSE_MODEL%" ^
  --output_path "%WORKSPACE%\\dense" ^
  --output_type COLMAP ^
  --max_image_size {dense_max_image_size}
if errorlevel 1 exit /b %errorlevel%

echo [2/3] Running CUDA PatchMatch stereo...
"%COLMAP_EXE%" patch_match_stereo ^
  --workspace_path "%WORKSPACE%\\dense" ^
  --workspace_format COLMAP ^
  --PatchMatchStereo.geom_consistency 1
if errorlevel 1 exit /b %errorlevel%

echo [3/3] Fusing dense point cloud...
"%COLMAP_EXE%" stereo_fusion ^
  --workspace_path "%WORKSPACE%\\dense" ^
  --workspace_format COLMAP ^
  --input_type geometric ^
  --output_path "%WORKSPACE%\\dense\\fused.ply"
if errorlevel 1 exit /b %errorlevel%

echo [done] Dense point cloud: %WORKSPACE%\\dense\\fused.ply
"""
    write_text(workspace_dir / "run_colmap_dense.bat", content)


def write_full_batch(workspace_dir: Path) -> None:
    content = """@echo off
setlocal
call "%~dp0run_colmap_sparse.bat"
if errorlevel 1 exit /b %errorlevel%
call "%~dp0run_colmap_dense.bat"
if errorlevel 1 exit /b %errorlevel%
"""
    write_text(workspace_dir / "run_colmap_full.bat", content)


def write_automatic_batch(
    root: Path,
    workspace_dir: Path,
    colmap_exe: Path,
    camera_model: str,
    single_camera: bool,
    use_gpu: bool,
    quality: str,
) -> None:
    single_camera_flag = "1" if single_camera else "0"
    gpu_flag = "1" if use_gpu else "0"
    content = batch_header(root, workspace_dir, colmap_exe) + f"""
echo [info] Running COLMAP automatic_reconstructor...
"%COLMAP_EXE%" automatic_reconstructor ^
  --workspace_path "%WORKSPACE%" ^
  --image_path "%IMAGE_PATH%" ^
  --image_list_path "%IMAGE_LIST%" ^
  --camera_model {camera_model} ^
  --single_camera {single_camera_flag} ^
  --use_gpu {gpu_flag} ^
  --quality {quality} ^
  --dense 1
if errorlevel 1 exit /b %errorlevel%

echo [done] Automatic reconstruction complete.
"""
    write_text(workspace_dir / "run_colmap_automatic.bat", content)


def write_gui_batch(root: Path, workspace_dir: Path, colmap_exe: Path) -> None:
    content = batch_header(root, workspace_dir, colmap_exe) + """
echo [info] Opening COLMAP GUI...
"%COLMAP_EXE%" gui
"""
    write_text(workspace_dir / "open_colmap_gui.bat", content)


def write_reset_batch(workspace_dir: Path) -> None:
    content = """@echo off
setlocal
set "WORKSPACE=%~dp0"
echo This will delete database.db, sparse, PDR-aligned sparse, dense, and automatic output folders under:
echo %WORKSPACE%
set /p CONFIRM=Type YES to reset COLMAP outputs:
if /I not "%CONFIRM%"=="YES" (
  echo cancelled
  exit /b 0
)
if exist "%WORKSPACE%database.db" del "%WORKSPACE%database.db"
if exist "%WORKSPACE%sparse" rmdir /s /q "%WORKSPACE%sparse"
if exist "%WORKSPACE%sparse_pdr_aligned" rmdir /s /q "%WORKSPACE%sparse_pdr_aligned"
if exist "%WORKSPACE%dense" rmdir /s /q "%WORKSPACE%dense"
if exist "%WORKSPACE%stereo" rmdir /s /q "%WORKSPACE%stereo"
if exist "%WORKSPACE%meshed-poisson.ply" del "%WORKSPACE%meshed-poisson.ply"
if exist "%WORKSPACE%meshed-delaunay.ply" del "%WORKSPACE%meshed-delaunay.ply"
echo [done] COLMAP outputs reset.
"""
    write_text(workspace_dir / "reset_colmap_outputs.bat", content)


def write_readme(workspace_dir: Path) -> None:
    content = """# COLMAP Workspace

This folder is prepared from the current WiFi/IMU/photo capture.

## Start Here

1. Run `run_colmap_sparse.bat` to build the sparse SfM model.
2. If `sparse/0` is created, run `run_colmap_dense.bat` to generate `dense/fused.ply`.
3. Or run `run_colmap_full.bat` to do both steps.

`run_colmap_automatic.bat` is also provided if you prefer COLMAP's automatic
pipeline. The manual sparse/dense scripts are easier to debug.

## Inputs

- `images/`: 112 renamed photos, grouped by WiFi point and slot.
- `image_list.txt`: exact image list passed to COLMAP.
- `metadata/camera_pose_seed.csv`: PDR XY plus 8-view yaw seed.
- `metadata/wifi_fingerprint_library.csv`: WiFi fingerprint table.
- `metadata/wifi_boundary_seed.csv`: rough boundary from WiFi/PDR point hull.
- `metadata/wifi_localization_model.json`: KNN WiFi fingerprint localizer.
- `metadata/wifi_signal_grid.csv`: interpolated RSSI field inside the sampled area.
- `metadata/wifi_signal_boundary.csv`: RSSI metrics on the WiFi/PDR boundary.
- `match_pairs.txt`: constrained COLMAP image pairs from WiFi point order,
  neighboring view slots, and PDR distance.
- `pdr_alignment_refs.txt`: image position seeds used to align sparse output.

The default sparse script now uses `matches_importer` with `match_pairs.txt` to
avoid matching unrelated indoor images. After sparse mapping, it tries to align
the model to PDR coordinates and dense reconstruction prefers that aligned model.
"""
    write_text(workspace_dir / "README.md", content)


def prepare_colmap_workspace(
    root: Path,
    reconstruction_dir: Path,
    workspace_dir: Path,
    colmap_dir: Optional[Path] = None,
    colmap_exe: Optional[Path] = None,
    camera_model: str = "SIMPLE_RADIAL",
    single_camera: bool = True,
    use_gpu: bool = True,
    matcher: str = "constrained",
    quality: str = "high",
    dense_max_image_size: int = 2000,
    constrained_point_window: int = 1,
    constrained_slot_window: int = 2,
    constrained_max_pdr_distance_m: float = 8.0,
) -> Dict[str, Any]:
    _fields, manifest_rows = read_csv(reconstruction_dir / "images_manifest.csv")
    workspace_dir.mkdir(parents=True, exist_ok=True)
    resolved_colmap_exe = find_colmap_exe(root, colmap_dir=colmap_dir, colmap_exe=colmap_exe)

    copied_images, image_names, missing_images = copy_images(reconstruction_dir, workspace_dir, manifest_rows)
    copied_metadata = copy_metadata(reconstruction_dir, workspace_dir)
    write_image_list(workspace_dir, image_names)
    match_pairs, match_pair_summary = build_constrained_match_pairs(
        manifest_rows=manifest_rows,
        point_window=constrained_point_window,
        slot_window=constrained_slot_window,
        max_pdr_distance_m=constrained_max_pdr_distance_m,
    )
    write_match_pairs(workspace_dir, match_pairs)
    pdr_alignment_ref_count = write_pdr_alignment_refs(workspace_dir, manifest_rows)

    matcher = matcher if matcher in {"constrained", "exhaustive", "sequential"} else "constrained"
    write_sparse_batch(
        root=root,
        workspace_dir=workspace_dir,
        colmap_exe=resolved_colmap_exe,
        camera_model=camera_model,
        single_camera=single_camera,
        use_gpu=use_gpu,
        matcher=matcher,
    )
    write_dense_batch(
        root=root,
        workspace_dir=workspace_dir,
        colmap_exe=resolved_colmap_exe,
        dense_max_image_size=dense_max_image_size,
    )
    write_full_batch(workspace_dir)
    write_automatic_batch(
        root=root,
        workspace_dir=workspace_dir,
        colmap_exe=resolved_colmap_exe,
        camera_model=camera_model,
        single_camera=single_camera,
        use_gpu=use_gpu,
        quality=quality,
    )
    write_gui_batch(root=root, workspace_dir=workspace_dir, colmap_exe=resolved_colmap_exe)
    write_reset_batch(workspace_dir)
    write_readme(workspace_dir)

    summary = {
        "status": "ok" if not missing_images and resolved_colmap_exe.exists() else "needs_attention",
        "workspace_dir": str(workspace_dir),
        "reconstruction_dir": str(reconstruction_dir),
        "colmap_exe": str(resolved_colmap_exe),
        "colmap_exists": resolved_colmap_exe.exists(),
        "image_count": len(image_names),
        "copied_image_count": copied_images,
        "missing_image_count": len(missing_images),
        "missing_images": missing_images[:20],
        "metadata_files": copied_metadata,
        "camera_model": camera_model,
        "single_camera": single_camera,
        "use_gpu": use_gpu,
        "matcher": matcher,
        "constrained_match_pairs": match_pair_summary,
        "pdr_alignment_ref_count": pdr_alignment_ref_count,
        "quality": quality,
        "dense_max_image_size": dense_max_image_size,
        "scripts": [
            "run_colmap_sparse.bat",
            "run_colmap_dense.bat",
            "run_colmap_full.bat",
            "run_colmap_automatic.bat",
            "open_colmap_gui.bat",
            "reset_colmap_outputs.bat",
        ],
    }
    write_json(workspace_dir / "colmap_workspace_summary.json", summary)
    return summary
