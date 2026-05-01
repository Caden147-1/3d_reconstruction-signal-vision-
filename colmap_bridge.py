"""Prepare a COLMAP workspace from the exported reconstruction dataset."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


METADATA_FILES = [
    "images_manifest.csv",
    "camera_pose_seed.csv",
    "wifi_fingerprint_library.csv",
    "wifi_boundary_seed.csv",
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
    content = batch_header(root, workspace_dir, colmap_exe) + f"""
if not exist "%WORKSPACE%\\sparse" mkdir "%WORKSPACE%\\sparse"

echo [1/3] Extracting features...
"%COLMAP_EXE%" feature_extractor ^
  --database_path "%DATABASE%" ^
  --image_path "%IMAGE_PATH%" ^
  --image_list_path "%IMAGE_LIST%" ^
  --ImageReader.camera_model {camera_model} ^
  --ImageReader.single_camera {single_camera_flag} ^
  --FeatureExtraction.use_gpu {gpu_flag}
if errorlevel 1 exit /b %errorlevel%

echo [2/3] Matching features with {matcher_command}...
"%COLMAP_EXE%" {matcher_command} ^
  --database_path "%DATABASE%" ^
  --FeatureMatching.use_gpu {gpu_flag}
if errorlevel 1 exit /b %errorlevel%

echo [3/3] Building sparse model...
"%COLMAP_EXE%" mapper ^
  --database_path "%DATABASE%" ^
  --image_path "%IMAGE_PATH%" ^
  --output_path "%WORKSPACE%\\sparse"
if errorlevel 1 exit /b %errorlevel%

echo [done] Sparse model written under %WORKSPACE%\\sparse
"""
    write_text(workspace_dir / "run_colmap_sparse.bat", content)


def write_dense_batch(root: Path, workspace_dir: Path, colmap_exe: Path, dense_max_image_size: int) -> None:
    content = batch_header(root, workspace_dir, colmap_exe) + f"""
set "SPARSE_MODEL=%WORKSPACE%\\sparse\\0"
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
echo This will delete database.db, sparse, dense, and automatic output folders under:
echo %WORKSPACE%
set /p CONFIRM=Type YES to reset COLMAP outputs:
if /I not "%CONFIRM%"=="YES" (
  echo cancelled
  exit /b 0
)
if exist "%WORKSPACE%database.db" del "%WORKSPACE%database.db"
if exist "%WORKSPACE%sparse" rmdir /s /q "%WORKSPACE%sparse"
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

COLMAP itself uses the images for reconstruction. WiFi/PDR metadata is stored
beside the reconstruction so you can align, constrain, or evaluate the generated
point cloud afterward.
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
    matcher: str = "exhaustive",
    quality: str = "high",
    dense_max_image_size: int = 2000,
) -> Dict[str, Any]:
    _fields, manifest_rows = read_csv(reconstruction_dir / "images_manifest.csv")
    workspace_dir.mkdir(parents=True, exist_ok=True)
    resolved_colmap_exe = find_colmap_exe(root, colmap_dir=colmap_dir, colmap_exe=colmap_exe)

    copied_images, image_names, missing_images = copy_images(reconstruction_dir, workspace_dir, manifest_rows)
    copied_metadata = copy_metadata(reconstruction_dir, workspace_dir)
    write_image_list(workspace_dir, image_names)

    matcher = matcher if matcher in {"exhaustive", "sequential"} else "exhaustive"
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
