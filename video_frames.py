"""Extract route video frames and bind them as weak visual base images."""

from __future__ import annotations

import csv
import json
import math
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


VIDEO_SUFFIXES = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}


def write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def read_csv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    if not path.exists():
        return [], []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return reader.fieldnames or [], list(reader)


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


def natural_sort_key(path: Path) -> List[Tuple[int, Any]]:
    parts = re.split(r"(\d+)", path.name.lower())
    key: List[Tuple[int, Any]] = []
    for part in parts:
        key.append((0, int(part)) if part.isdigit() else (1, part))
    return key


def safe_stem(name: str) -> str:
    stem = Path(name).stem
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return cleaned or "video"


def relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def find_video_files(video_dir: Path) -> List[Path]:
    if not video_dir.exists() or not video_dir.is_dir():
        return []
    return sorted(
        (path for path in video_dir.iterdir() if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES),
        key=natural_sort_key,
    )


def resolve_ffmpeg(root: Path, ffmpeg_exe: Optional[Path]) -> Optional[Path]:
    candidates: List[Path] = []
    if ffmpeg_exe is not None:
        candidates.append(ffmpeg_exe if ffmpeg_exe.is_absolute() else root / ffmpeg_exe)

    found = shutil.which("ffmpeg")
    if found:
        candidates.append(Path(found))

    candidates.extend(
        [
            root / "ffmpeg.exe",
            root / "ffmpeg" / "bin" / "ffmpeg.exe",
            root / "colmap-x64-windows-cuda" / "bin" / "ffmpeg.exe",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def extract_with_ffmpeg(
    ffmpeg_exe: Path,
    video_path: Path,
    output_dir: Path,
    frame_interval_s: float,
    max_frames: int,
) -> Tuple[str, str]:
    fps = 1.0 / frame_interval_s
    command = [
        str(ffmpeg_exe),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps:.6f}",
        "-q:v",
        "2",
    ]
    if max_frames > 0:
        command.extend(["-frames:v", str(max_frames)])
    command.append(str(output_dir / "frame_%06d.jpg"))

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip()
        return "failed", message[:1000]
    return "ok", ""


def extract_with_opencv(
    video_path: Path,
    output_dir: Path,
    frame_interval_s: float,
    max_frames: int,
) -> Tuple[str, str]:
    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return "missing_opencv", "OpenCV is not installed."

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return "failed", "OpenCV could not open the video."

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(round(fps * frame_interval_s)))
    written = 0
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_index % frame_step == 0:
            written += 1
            output_path = output_dir / f"frame_{written:06d}.jpg"
            cv2.imwrite(str(output_path), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if max_frames > 0 and written >= max_frames:
                break
        frame_index += 1
    capture.release()
    return ("ok", "") if written else ("failed", "No frames were written.")


def frame_original_index(frame_path: Path, fallback: int) -> int:
    matches = re.findall(r"\d+", frame_path.stem)
    if not matches:
        return fallback
    return safe_int(matches[-1], fallback)


def filter_keyframes(
    frames: Sequence[Path],
    blur_threshold: float,
    min_visual_change: float,
    min_gap: int,
) -> Tuple[List[Path], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    if not frames:
        return [], {}, {
            "status": "no_frames",
            "input_frame_count": 0,
            "kept_frame_count": 0,
            "rejected_blurry_count": 0,
            "rejected_low_change_count": 0,
            "rejected_too_close_count": 0,
            "rejected_unreadable_count": 0,
        }

    needs_cv = blur_threshold > 0 or min_visual_change > 0
    if not needs_cv and min_gap <= 1:
        return list(frames), {}, {
            "status": "disabled",
            "input_frame_count": len(frames),
            "kept_frame_count": len(frames),
            "rejected_blurry_count": 0,
            "rejected_low_change_count": 0,
            "rejected_too_close_count": 0,
            "rejected_unreadable_count": 0,
        }

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError:
        return list(frames), {}, {
            "status": "missing_opencv",
            "input_frame_count": len(frames),
            "kept_frame_count": len(frames),
            "rejected_blurry_count": 0,
            "rejected_low_change_count": 0,
            "rejected_too_close_count": 0,
            "rejected_unreadable_count": 0,
        }

    kept: List[Path] = []
    metrics: Dict[str, Dict[str, Any]] = {}
    best_frame: Optional[Path] = None
    best_blur = -1.0
    last_kept_gray = None
    last_kept_frame_pos = -10**9
    rejected_blurry = 0
    rejected_low_change = 0
    rejected_too_close = 0
    rejected_unreadable = 0
    min_gap = max(1, min_gap)

    for frame_pos, frame_path in enumerate(frames):
        gray = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            rejected_unreadable += 1
            continue

        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var()) if blur_threshold > 0 else 0.0
        if blur_score > best_blur:
            best_blur = blur_score
            best_frame = frame_path

        if blur_threshold > 0 and blur_score < blur_threshold:
            rejected_blurry += 1
            continue

        if kept and frame_pos - last_kept_frame_pos < min_gap:
            rejected_too_close += 1
            continue

        visual_change = ""
        if last_kept_gray is not None and min_visual_change > 0:
            compare_gray = gray
            if gray.shape != last_kept_gray.shape:
                compare_gray = cv2.resize(gray, (last_kept_gray.shape[1], last_kept_gray.shape[0]))
            visual_change_value = float(cv2.mean(cv2.absdiff(compare_gray, last_kept_gray))[0])
            visual_change = round(visual_change_value, 6)
            if visual_change_value < min_visual_change:
                rejected_low_change += 1
                continue

        kept.append(frame_path)
        metrics[frame_path.name] = {
            "blur_score": round(blur_score, 6) if blur_threshold > 0 else "",
            "visual_change": visual_change,
            "keyframe_filter_status": "kept",
        }
        last_kept_gray = gray
        last_kept_frame_pos = frame_pos

    fallback_used = False
    if not kept and best_frame is not None:
        kept.append(best_frame)
        metrics[best_frame.name] = {
            "blur_score": round(best_blur, 6) if blur_threshold > 0 else "",
            "visual_change": "",
            "keyframe_filter_status": "fallback_best_blur",
        }
        fallback_used = True

    return kept, metrics, {
        "status": "ok",
        "input_frame_count": len(frames),
        "kept_frame_count": len(kept),
        "rejected_blurry_count": rejected_blurry,
        "rejected_low_change_count": rejected_low_change,
        "rejected_too_close_count": rejected_too_close,
        "rejected_unreadable_count": rejected_unreadable,
        "fallback_used": fallback_used,
        "blur_threshold": blur_threshold,
        "min_visual_change": min_visual_change,
        "min_gap": min_gap,
    }


def clean_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def point_sort_key(row: Dict[str, str]) -> Tuple[int, int]:
    return safe_int(row.get("global_point_index")), safe_int(row.get("point_index"))


def load_anchor_points(processed_dir: Path) -> List[Dict[str, str]]:
    _fields, rows = read_csv(processed_dir / "image_wifi_points.csv")
    points = [row for row in rows if row.get("global_point_index")]
    points.sort(key=point_sort_key)
    return points


def interpolate_value(left: Dict[str, str], right: Dict[str, str], field: str, ratio: float) -> str:
    left_value = safe_float(left.get(field))
    right_value = safe_float(right.get(field))
    if math.isnan(left_value) or math.isnan(right_value):
        return ""
    return f"{left_value + ratio * (right_value - left_value):.6f}"


def bind_frame_to_route(
    points: Sequence[Dict[str, str]],
    frame_index_zero_based: int,
    frame_count: int,
) -> Dict[str, Any]:
    if not points:
        return {
            "bind_type": "visual_only",
            "wifi_label_weight": 0.0,
            "nearest_global_point_index": "",
            "left_global_point_index": "",
            "right_global_point_index": "",
            "route_ratio": "",
            "pdr_x": "",
            "pdr_y": "",
            "pdr_bind_source": "no_anchor_points",
        }

    if len(points) == 1 or frame_count <= 1:
        point = points[0]
        return {
            "bind_type": "single_anchor_interpolated",
            "wifi_label_weight": 0.15,
            "nearest_global_point_index": point.get("global_point_index", ""),
            "left_global_point_index": point.get("global_point_index", ""),
            "right_global_point_index": point.get("global_point_index", ""),
            "route_ratio": "0.000000",
            "pdr_x": point.get("pdr_x", ""),
            "pdr_y": point.get("pdr_y", ""),
            "pdr_bind_source": "single_anchor_point",
        }

    max_frame_index = max(0, frame_count - 1)
    bounded_frame_index = min(max(0, frame_index_zero_based), max_frame_index)
    progress = bounded_frame_index / max(1, max_frame_index)
    route_position = progress * (len(points) - 1)
    left_index = min(len(points) - 1, max(0, int(math.floor(route_position))))
    right_index = min(len(points) - 1, int(math.ceil(route_position)))
    ratio = route_position - left_index
    left = points[left_index]
    right = points[right_index]
    nearest = left if ratio <= 0.5 else right

    return {
        "bind_type": "order_interpolated_video_base",
        "wifi_label_weight": 0.15,
        "nearest_global_point_index": nearest.get("global_point_index", ""),
        "left_global_point_index": left.get("global_point_index", ""),
        "right_global_point_index": right.get("global_point_index", ""),
        "route_ratio": f"{progress:.6f}",
        "pdr_x": interpolate_value(left, right, "pdr_x", ratio),
        "pdr_y": interpolate_value(left, right, "pdr_y", ratio),
        "pdr_bind_source": "route_order_interpolation",
    }


def build_frame_rows(
    root: Path,
    video_path: Path,
    frames: Sequence[Path],
    points: Sequence[Dict[str, str]],
    frame_interval_s: float,
    start_sequence_index: int,
    source_frame_count: Optional[int] = None,
    frame_metrics: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    frame_count = source_frame_count or len(frames)
    frame_metrics = frame_metrics or {}
    for idx, frame_path in enumerate(frames):
        original_index = frame_original_index(frame_path, idx + 1)
        binding = bind_frame_to_route(points, max(0, original_index - 1), frame_count)
        metrics = frame_metrics.get(frame_path.name, {})
        rows.append(
            {
                "source_type": "video_base_frame",
                "video_name": video_path.name,
                "video_path": relative_path(video_path, root),
                "frame_name": frame_path.name,
                "frame_path": relative_path(frame_path, root),
                "video_frame_index": original_index,
                "video_sequence_index": start_sequence_index + idx,
                "video_time_s": f"{(original_index - 1) * frame_interval_s:.3f}",
                "blur_score": metrics.get("blur_score", ""),
                "visual_change": metrics.get("visual_change", ""),
                "keyframe_filter_status": metrics.get("keyframe_filter_status", "kept"),
                **binding,
            }
        )
    return rows


def run_video_frame_pipeline(
    root: Path,
    processed_dir: Path,
    video_dir: Path,
    frame_interval_s: float = 1.5,
    max_frames: int = 120,
    ffmpeg_exe: Optional[Path] = None,
    blur_threshold: float = 80.0,
    keyframe_min_change: float = 6.0,
    keyframe_min_gap: int = 2,
) -> Dict[str, Any]:
    if frame_interval_s <= 0:
        raise ValueError("--video-frame-interval-s must be greater than zero")

    resolved_video_dir = video_dir if video_dir.is_absolute() else root / video_dir
    frames_root = processed_dir / "video_frames"
    manifest_path = processed_dir / "video_frames_manifest.csv"
    summary_path = processed_dir / "video_frames_summary.json"

    if not resolved_video_dir.exists():
        summary = {
            "status": "skipped",
            "reason": "missing_video_dir",
            "video_dir": str(resolved_video_dir),
            "frame_manifest": str(manifest_path),
        }
        write_csv(manifest_path, VIDEO_FRAME_FIELDS, [])
        write_json(summary_path, summary)
        return summary

    videos = find_video_files(resolved_video_dir)
    if not videos:
        summary = {
            "status": "skipped",
            "reason": "no_video_files",
            "video_dir": str(resolved_video_dir),
            "frame_manifest": str(manifest_path),
        }
        write_csv(manifest_path, VIDEO_FRAME_FIELDS, [])
        write_json(summary_path, summary)
        return summary

    ffmpeg = resolve_ffmpeg(root, ffmpeg_exe)
    points = load_anchor_points(processed_dir)
    rows: List[Dict[str, Any]] = []
    video_summaries: List[Dict[str, Any]] = []

    for video_path in videos:
        output_dir = frames_root / safe_stem(video_path.name)
        clean_output_dir(output_dir)
        if ffmpeg is not None:
            extract_status, message = extract_with_ffmpeg(
                ffmpeg_exe=ffmpeg,
                video_path=video_path,
                output_dir=output_dir,
                frame_interval_s=frame_interval_s,
                max_frames=max_frames,
            )
            extractor = "ffmpeg"
        else:
            extract_status, message = extract_with_opencv(
                video_path=video_path,
                output_dir=output_dir,
                frame_interval_s=frame_interval_s,
                max_frames=max_frames,
            )
            extractor = "opencv"

        frames = sorted(output_dir.glob("*.jpg"), key=natural_sort_key)
        selected_frames, frame_metrics, keyframe_summary = filter_keyframes(
            frames=frames,
            blur_threshold=blur_threshold,
            min_visual_change=keyframe_min_change,
            min_gap=keyframe_min_gap,
        )
        video_rows = build_frame_rows(
            root=root,
            video_path=video_path,
            frames=selected_frames,
            points=points,
            frame_interval_s=frame_interval_s,
            start_sequence_index=len(rows) + 1,
            source_frame_count=len(frames),
            frame_metrics=frame_metrics,
        )
        rows.extend(video_rows)
        video_summaries.append(
            {
                "video": relative_path(video_path, root),
                "extractor": extractor,
                "status": extract_status,
                "message": message,
                "extracted_frame_count": len(frames),
                "frame_count": len(selected_frames),
                "keyframe_filter": keyframe_summary,
                "output_dir": str(output_dir),
            }
        )

    write_csv(manifest_path, VIDEO_FRAME_FIELDS, rows)
    status = "ok" if rows else "no_frames"
    if all(item["status"] != "ok" for item in video_summaries):
        status = "extract_failed"
    summary = {
        "status": status,
        "video_dir": str(resolved_video_dir),
        "video_count": len(videos),
        "frame_count": len(rows),
        "anchor_point_count": len(points),
        "frame_interval_s": frame_interval_s,
        "max_frames_per_video": max_frames,
        "blur_threshold": blur_threshold,
        "keyframe_min_change": keyframe_min_change,
        "keyframe_min_gap": keyframe_min_gap,
        "ffmpeg_exe": "" if ffmpeg is None else str(ffmpeg),
        "frame_manifest": str(manifest_path),
        "videos": video_summaries,
    }
    write_json(summary_path, summary)
    return summary


VIDEO_FRAME_FIELDS = [
    "source_type",
    "video_name",
    "video_path",
    "frame_name",
    "frame_path",
    "video_frame_index",
    "video_sequence_index",
    "video_time_s",
    "blur_score",
    "visual_change",
    "keyframe_filter_status",
    "bind_type",
    "wifi_label_weight",
    "nearest_global_point_index",
    "left_global_point_index",
    "right_global_point_index",
    "route_ratio",
    "pdr_x",
    "pdr_y",
    "pdr_bind_source",
]
