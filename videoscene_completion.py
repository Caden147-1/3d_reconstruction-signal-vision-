"""Post-reconstruction VideoScene proposals, fixed-pose MVS, and gated PLY merge."""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import shutil
import sqlite3
import subprocess
import time
import uuid

import cv2
import numpy as np

from colmap_workspace_check import contained_path, file_identity, inspect_provenance, inspect_sparse_model, write_provenance
from dense_cache import CACHE_RECORD, cache_identity, checked_dense_path, read_record, reconstruction_binding
from dense_surface_filter import positions, statistics_sample, vertex_chunks
from ply_io import parse_ply_header, validate_ply
import videoscene_geometry as geometry
from videoscene_conditioning import condition_holes
from videoscene_runner import readiness, CODE_REVISION, isolated_python, runtime_readiness, runtime_preflight
from workspace_lock import WorkspaceBusyError, workspace_lock
from process_control import managed_process


MVS_INDICES = tuple(range(0, 49, 6))


def run_command(command, cwd, deadline, log):
    remaining = deadline-time.monotonic()
    if remaining <= 0:
        raise TimeoutError("VideoScene completion budget exhausted")
    with log.open("a", encoding="utf-8") as handle:
        handle.write("\n" + subprocess.list2cmdline(list(map(str, command))) + "\n")
        handle.flush()
        with managed_process(list(map(str, command)), cwd=cwd, stdout=handle, stderr=subprocess.STDOUT) as process:
            code = process.wait(timeout=remaining)
        if code:
            raise subprocess.CalledProcessError(code, command)


def route_compatible(first, second, metadata, visual_only):
    if visual_only:
        return True
    a, b = metadata.get(first["name"], {}), metadata.get(second["name"], {})
    contour = a.get("trajectory_contour_id")
    if not contour or contour != b.get("trajectory_contour_id") or not a.get("session") or a.get("session") != b.get("session"):
        return False
    try:
        distances = [float(row["trajectory_contour_distance_m"]) for row in (a, b)]
    except (KeyError, TypeError, ValueError):
        return False
    if not all(math.isfinite(v) for v in distances):
        return False
    # PDR units are uncalibrated; adjacent stops can exceed the distance cutoff.
    # Actual shared tracks and parallax are checked before preparing any job.
    try:
        adjacent = abs(int(a['point_index']) - int(b['point_index'])) == 1
    except (KeyError, TypeError, ValueError):
        adjacent = False
    return adjacent or abs(distances[0]-distances[1]) <= 2.


def candidate_pairs(views, metadata, visual_only):
    pairs = []
    for a, b in itertools.combinations(views, 2):
        if a["camera_id"] != b["camera_id"] or not route_compatible(a, b, metadata, visual_only):
            continue
        delta = cv2.Rodrigues(b["R"] @ a["R"].T)[0]
        angle = float(np.linalg.norm(delta))
        baseline = float(np.linalg.norm(a["C"]-b["C"]))
        if angle <= math.radians(15) and baseline > 1e-6:
            pairs.append((angle, a["name"], b["name"], a, b, baseline))
    pairs.sort(key=lambda p: p[:3])
    return pairs


def native_image(workspace, view):
    path = contained_path(workspace / "dense/images", view["name"])
    image = cv2.imread(str(path))
    if image is None or (image.shape[1], image.shape[0]) != tuple(view["size"]):
        raise ValueError(f"Undistorted image size/pose mismatch: {view['name']}")
    # No crop: K is rescaled using the same independent horizontal/vertical factors.
    return cv2.cvtColor(cv2.resize(image, (geometry.WIDTH, geometry.HEIGHT), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def hole_cells(mask, expected, view, cell_size):
    y, x, xyz = geometry.hole_positions(mask, expected, view)
    cells = [tuple(row) for row in np.floor(xyz / cell_size).astype(np.int64)]
    return y, x, cells


def prepare_jobs(workspace, cloud, run, executable, deadline, max_clips, prior_conditioning="endpoints"):
    from mvs_observations import read_model, pair_depth_support

    model = run / "observed_model"
    model.mkdir()
    run_command([executable, "model_converter", "--input_path", workspace / "dense/sparse",
                 "--output_path", model, "--output_type", "TXT"], run, deadline, run / "commands.log")
    views = geometry.read_cameras(model)
    with (workspace / "metadata/images_manifest.csv").open(newline="", encoding="utf-8") as handle:
        metadata = {row["image_file"].replace("\\", "/"): row for row in csv.DictReader(handle)}
    summary = read_record(workspace / "colmap_workspace_summary.json")
    pairs = candidate_pairs(views, metadata, bool(summary.get("visual_only")))
    calibrated, sparse_points = read_model(workspace / 'dense/sparse')
    by_name = {v['name']: v for v in calibrated}
    supported_pairs = []
    rejected_pairs = []
    for pair in pairs:
        first, last = pair[3], pair[4]
        a, b = metadata.get(first['name'], {}), metadata.get(last['name'], {})
        same_stop = bool(a.get('session') and a.get('global_point_index')
                         and a.get('session') == b.get('session')
                         and a.get('global_point_index') == b.get('global_point_index'))
        evidence = pair_depth_support(by_name[first['name']], by_name[last['name']], sparse_points)
        if same_stop or not evidence['usable']:
            rejected_pairs.append(dict(first=first['name'], last=last['name'], same_checkpoint=same_stop, **evidence))
        else:
            supported_pairs.append(pair)
    write_provenance(run / 'endpoint_geometry.json', dict(candidate_count=len(pairs),
                     accepted_count=len(supported_pairs), rejected=rejected_pairs))
    pairs = supported_pairs
    if not pairs:
        write_provenance(run / 'selection.json', dict(route_pairs=0, jobs=[],
                         reason='no_cross_checkpoint_endpoints_with_reliable_parallax'))
        return []
    with cloud.open("rb") as handle:
        point_count = parse_ply_header(handle)["vertex_count"]
    sample, _, _ = statistics_sample(cloud, point_count, max_samples=10000)
    ranked = []
    for pair in pairs:
        middle = geometry.interpolate_views(pair[3], pair[4], 3)[1]
        _, z, visible = geometry.project(sample, middle)
        if visible.sum() < 100:
            continue
        ratio = pair[-1] / float(np.median(z[visible]))
        if .025 <= ratio <= .18:
            ranked.append((abs(ratio-.08), pair))
    ranked.sort(key=lambda item: (item[0], item[1][1], item[1][2]))
    # Sampling only shortlists baselines. Rank actual, full-cloud gap coverage
    # before spending inference time; reuse calibrated endpoint projections.
    jobs, report, candidates, examined, endpoint_cache = [], [], [], [], {}
    cell_size = max(float(np.median([p[-1] for _, p in ranked]))*.25, 1e-6) if ranked else 1.

    def render(view):
        if time.monotonic() >= deadline:
            raise TimeoutError("Completion prior rendering budget exhausted")
        color, depth = geometry.render_cloud(cloud, view)
        mask, estimate = geometry.hole_gate(depth)
        return color, depth, mask, estimate

    def endpoint(view):
        if view["name"] not in endpoint_cache:
            endpoint_cache[view["name"]] = render(view)
        return endpoint_cache[view["name"]]

    for _, (_, _, _, first, last, baseline) in ranked[:24]:
        triplet = geometry.interpolate_views(first, last, 3)
        middle = triplet[1]
        rendered = render(middle)
        depth, mask = rendered[1:3]
        known = np.isfinite(depth)
        record = {"first": first["name"], "last": last["name"], "local_hole_pixels": int(mask.sum())}
        examined.append(record)
        if known.mean() < .35:
            record["status"] = "insufficient_observed_coverage"
            continue
        median = float(np.median(depth[known]))
        if not .025 <= baseline / median <= .18 or mask.sum() < 100:
            record["status"] = "insufficient_local_support_or_baseline"
            continue
        checks = [endpoint(first), rendered, endpoint(last)]
        checked = geometry.supported_holes(triplet, np.stack([v[1] for v in checks]),
                                           np.stack([v[2] for v in checks]), np.stack([v[3] for v in checks]))
        record["supported_hole_pixels"] = int(checked[1].sum())
        if checked[1].sum() < 100:
            record["status"] = "insufficient_endpoint_support"
            continue
        _, _, cells = hole_cells(checked[1], rendered[3], middle, cell_size)
        candidates.append(dict(first=first, last=last, baseline=baseline, median=median,
                               middle=rendered, cells=cells, record=record))
        record["status"] = "eligible"
        print(f"[VideoScene] Supported gap candidate: {record['supported_hole_pixels']:,} pixels ({first['name']} -> {last['name']})", flush=True)

    covered = set()
    while candidates and len(jobs) < max_clips:
        scores = [sum(cell not in covered for cell in c["cells"]) for c in candidates]
        index = max(range(len(candidates)), key=lambda i: scores[i])
        candidate = candidates.pop(index)
        if scores[index] < max(100, .35*len(candidate["cells"])):
            candidate["record"]["status"] = "overlaps_selected_region"
            continue
        first, last = candidate["first"], candidate["last"]
        candidate["record"]["novel_hole_pixels"] = scores[index]
        job = run / f"clip_{len(jobs):03d}"
        trajectory = geometry.interpolate_views(first, last)
        rgb, observed_rgb, depths, masks, expected, conditioning = [], [], [], [], [], []
        endpoints = np.stack([native_image(workspace, v) for v in (first, last)])
        endpoint_renders = [endpoint(first), endpoint(last)]
        endpoint_depths = np.stack([v[1] for v in endpoint_renders])
        endpoint_masks = np.stack([v[2] for v in endpoint_renders])
        endpoint_expected = np.stack([v[3] for v in endpoint_renders])
        print(f"[VideoScene] Rendering calibrated prior {len(jobs)+1}/{max_clips}: {first['name']} -> {last['name']}", flush=True)
        for index, view in enumerate(trajectory):
            if time.monotonic() >= deadline:
                raise TimeoutError("Completion prior rendering budget exhausted")
            if index in (0, 48):
                color, view_depth, hole, estimate = endpoint(first if index == 0 else last)
            elif index == 24:
                color, view_depth, hole, estimate = candidate["middle"]
            else:
                color, view_depth, hole, estimate = render(view)
            if index in MVS_INDICES:
                observed_rgb.append(color)
                depths.append(view_depth)
                masks.append(hole)
                expected.append(estimate)
            if prior_conditioning == "endpoints":
                supported = geometry.supported_holes(
                    [first, view, last], np.stack([endpoint_depths[0], view_depth, endpoint_depths[1]]),
                    np.stack([endpoint_masks[0], hole, endpoint_masks[1]]),
                    np.stack([endpoint_expected[0], estimate, endpoint_expected[1]]))[1]
                color, stats = condition_holes(color, view, supported, estimate, endpoints,
                                              [first, last], endpoint_depths, endpoint_masks,
                                              endpoint_expected, t=index/48)
                conditioning.append(stats)
            rgb.append(color)
        selected_views = [trajectory[i] for i in MVS_INDICES]
        depths, expected = np.stack(depths), np.stack(expected)
        masks = geometry.supported_holes(selected_views, depths, np.stack(masks), expected)
        new_cells = set()
        for i, view in enumerate(selected_views):
            y, x, cells = hole_cells(masks[i], expected[i], view, cell_size)
            novel = np.array([cell not in covered for cell in cells], dtype=bool)
            masks[i, y[~novel], x[~novel]] = False
            new_cells.update(cell for cell, keep in zip(cells, novel) if keep)
        if masks[4].sum() < 100 or (masks.sum(axis=(1, 2)) >= 100).sum() < 3:
            candidate["record"]["status"] = "insufficient_novel_multiview_support"
            continue
        expected[~masks] = 0
        job.mkdir()
        rgb = np.stack(rgb)
        # Keep unconditioned observed colors for the MVS context; warped hole
        # hypotheses condition the diffusion model only, not observed geometry.
        np.savez(job / "prior.npz", rgb=rgb, endpoints=endpoints, observed_rgb=np.stack(observed_rgb))
        np.savez(job / "geometry.npz", depth=depths, mask=masks, expected=expected,
                 K=np.stack([trajectory[i]["K"] for i in MVS_INDICES]),
                 R=np.stack([trajectory[i]["R"] for i in MVS_INDICES]),
                 C=np.stack([trajectory[i]["C"] for i in MVS_INDICES]))
        payload = {"first": first["name"], "last": last["name"], "baseline": candidate["baseline"],
                   "median_depth": candidate["median"], "hole_pixels": int(masks[4].sum()), "frames": 49,
                   "hole_pixels_by_view": masks.sum(axis=(1, 2)).tolist(),
                   "hole_policy": "two_axis_boundary_prior_with_multiview_and_endpoint_support",
                   "selection_policy": "novel_supported_gap_coverage_not_baseline_only",
                   "prior_conditioning": prior_conditioning,
                   "sensor_role": "same_contour_local_pair_selection_only",
                   "camera_policy": "interpolated_fixed_colmap_poses_not_sensor_pose_seeds"}
        write_provenance(job / "job.json", payload)
        if conditioning:
            write_provenance(job / "conditioning.json", {
                "policy": "calibrated_endpoint_rgb_for_supported_holes_only",
                "interpretation": "Expected depths guide RGB only; they are not new geometry observations",
                "frames": conditioning})
        cv2.imwrite(str(job / "holes.png"), masks[4].astype(np.uint8)*255)
        jobs.append(job)
        report.append(payload)
        covered.update(new_cells)
        candidate["record"]["status"] = "selected"
    write_provenance(run / "selection.json", {"route_pairs": len(pairs), "eligible_pairs": len(ranked),
                     "examined_pair_cap": 24, "coverage_cell_size_scene_units": cell_size,
                     "examined": examined, "jobs": report})
    return jobs


def triangulate_fixed_views(mvs, views, executable, deadline, log):
    database = mvs / "database.db"
    k = views[0]["K"]
    if not all(np.allclose(v["K"], k, atol=1e-10, rtol=0) for v in views):
        raise ValueError("A completion clip must use one calibrated camera model")
    params = ",".join(str(v) for v in (k[0, 0], k[1, 1], k[0, 2], k[1, 2]))
    run_command([executable, "feature_extractor", "--database_path", database, "--image_path", mvs / "images",
                 "--ImageReader.camera_model", "PINHOLE", "--ImageReader.single_camera", 1,
                 "--ImageReader.camera_params", params, "--FeatureExtraction.num_threads", 4,
                 "--SiftExtraction.max_image_size", 720, "--SiftExtraction.max_num_features", 4096], mvs, deadline, log)
    run_command([executable, "exhaustive_matcher", "--database_path", database,
                 "--FeatureMatching.guided_matching", 1, "--FeatureMatching.max_num_matches", 16384,
                 "--FeatureMatching.num_threads", 4], mvs, deadline, log)
    connection = sqlite3.connect(database.resolve().as_uri()+"?mode=ro", uri=True)
    try:
        records = {name: (iid, cid) for iid, cid, name in connection.execute("SELECT image_id,camera_id,name FROM images")}
    finally:
        connection.close()
    geometry.write_fixed_model(mvs / "fixed_poses", views, records)
    (mvs / "sparse").mkdir()
    run_command([executable, "point_triangulator", "--database_path", database, "--image_path", mvs / "images",
                 "--input_path", mvs / "fixed_poses", "--output_path", mvs / "sparse",
                 "--refine_intrinsics", 0, "--Mapper.fix_existing_frames", 1,
                 "--Mapper.ba_refine_sensor_from_rig", 0, "--Mapper.num_threads", 4], mvs, deadline, log)
    if inspect_sparse_model(mvs / "sparse")["point_count"] < 20:
        raise ValueError("Insufficient fixed-pose visual tracks for completion fusion")
    check = mvs / "pose_check"
    check.mkdir()
    run_command([executable, "model_converter", "--input_path", mvs / "sparse", "--output_path", check,
                 "--output_type", "TXT"], mvs, deadline, log)
    actual = {v["name"]: v for v in geometry.read_cameras(check)}
    for index, original in enumerate(views, 1):
        result = actual.get(f"{index:03d}.png")
        if result is None or any(not np.allclose(original[key], result[key], rtol=0, atol=1e-6) for key in ("K", "R", "C")):
            raise ValueError("Completion triangulation changed the fixed camera calibration")


def reconstruct_job(job, executable, deadline):
    with np.load(job / "geometry.npz", allow_pickle=False) as data:
        depths, masks, expected = data["depth"], data["mask"], data["expected"]
        views = [dict(K=k, R=r, C=c) for k, r, c in zip(data["K"], data["R"], data["C"])]
    with np.load(job / "prior.npz", allow_pickle=False) as data:
        prior = data["observed_rgb"] if "observed_rgb" in data else data["rgb"][list(MVS_INDICES)]
    mvs = job / "mvs"
    (mvs / "images").mkdir(parents=True)
    (mvs / "stereo").mkdir()
    for name in ("depth_maps", "normal_maps", "consistency_graphs"):
        (mvs / "stereo" / name).mkdir()
    for i, frame_index in enumerate(MVS_INDICES):
        generated = cv2.imread(str(job / "generated" / f"{frame_index:03d}.png"))
        if generated is None or generated.shape != (geometry.HEIGHT, geometry.WIDTH, 3):
            raise ValueError("Missing or wrongly sized generated view")
        generated = cv2.cvtColor(generated, cv2.COLOR_BGR2RGB)
        # A narrow coherent context gives MVS continuous texture near gaps.
        # Context pixels are never eligible for generated-point promotion.
        composite = geometry.completion_composite(prior[i], generated, depths[i], masks[i], expected[i])
        if not cv2.imwrite(str(mvs / "images" / f"{i+1:03d}.png"), cv2.cvtColor(composite, cv2.COLOR_RGB2BGR)):
            raise OSError("Failed to write completion view")
    triangulate_fixed_views(mvs, views, executable, deadline, job / "mvs.log")
    names = [f"{i+1:03d}.png" for i in range(len(views))]
    config = "".join(name+"\n"+", ".join(n for n in names if n != name)+"\n" for name in names)
    (mvs / "stereo/patch-match.cfg").write_text(config, encoding="ascii")
    (mvs / "stereo/fusion.cfg").write_text("\n".join(names)+"\n", encoding="ascii")
    valid = expected[expected > 0]
    if not len(valid):
        return None
    run_command([executable, "patch_match_stereo", "--workspace_path", mvs,
                 "--PatchMatchStereo.depth_min", float(valid.min()*.8),
                 "--PatchMatchStereo.depth_max", float(valid.max()*1.2),
                 "--PatchMatchStereo.max_image_size", 720, "--PatchMatchStereo.cache_size", 2,
                 "--PatchMatchStereo.geom_consistency", 1, "--PatchMatchStereo.filter_min_num_consistent", 3,
                 "--PatchMatchStereo.filter_min_triangulation_angle", 1], job, deadline, job / "mvs.log")
    # COLMAP worker errors can leave the process exit code at zero. Require each
    # geometric depth/normal output before interpreting fusion as successful.
    for folder, channels in (("depth_maps", 1), ("normal_maps", 3)):
        for name in names:
            path = mvs / "stereo" / folder / (name + ".geometric.bin")
            prefix = f"{geometry.WIDTH}&{geometry.HEIGHT}&{channels}&".encode("ascii")
            if not path.is_file() or path.stat().st_size != len(prefix)+geometry.WIDTH*geometry.HEIGHT*channels*4:
                raise ValueError(f"Missing/truncated geometric stereo result: {path}")
            with path.open("rb") as handle:
                if handle.read(len(prefix)) != prefix:
                    raise ValueError("Stereo output dimensions changed")
    raw = mvs / "fused.ply"
    run_command([executable, "stereo_fusion", "--workspace_path", mvs, "--input_type", "geometric",
                 "--StereoFusion.min_num_pixels", 3, "--StereoFusion.max_reproj_error", 1,
                 "--StereoFusion.max_depth_error", .01, "--StereoFusion.num_threads", 4,
                 "--StereoFusion.cache_size", 2, "--output_path", raw], job, deadline, job / "mvs.log")
    selected = []
    totals = dict.fromkeys(("raw_points", "three_hole_views", "any_hole_views", "three_supported_views",
                           "eligible_target_points", "endpoint_supported", "rejected_endpoint_support",
                           "rejected_free_space", "free_space_conflicts", "accepted_points"), 0)
    coverage = np.zeros_like(masks, dtype=bool)
    for chunk in vertex_chunks(raw):
        if time.monotonic() >= deadline:
            raise TimeoutError("Completion geometry validation budget exhausted")
        keep, stats = geometry.gate_points(positions(chunk), views, depths, masks, expected, diagnostics=True)
        for key, value in stats.items():
            totals[key] = totals.get(key, 0) + value
        selected.append(chunk[keep])
        for i, view in enumerate(views):
            pixels, z, good = geometry.project(positions(chunk[keep]), view)
            ids = np.flatnonzero(good)
            x, y = pixels[good].T
            supported = masks[i, y, x] & (np.abs(z[ids]-expected[i, y, x]) < .15*expected[i, y, x])
            coverage[i, y[supported], x[supported]] = True
    count = sum(map(len, selected))
    target_pixels = masks.sum(axis=(1, 2))
    covered_pixels = coverage.sum(axis=(1, 2))
    fraction = float(covered_pixels.sum() / max(int(target_pixels.sum()), 1))
    write_provenance(job / "geometry_validation.json", {**totals,
                     "eligible_for_merge": count >= 100,
                     "target_pixels_by_view": target_pixels.tolist(), "covered_pixels_by_view": covered_pixels.tolist(),
                     "target_projection_coverage": fraction, "low_coverage": fraction < .1,
                     "policy": "fixed_pose_mvs_one_hole_three_supported_views_both_endpoints_and_observed_free_space_check",
                     "coverage_interpretation": "Projected accepted point pixels, not completed surface area",
                     "interpretation": "Consistency checks do not establish real-world truth"})
    print(f"[VideoScene] Accepted {count:,}/{totals['raw_points']:,} local MVS points; target projection coverage {fraction:.1%}.", flush=True)
    if fraction < .1:
        print("[VideoScene warning] Low target coverage; successful inference does not imply effective hole completion.", flush=True)
    if count < 100:
        return None
    with raw.open("rb") as handle:
        properties = parse_ply_header(handle)["vertex_properties"]
    output = job / "accepted.ply"
    geometry.write_vertices(output, properties, selected, count)
    return output


def observed_source(workspace):
    dense = workspace / "dense"
    final = dense / "final.ply"
    summary = read_record(dense / "final_output_summary.json")
    if summary.get("source_kind") == "videoscene_completed_point_cloud":
        if file_identity(final) != summary.get("final_identity"):
            raise ValueError("Completed final changed since its provenance record")
        source = Path(summary["observed_source"])
        if not source.resolve().is_relative_to(dense.resolve()) or file_identity(source) != summary.get("observed_identity"):
            raise ValueError("Observed backup is missing or changed")
        return source, summary
    if summary.get("source_kind") != "filtered_point_cloud":
        raise ValueError("VideoScene completion requires a filtered observed final, not raw fused or a mesh")
    elements = validate_ply(final)
    if elements.get("face", 0) or elements["vertex"] != summary.get("vertex_count"):
        raise ValueError("Final PLY does not match its observed output record")
    if summary.get("final_identity") and file_identity(final) != summary["final_identity"]:
        raise ValueError("Observed final changed since its provenance record")
    return final, summary


def complete(args):
    workspace, repo = args.workspace.resolve(), args.repo.resolve()
    python = args.python or isolated_python(repo)
    if args.check:
        missing = readiness(repo, python)
        report = {"status": "missing_model", "missing": missing} if missing else runtime_readiness(repo, python)
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "runtime_ready_not_inference_tested" else 2
    with workspace_lock(workspace):
        dense = checked_dense_path(workspace)
        dense.mkdir(exist_ok=True)
        status_path = dense / "videoscene_summary.json"
        payload = {"status": "started", "model": "VideoScene", "generated_points": 0, "promoted": False,
                   "prior_conditioning": args.prior_conditioning,
                   "effective_config": {"scene_completion": "videoscene", "max_clips": args.max_clips,
                                        "budget_hours": args.budget_hours, "seed": args.seed,
                                        "merge_generated": bool(getattr(args, 'merge_generated', False)),
                                        "prior_conditioning": args.prior_conditioning}}
        write_provenance(status_path, payload)
        try:
            if not args.prepare_only:
                missing = readiness(repo, python)
                if missing:
                    raise ValueError("VideoScene weights/environment not ready (Git LFS pointers are not weights): " + ", ".join(missing))
                if not runtime_preflight(repo, python):
                    raise ValueError("VideoScene runtime is not ready; observed final unchanged")
            proof = inspect_provenance(workspace)
            cache = read_record(dense / CACHE_RECORD)
            if proof["status"] != "verified" or cache.get("status") != "complete" or cache.get("identity") != cache_identity(workspace):
                raise ValueError("Unverified or stale observed reconstruction; refusing cross-run camera/PLY fusion")
            cloud, original_summary = observed_source(workspace)
            binding = original_summary.get("reconstruction_binding")
            if binding:
                if binding != reconstruction_binding(dense):
                    raise ValueError("Final PLY belongs to different MVS inputs/cameras; refusing cross-run fusion")
                payload["observed_binding"] = "verified"
            elif args.prepare_only:
                payload["observed_binding"] = "legacy_unverified_preview_only"
                print("[VideoScene warning] Legacy final has no reconstruction binding; preparing previews only, not authorizing PLY promotion.", flush=True)
            else:
                raise ValueError("Legacy final has no reconstruction binding. Re-export a verified filtered result with current cameras before completion; a point count alone cannot authenticate it")
            if not args.prepare_only and not original_summary.get("final_identity"):
                raise ValueError("Final PLY has no recorded hash; refusing automatic completion")
            source_identity = file_identity(cloud)
            payload.update(observed_identity=source_identity, input_final_identity=original_summary.get("final_identity"),
                           reconstruction_binding=binding, inference_requested=not args.prepare_only,
                           completion_code_identity={name: file_identity(Path(__file__).with_name(name)) for name in
                               ("videoscene_completion.py", "videoscene_geometry.py", "videoscene_conditioning.py", "videoscene_runner.py", "mvs_observations.py")})
            summary = read_record(workspace / "colmap_workspace_summary.json")
            executable = Path(summary["colmap_exe"])
            if not executable.is_file():
                raise ValueError("COLMAP executable is missing")
            run = dense / "videoscene" / uuid.uuid4().hex
            if run.resolve() != run:
                raise ValueError("Completion output directory must not redirect through a symlink")
            run.mkdir(parents=True)
            payload["run"] = str(run)
            deadline = time.monotonic() + args.budget_hours*3600
            jobs = prepare_jobs(workspace, cloud, run, executable, deadline, args.max_clips, args.prior_conditioning)
            payload["clip_count"] = len(jobs)
            if not jobs or args.prepare_only:
                payload["status"] = "prepared_not_generated" if jobs else "no_supported_holes"
                write_provenance(status_path, payload)
                print("[VideoScene] " + payload["status"] + "; final.ply unchanged.", flush=True)
                return 0
            additions = []
            for index, job in enumerate(jobs):
                print(f"[VideoScene] Inferring clip {index+1}/{len(jobs)} with sequential CPU offload...", flush=True)
                run_command([python, Path(__file__).with_name("videoscene_runner.py"), "--repo", repo, "--job", job,
                             "--seed", args.seed+index], run, deadline, job / "inference.log")
                accepted = reconstruct_job(job, executable, deadline)
                if accepted:
                    additions.append(accepted)
            payload["clip_validation"] = [read_record(job / "geometry_validation.json") for job in jobs]
            if not additions:
                payload["status"] = "no_geometry_passed"
                write_provenance(status_path, payload)
                print("[VideoScene] No generated geometry passed validation; final.ply unchanged.", flush=True)
                return 0
            with additions[0].open("rb") as handle:
                properties = parse_ply_header(handle)["vertex_properties"]
            count = sum(validate_ply(p)["vertex"] for p in additions)
            observed_count = validate_ply(cloud)["vertex"]
            if count > observed_count*.05:
                raise ValueError("Generated additions exceed 5% of the scene; refusing automatic promotion")
            merged_additions = run / "generated_additions.ply"
            geometry.write_vertices(merged_additions, properties, itertools.chain.from_iterable(vertex_chunks(p) for p in additions), count)
            if file_identity(cloud) != source_identity:
                raise ValueError("Observed point cloud changed during completion")
            observed = run / "observed.ply"
            shutil.copy2(cloud, observed)
            if file_identity(observed) != source_identity:
                raise ValueError("Observed backup differs from the reconstruction snapshot")
            staged = run / "completed.ply"
            geometry.merge_clouds(observed, merged_additions, staged)
            if reconstruction_binding(dense) != binding or cache_identity(workspace) != binding["cache_identity"]:
                raise ValueError("MVS inputs/cameras changed during completion; final.ply unchanged")
            if time.monotonic() >= deadline:
                raise TimeoutError("Completion budget exhausted before promotion; final.ply unchanged")
            final_identity = file_identity(staged)
            if not getattr(args, 'merge_generated', False):
                payload.update(status="hypotheses_only", generated_points=count, observed_points=observed_count,
                               generated_source=str(merged_additions), generated_identity=file_identity(merged_additions),
                               hypothesis_preview=str(staged), hypothesis_preview_identity=final_identity,
                               promoted=False, code_revision=CODE_REVISION)
                write_provenance(status_path, payload)
                print(f"[VideoScene] Saved {count:,} hypothesis points separately: {merged_additions}; observed final unchanged.", flush=True)
                return 0
            output_summary = {**original_summary, "source_kind": "videoscene_completed_point_cloud",
                              "observed_source": str(observed), "observed_identity": source_identity,
                              "source": str(observed), "output": str(dense / "final.ply"),
                              "generated_source": str(merged_additions), "generated_vertex_count": count,
                              "generated_identity": file_identity(merged_additions),
                              "observed_vertex_count": observed_count,
                              "vertex_count": validate_ply(staged)["vertex"], "final_identity": final_identity,
                              "explanation": "Observed geometry preserved; appended VideoScene hypotheses are not measured wireless/visual structure."}
            # Record the intended promotion before the atomic PLY replacement.
            write_provenance(run / "promotion.json", output_summary)
            if file_identity(dense / "final.ply") != original_summary["final_identity"]:
                raise ValueError("Final PLY changed during completion; refusing to overwrite external edits")
            previous_final = run / "previous_final.ply"
            shutil.copy2(dense / "final.ply", previous_final)
            try:
                staged.replace(dense / "final.ply")
                write_provenance(dense / "final_output_summary.json", output_summary)
                payload.update(status="completed", generated_points=count, observed_points=observed_count,
                               generated_fraction=count/observed_count, promoted=True, code_revision=CODE_REVISION)
                write_provenance(status_path, payload)
            except (OSError, ValueError, KeyboardInterrupt):
                payload.update(status="failed", generated_points=0, promoted=False)
                try:
                    previous_final.replace(dense / "final.ply")
                    write_provenance(dense / "final_output_summary.json", original_summary)
                except OSError as rollback_error:
                    payload.update(promoted=None, rollback_error=str(rollback_error), recovery_backup=str(previous_final))
                    raise OSError("Completion rollback could not be committed; inspect final/backup hashes before retrying") from rollback_error
                raise
            try:
                previous_final.unlink(missing_ok=True)
            except OSError:
                print(f"[VideoScene warning] Completed successfully; redundant rollback backup retained: {previous_final}", flush=True)
            print(f"[VideoScene] Appended {count:,} synthetic points; completed result: {dense / 'final.ply'}", flush=True)
            print(f"[VideoScene] Before-completion backup: {observed}", flush=True)
            return 0
        except (OSError, ValueError, KeyError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
            payload.update(status="failed", error=str(exc))
            try:
                write_provenance(status_path, payload)
            except OSError as status_error:
                print(f"[VideoScene error] Could not persist failure status: {status_error}", flush=True)
            print(f"[VideoScene error] {exc}. See videoscene_summary.json; no fallback model was used.", flush=True)
            return 2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=Path(__file__).resolve().parent / "colmap_workspace")
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent / "VideoScene")
    parser.add_argument("--python", type=Path)
    parser.add_argument("--max-clips", type=int, default=3)
    parser.add_argument("--prior-conditioning", choices=("endpoints", "observed"), default="endpoints",
                        help="Condition supported holes with calibrated endpoint RGB, or use the original observed render")
    parser.add_argument("--budget-hours", type=float, default=4.)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--merge-generated", action="store_true",
                        help="Explicitly append generated hypotheses to final.ply; by default save separate hypothesis layers.")
    args = parser.parse_args(argv)
    if not 1 <= args.max_clips <= 8 or not math.isfinite(args.budget_hours) or not 0 < args.budget_hours <= 20:
        parser.error("Use 1..8 local clips and a finite completion budget in (0,20] hours")
    try:
        result = complete(args)
        if result == 0 and not args.check:
            from reconstruction_quality import refresh_quality_report
            try:
                refresh_quality_report(args.workspace.resolve())
            except (OSError, ValueError, RuntimeError) as exc:
                print(f"[VideoScene warning] Completion finished, but quality report refresh failed: {exc}", flush=True)
        return result
    except WorkspaceBusyError as exc:
        print(f"[VideoScene error] {exc}", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
