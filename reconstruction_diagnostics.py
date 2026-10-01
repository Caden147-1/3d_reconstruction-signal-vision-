"""Source-bound sparse recovery and sensor reliability diagnostics.

PDR checkpoints contribute once each. Cross-check residuals are consistency
measurements, never independent accuracy or a reason to move observed geometry.
"""
from collections import Counter, defaultdict
from itertools import combinations
import math

import numpy as np

from colmap_workspace_check import open_readonly_database
from reconstruction_evaluation import number_summary
from wifi_observations import fingerprint_confidence, fingerprint_similarity


def fit_similarity(source, target, weights=None):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("At least three paired 3D checkpoints are required")
    weights = np.ones(len(source)) if weights is None else np.asarray(weights, dtype=float)
    if (weights.shape != (len(source),) or not np.isfinite(weights).all()
            or np.any(weights < 0) or np.count_nonzero(weights) < 3):
        raise ValueError("Similarity needs at least three finite positive weights")
    weights = weights / weights.max()
    weights /= weights.sum()
    source_mean, target_mean = weights @ source, weights @ target
    x, y = source-source_mean, target-target_mean
    if not np.isfinite(x).all() or not np.isfinite(y).all() or min(np.linalg.matrix_rank(x), np.linalg.matrix_rank(y)) < 2:
        raise ValueError("Non-finite or collinear checkpoints cannot constrain alignment")
    if min(np.linalg.matrix_rank(x[weights > 0]), np.linalg.matrix_rank(y[weights > 0])) < 2:
        raise ValueError("Weighted checkpoints cannot constrain alignment")
    u, singular, vt = np.linalg.svd((y * weights[:, None]).T @ x)
    signs = np.ones(3)
    signs[-1] = np.linalg.det(u @ vt)
    rotation = (u * signs) @ vt
    scale = float(singular @ signs / (weights @ np.sum(x*x, axis=1)))
    if scale <= 0:
        raise ValueError("Invalid similarity scale")
    return scale, rotation, target_mean-scale*(rotation @ source_mean)


def transform(points, model):
    scale, rotation, translation = model
    return scale * (np.asarray(points) @ rotation.T) + translation


def robust_similarity(source, target, threshold, weights=None):
    source, target = np.asarray(source), np.asarray(target)
    weights = np.ones(len(source)) if weights is None else np.asarray(weights, dtype=float)
    if weights.shape != (len(source),) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Consensus weights must be finite and positive")
    best = None
    triples = combinations(range(len(source)), 3)
    if len(source) >= 3 and math.comb(len(source), 3) > 1024:
        rng, sampled = np.random.default_rng(0), set()
        while len(sampled) < 1024:
            sampled.add(tuple(sorted(map(int, rng.choice(len(source), 3, replace=False)))))
        triples = sorted(sampled)
    for indices in triples:
        try:
            model = fit_similarity(source[list(indices)], target[list(indices)])
        except ValueError:
            continue
        residuals = np.linalg.norm(transform(source, model)-target, axis=1)
        good = residuals <= threshold
        score = (int(good.sum()), float(weights[good].sum()), -float(weights @ np.minimum(residuals, threshold)))
        if best is None or score > best[0]:
            best = score, model, good
    minimum = max(4, math.ceil(.6 * len(source)))
    if best is None or best[2].sum() < minimum:
        return None
    model = best[1]
    good = best[2]
    for _ in range(5):
        try:
            model = fit_similarity(source[good], target[good], weights[good])
        except ValueError:
            return None
        updated = np.linalg.norm(transform(source, model)-target, axis=1) <= threshold
        if updated.sum() < minimum:
            return None
        if np.array_equal(updated, good):
            break
        good = updated
    return model, np.linalg.norm(transform(source, model)-target, axis=1)


def stationary_group_center(xyz):
    """Estimate one stop using positive Cauchy weights, retaining every camera.

    Distances and the adaptive scale rotate/scale with the reconstruction; no
    scene-unit cutoff or capture-specific image number is used.
    """
    xyz = np.asarray(xyz, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all():
        raise ValueError("Stationary group requires finite 3D camera centers")
    origin = xyz.mean(0)
    local = xyz - origin
    extent = float(np.max(np.linalg.norm(local, axis=1)))
    if extent == 0:
        return origin, np.ones(len(xyz)), 0.0
    local /= extent
    center = np.zeros(3)
    # Geometric median initialization avoids dependence on world-axis choices.
    for _ in range(100):
        distances = np.linalg.norm(local-center, axis=1)
        weights = 1 / np.maximum(distances, 1e-9)
        updated = np.average(local, axis=0, weights=weights)
        if np.linalg.norm(updated-center) < 1e-10:
            center = updated
            break
        center = updated
    distances = np.linalg.norm(local-center, axis=1)
    median = float(np.median(distances))
    scale = max(median + 1.4826 * float(np.median(np.abs(distances-median))), 1e-9)
    for _ in range(100):
        weights = 1 / (1 + (np.linalg.norm(local-center, axis=1) / scale)**2)
        updated = np.average(local, axis=0, weights=weights)
        if np.linalg.norm(updated-center) < 1e-10:
            center = updated
            break
        center = updated
    weights = 1 / (1 + (np.linalg.norm(local-center, axis=1) / scale)**2)
    distances = np.linalg.norm(local-center, axis=1) * extent
    order = np.argsort(distances)
    index = min(int(np.searchsorted(np.cumsum(weights[order]), .9 * weights.sum())), len(order)-1)
    return origin + extent * center, weights, float(distances[order[index]])


def alignment_failure_evidence(entries, threshold):
    """Explain failed consensus without weakening or approving the alignment."""
    source = np.asarray([e["center"] for e in entries])
    target = np.asarray([e["pdr"] for e in entries])
    required = max(4, math.ceil(.6 * len(entries)))
    result = {"required_consensus_count": required, "best_candidate_consensus_count": 0,
              "candidate_inlier_checkpoints": [], "diagnostic_only": True}
    if len(entries) < 4:
        return {**result, "reason": "insufficient_checkpoints"}
    try:
        model = fit_similarity(source, target)
        residuals = np.linalg.norm(transform(source, model) - target, axis=1)
        result.update(all_point_fit_scale=float(model[0]),
                      all_point_fit_residuals=number_summary(residuals),
                      checkpoint_residuals={e["checkpoint"]: float(r) for e, r in zip(entries, residuals)})
    except ValueError:
        return {**result, "reason": "degenerate_or_nonfinite_checkpoint_geometry"}
    best_score = (-1, -math.inf)
    triples = combinations(range(len(entries)), 3)
    if math.comb(len(entries), 3) > 1024:
        rng = np.random.default_rng(0)
        sampled = set()
        while len(sampled) < 1024:
            sampled.add(tuple(sorted(map(int, rng.choice(len(entries), 3, replace=False)))))
        triples = sorted(sampled)
    result["candidate_triplets_evaluated"] = 0
    for indices in triples:
        result["candidate_triplets_evaluated"] += 1
        try:
            candidate = fit_similarity(source[list(indices)], target[list(indices)])
        except ValueError:
            continue
        errors = np.linalg.norm(transform(source, candidate) - target, axis=1)
        mask = errors <= threshold
        score = (int(mask.sum()), -float(np.minimum(errors, threshold).sum()))
        if score > best_score:
            best_score = score
            result["best_candidate_consensus_count"] = score[0]
            result["candidate_inlier_checkpoints"] = [e["checkpoint"] for e, good in zip(entries, mask) if good]
    result["reason"] = "insufficient_global_consensus" if result["best_candidate_consensus_count"] < required else "consensus_unstable_after_refit"
    result["interpretation"] = "Residuals expose disagreement; they do not distinguish PDR drift, capture grouping error or visual geometry error and do not authorize alignment."
    return result


def checkpoint_motion_metadata(members):
    """Require agreement within a stop before trusting its route metadata."""
    result, issues = {}, []
    for field in ("point_elapsed_ts", "trajectory_contour_id"):
        values = {str(row.get(field, "") or "").strip() for row, _ in members}
        if "" in values or len(values) != 1:
            result[field] = None
            issues.append(("missing_" if "" in values else "conflicting_") + field)
            continue
        value = values.pop()
        if field == "point_elapsed_ts":
            try:
                value = int(value)
                if value < 0:
                    raise ValueError("negative timestamp")
            except ValueError:
                value = None
                issues.append("invalid_point_elapsed_ts")
        result[field] = value
    result["motion_metadata_issues"] = issues
    return result


def local_motion_diagnostics(entries):
    """Compare route-relative invariants without fitting or changing any pose.

    Contour boundaries and missing metadata remain explicit unsupported edges.
    Length ratios and unsigned turning angles need no shared coordinate frame.
    """
    result = {"diagnostic_only": True, "edges": [], "turns": [],
              "interpretation": "Relative consistency does not identify which sensor is wrong, calibrate metres, or authorize local pose corrections."}
    if len(entries) < 2:
        return {**result, "status": "insufficient_checkpoints"}
    times = [e.get("point_elapsed_ts") for e in entries]
    if any(t is None for t in times) or len(set(times)) != len(times):
        return {**result, "status": "missing_or_ambiguous_checkpoint_times"}
    ordered = sorted(entries, key=lambda e: e["point_elapsed_ts"])
    result["checkpoint_order"] = [e["checkpoint"] for e in ordered]
    vectors = []
    for a, b in zip(ordered, ordered[1:]):
        visual = np.asarray(b["center"]) - a["center"]
        pdr = np.asarray(b["pdr"]) - a["pdr"]
        visual_length, pdr_length = float(np.linalg.norm(visual)), float(np.linalg.norm(pdr))
        issues = sorted(set(a.get("motion_metadata_issues", []) + b.get("motion_metadata_issues", [])))
        if not a.get("trajectory_contour_id") or not b.get("trajectory_contour_id"):
            issues.append("contour_unavailable")
        elif a["trajectory_contour_id"] != b["trajectory_contour_id"]:
            issues.append("contour_break")
        if a.get("target_spread", 0) > 1e-6 or b.get("target_spread", 0) > 1e-6:
            issues.append("inconsistent_checkpoint_targets")
        if visual_length == 0 or pdr_length == 0:
            issues.append("zero_displacement")
        scatter = a["raw_scatter_scene_units"] + b["raw_scatter_scene_units"]
        # Weakly resolved movement stays present but contributes less evidence.
        resolution = 1 / (1 + (scatter / visual_length)**2) if visual_length else 0.0
        support = math.sqrt(a["group_soft_weight"] * b["group_soft_weight"]) * resolution
        edge = {"checkpoint_a": a["checkpoint"], "checkpoint_b": b["checkpoint"],
                "elapsed_gap_ms": b["point_elapsed_ts"] - a["point_elapsed_ts"],
                "contour_a": a.get("trajectory_contour_id"), "contour_b": b.get("trajectory_contour_id"),
                "visual_displacement_scene_units": visual_length, "pdr_displacement_units": pdr_length,
                "visual_endpoint_scatter_sum_scene_units": scatter,
                "motion_resolution_weight": resolution, "support_weight": support if not issues else 0.0,
                "status": "comparable" if not issues else "unavailable", "reasons": issues,
                "scale_pdr_per_scene_unit": pdr_length / visual_length if not issues else None}
        result["edges"].append(edge)
        vectors.append((visual, pdr))
    eligible = [e for e in result["edges"] if e["status"] == "comparable"]
    # Robust scale is a diagnostic reference, never a replacement alignment.
    def reference_scale(edges):
        if len(edges) < 3:
            return None
        values = sorted((math.log(e["scale_pdr_per_scene_unit"]), e["support_weight"]) for e in edges)
        total = sum(w for _, w in values)
        if total <= 0:
            return None
        cumulative = 0.0
        for value, weight in values:
            cumulative += weight
            if cumulative >= total / 2:
                return math.exp(value)
        return math.exp(values[-1][0])
    result["diagnostic_scale_pdr_per_scene_unit"] = reference_scale(eligible)
    for edge in result["edges"]:
        reference = reference_scale([other for other in eligible if other is not edge]) if edge in eligible else None
        edge["held_out_scale_ratio"] = edge["scale_pdr_per_scene_unit"] / reference if reference else None
    for i in range(1, len(vectors)):
        previous, current = result["edges"][i-1:i+1]
        turn = {"checkpoint": ordered[i]["checkpoint"], "status": "unavailable",
                "visual_turn_deg": None, "pdr_turn_deg": None, "difference_deg": None}
        if previous["status"] == current["status"] == "comparable":
            angles = []
            for left, right in zip(vectors[i-1], vectors[i]):
                cosine = (left / np.linalg.norm(left)) @ (right / np.linalg.norm(right))
                angles.append(math.degrees(math.acos(float(np.clip(cosine, -1, 1)))))
            turn.update(status="comparable", visual_turn_deg=angles[0], pdr_turn_deg=angles[1],
                        difference_deg=abs(angles[0]-angles[1]),
                        support_weight=min(previous["support_weight"], current["support_weight"]))
        result["turns"].append(turn)
    result.update(status="evaluated", comparable_edge_count=len(eligible),
                  unavailable_edge_count=len(result["edges"])-len(eligible),
                  comparable_turn_count=sum(t["status"] == "comparable" for t in result["turns"]))
    return result


def photo_grouping_diagnostics(manifest, centers, capture_mode="auto"):
    """Flag stationary grouping ambiguity without using PDR targets or changing labels."""
    if capture_mode not in {"auto", "stationary", "off"}:
        raise ValueError("Unknown capture mode")
    groups, metadata = defaultdict(list), {}
    for row in manifest:
        name = row.get("image_file")
        if (row.get("source_type") != "wifi_anchor_photo" or not row.get("global_point_index")
                or name not in centers or capture_mode == "off"
                or (capture_mode == "auto" and row.get("image_association_basis") != "ordered_checkpoint_same_location")):
            continue
        if name in metadata:
            raise ValueError("Duplicate checkpoint image in manifest")
        metadata[name] = row
        groups[(row.get("session", ""), str(row["global_point_index"]))].append(
            (name, np.asarray(centers[name]["center_scene_units"], dtype=float)))
    estimates = {}
    for key, members in groups.items():
        xyz = np.asarray([xyz for _, xyz in members])
        center, _, _ = stationary_group_center(xyz)
        estimates[key] = (center, float(np.quantile(np.linalg.norm(xyz-center, axis=1), .9)))
    reports = []
    for key, members in sorted(groups.items()):
        center, scatter = estimates[key]
        others = [other for other in sorted(groups) if other != key and other[0] == key[0]
                  and len(groups[other]) >= 3]
        spacing = sorted(float(np.linalg.norm(center-estimates[other][0])) for other in others)
        positive = [value for value in spacing if value > 0]
        local_spacing = float(np.median(positive[:3])) if positive else None
        candidates = []
        if len(members) >= 3 and local_spacing is not None:
            for name, xyz in members:
                own_distance = float(np.linalg.norm(xyz-center))
                nearest = min(others, key=lambda other: float(np.linalg.norm(xyz-estimates[other][0])))
                other_distance = float(np.linalg.norm(xyz-estimates[nearest][0]))
                if other_distance < .5 * own_distance and own_distance-other_distance > .25 * local_spacing:
                    candidates.append({"image_file": name, "nearer_checkpoint": nearest[1],
                                       "source_image_path": metadata[name].get("source_image_path", ""),
                                       "point_index": metadata[name].get("point_index", ""),
                                       "nearer_point_index": metadata[groups[nearest][0][0]].get("point_index", ""),
                                       "own_center_distance_scene_units": own_distance,
                                       "other_center_distance_scene_units": other_distance})
        ratio = scatter / local_spacing if local_spacing is not None else None
        flags = []
        if len(members) < 3 or local_spacing is None:
            flags.append("insufficient_distinct_group_support")
        if ratio is not None and ratio >= .5:
            flags.append("large_within_group_spread")
        if candidates:
            flags.append("photos_closer_to_other_group")
        overlapping = [other[1] for other in others
                       if float(np.linalg.norm(center-estimates[other][0])) <= scatter+estimates[other][1]]
        if overlapping:
            flags.append("overlapping_group_extents_or_revisit")
        reports.append({"session": key[0], "checkpoint": key[1], "registered_images": len(members),
                        "scatter_to_spacing": ratio, "flags": flags, "review_photos": candidates,
                        "overlapping_checkpoints": overlapping})
    return {"status": "disabled" if capture_mode == "off" else "evaluated" if reports else "unavailable",
            "checkpoint_count": len(reports), "flagged_checkpoint_count": sum(bool(r["flags"]) for r in reports),
            "priority_review_checkpoint_count": sum(
                "large_within_group_spread" in r["flags"] or bool(r["review_photos"]) for r in reports),
            "review_photo_count": sum(len(r["review_photos"]) for r in reports), "checkpoints": reports,
            "thresholds": {"scatter_to_spacing": .5, "other_to_own_distance": .5, "distance_margin_to_spacing": .25},
            "interpretation": "Diagnostic only. Camera error, revisits and grouping mistakes can all cause flags. No photo is removed or reassigned; no sensor timestamps or PDR targets are required."}


PHOTO_GROUPING_REVIEW_FIELDS = (
    "session", "point_index", "global_point_index", "image_slot", "image_name",
    "source_image_path", "image_file", "registered", "review_status", "group_flags",
    "nearer_point_index", "nearer_global_point_index", "own_center_distance_scene_units",
    "other_center_distance_scene_units", "linked_observations", "occupied_grid_cells",
    "grid_size", "scatter_to_spacing",
)


def photo_grouping_review_rows(manifest, centers, grouping, camera_support=None):
    """Flatten review evidence, retaining unregistered photos and original identities.

    Current labels are copied verbatim. 'Nearer' means geometric proximity only;
    these rows are not a corrected checkpoint map or independently verified poses.
    """
    groups = {(row["session"], str(row["checkpoint"])): row for row in grouping["checkpoints"]}
    candidates = {row["image_file"]: row for group in groups.values() for row in group["review_photos"]}
    support = (camera_support or {}).get("views", {})
    result, seen = [], set()
    for row in manifest:
        name = row.get("image_file")
        if row.get("source_type") != "wifi_anchor_photo" or not row.get("global_point_index") or not name:
            continue
        if name in seen:
            raise ValueError("Duplicate checkpoint image in review manifest")
        seen.add(name)
        group = groups.get((row.get("session", ""), str(row["global_point_index"])), {})
        candidate = candidates.get(name, {})
        registered = name in centers
        if not registered:
            status = "not_registered"
        elif grouping["status"] == "disabled":
            status = "stationary_check_disabled"
        elif not group:
            status = "stationary_check_unavailable"
        elif candidate:
            status = "position_review"
        elif "insufficient_distinct_group_support" in group["flags"]:
            status = "insufficient_group_support"
        else:
            status = "group_review" if group["flags"] else "no_flag"
        source = row.get("source_image_path") or ""
        view = support.get(name, {}) if registered else {}
        result.append({
            **{field: row.get(field, "") for field in ("session", "point_index", "global_point_index", "image_slot")},
            "image_name": source.replace("\\", "/").rsplit("/", 1)[-1],
            "source_image_path": source, "image_file": name, "registered": int(registered),
            "review_status": status, "group_flags": ";".join(group.get("flags", [])),
            "nearer_point_index": candidate.get("nearer_point_index", ""),
            "nearer_global_point_index": candidate.get("nearer_checkpoint", ""),
            **{field: candidate.get(field, "") for field in (
                "own_center_distance_scene_units", "other_center_distance_scene_units")},
            **{field: view.get(field, "") for field in ("linked_observations", "occupied_grid_cells", "grid_size")},
            "scatter_to_spacing": group.get("scatter_to_spacing"),
        })
    return result


def pdr_diagnostics(manifest, centers, max_residual=2.0, max_scatter=2.0, capture_mode="auto"):
    if capture_mode not in {"auto", "stationary", "off"}:
        raise ValueError("Unknown PDR capture mode")
    groups = defaultdict(list)
    skipped = Counter()
    seen = set()
    for row in manifest:
        if row.get("global_point_index") and row.get("image_file") in centers and row.get("source_type") == "wifi_anchor_photo":
            if row["image_file"] in seen:
                raise ValueError("Duplicate checkpoint image in manifest")
            seen.add(row["image_file"])
            if capture_mode == "off" or (capture_mode == "auto" and row.get("image_association_basis") != "ordered_checkpoint_same_location"):
                skipped["capture_mode_off" if capture_mode == "off" else "stationary_capture_unconfirmed"] += 1
                continue
            try:
                target = [float(row[key]) for key in ("pdr_x", "pdr_y", "pdr_z")]
            except (ValueError, KeyError, TypeError):
                skipped["invalid_pdr_target"] += 1
                continue
            if np.isfinite(target).all():
                groups[(row.get("session", ""), str(row["global_point_index"]))].append((row, target))
            else:
                skipped["invalid_pdr_target"] += 1
    reports, refs = [], []
    sessions = sorted({key[0] for key in groups})
    for session in sessions:
        entries = []
        for key in sorted(groups):
            if key[0] != session:
                continue
            members = groups[key]
            xyz = np.array([centers[r["image_file"]]["center_scene_units"] for r, _ in members])
            center, weights, scatter = stationary_group_center(xyz)
            distances = np.linalg.norm(xyz-center, axis=1)
            targets = np.array([target for _, target in members])
            entries.append(dict(checkpoint=key[1], registered_images=len(members), center=center.tolist(),
                **checkpoint_motion_metadata(members),
                pdr=np.median(targets, axis=0).tolist(), scatter_scene_units=scatter,
                raw_scatter_scene_units=float(np.quantile(distances, .9)),
                effective_camera_count=float(weights.sum()**2 / (weights @ weights)),
                camera_soft_weights={r["image_file"]: float(w) for (r, _), w in zip(members, weights)},
                target_spread=float(np.max(np.linalg.norm(targets-np.median(targets, axis=0), axis=1))),
                representative=members[int(distances.argmin())][0]["image_file"]))
        source = np.array([e["center"] for e in entries])
        target = np.array([e["pdr"] for e in entries])
        distances = np.linalg.norm(source[:, None, :] - source[None, :, :], axis=2)
        nonzero = distances[distances > 0]
        spacing = float(np.median(nonzero)) if len(nonzero) else None
        for i, entry in enumerate(entries):
            nearby = distances[i][distances[i] > 0]
            # A revisit or a single very close stop must not collapse the scale.
            local_spacing = float(np.median(np.sort(nearby)[:3])) if len(nearby) else spacing
            ratio = entry["raw_scatter_scene_units"] / local_spacing if local_spacing else None
            entry.update(local_spacing_scene_units=local_spacing, scatter_to_spacing=ratio,
                         group_soft_weight=1 / (1 + ratio**2) if ratio is not None else 1.0)
        group_weights = np.array([e["group_soft_weight"] for e in entries])
        fit = robust_similarity(source, target, max_residual, group_weights) if len(entries) >= 4 else None
        for i, entry in enumerate(entries):
            entry.update(accepted_reference=False, residual_pdr_units=None, loo_residual_pdr_units=None,
                         scatter_pdr_units=None, rejection="insufficient_consensus")
            if fit is None:
                continue
            model, residuals = fit
            entry["residual_pdr_units"] = float(residuals[i])
            entry["scatter_pdr_units"] = entry["scatter_scene_units"] * model[0]
            mask = np.arange(len(entries)) != i
            heldout = robust_similarity(source[mask], target[mask], max_residual, group_weights[mask]) if mask.sum() >= 4 else None
            if heldout is not None:
                entry["loo_residual_pdr_units"] = float(np.linalg.norm(transform(source[i:i+1], heldout[0])[0]-target[i]))
            if entry["target_spread"] > 1e-6:
                reason = "inconsistent_checkpoint_targets"
            elif residuals[i] > max_residual:
                reason = "high_fit_residual"
            elif heldout is None or entry["loo_residual_pdr_units"] > max_residual:
                reason = "fails_held_out_checkpoint_check"
            elif entry["scatter_pdr_units"] > max_scatter:
                reason = "camera_centers_disagree_with_stationary_capture"
            else:
                reason = "accepted"
            entry.update(rejection=reason, accepted_reference=reason == "accepted")
        accepted = [e for e in entries if e["accepted_reference"]]
        # A small residual subset must not silently become a strong alignment prior.
        usable = len(accepted) >= max(4, math.ceil(.6 * len(entries)))
        if usable:
            refs.extend(f"{e['representative']} " + " ".join(map(str, e["pdr"])) for e in accepted)
        reports.append(dict(session=session, status="reviewable_reference_subset" if usable else "withhold_alignment",
            checkpoint_count=len(entries), accepted_checkpoint_count=len(accepted), checkpoints=entries,
            local_motion=local_motion_diagnostics(entries),
            robust_fit_scale=None if fit is None else fit[0][0],
            fit_failure_diagnostics=alignment_failure_evidence(entries, max_residual) if fit is None else None))
    return dict(max_residual_pdr_units=max_residual, max_scatter_pdr_units=max_scatter, sessions=reports,
        photo_grouping=photo_grouping_diagnostics(manifest, centers, capture_mode),
        capture_mode=capture_mode, skipped_registered_image_counts=dict(skipped),
        weighting="Cauchy camera weights; one group contribution scaled by scatter relative to the median of up to three nearest distinct checkpoints",
        interpretation="Soft weights estimate stop centers and fit global alignment; they do not alter individual camera poses or bundle adjustment. PDR units are not independently calibrated metres."), refs


def wifi_diagnostics(manifest, rms_limit=12.0):
    groups = {}
    for row in manifest:
        if row.get("global_point_index") and row.get("source_type") == "wifi_anchor_photo":
            groups.setdefault((row.get("session", ""), row["global_point_index"]), row)
    valid = {key: row for key, row in groups.items() if fingerprint_confidence(row) > 0}
    pairs = []
    for (key_a, a), (key_b, b) in combinations(sorted(valid.items()), 2):
        comparison = fingerprint_similarity(a, b, rms_limit)
        pairs.append(dict(checkpoint_a=list(key_a), checkpoint_b=list(key_b), **comparison))
    return dict(checkpoint_count=len(groups), valid_checkpoint_count=len(valid),
        invalid_status_counts=dict(Counter(row.get("wifi_observation_status", "missing") for key, row in groups.items() if key not in valid)),
        rssi_similarity_scale_db=rms_limit, checkpoint_pairs=pairs,
        checkpoint_confidences={"|".join(key): fingerprint_confidence(row) for key, row in valid.items()},
        interpretation="RF retrieval only. Reused scans are not revisits; RSSI does not measure 3D accuracy. Video frames receive no sensor association without clock synchronization.")


def sparse_recovery_diagnostics(workspace, manifest, centers, snapshot):
    registered = set(centers)
    neighbors = defaultdict(list)
    db = workspace / "database.db"
    if snapshot.identity(db) is not None:
        snapshot.identity(workspace / "database.db-wal")
        connection = open_readonly_database(db)
        try:
            tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if {"images", "two_view_geometries"} <= tables:
                names = dict(connection.execute("SELECT image_id, name FROM images"))
                for pair_id, count in connection.execute("SELECT pair_id, rows FROM two_view_geometries WHERE rows >= 15"):
                    a, b = names.get(pair_id // 2147483647), names.get(pair_id % 2147483647)
                    if a and b:
                        neighbors[a].append((b, count)); neighbors[b].append((a, count))
        finally:
            connection.close()
    rows = []
    for row in manifest:
        name = row["image_file"]
        if name in registered:
            continue
        links = [(name, count) for name, count in neighbors[name] if name in registered]
        rows.append(dict(image_file=name, source_type=row["source_type"], verified_pair_count=len(neighbors[name]),
            registered_neighbor_count=len(links), total_inliers_to_registered=sum(count for _, count in links),
            action="retry_registration_after_expanded_matching" if len(links) >= 2 else "expand_matching_or_recapture_overlap",
            strongest_registered_neighbor=max(links, key=lambda p: p[1])[0] if links else ""))
    rows.sort(key=lambda r: (-r["registered_neighbor_count"], -r["total_inliers_to_registered"], r["image_file"]))
    return dict(registered_count=len(registered), unregistered_count=len(rows),
                unregistered_by_source=dict(Counter(row["source_type"] for row in rows)),
                candidates=rows, interpretation="Pair inliers indicate registration opportunities, not a guaranteed consistent camera pose.")
