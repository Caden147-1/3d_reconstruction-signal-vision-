"""Calibrated endpoint RGB conditioning for proposed, unmeasured hole surfaces."""

import cv2
import numpy as np

from videoscene_geometry import HEIGHT, WIDTH, hole_positions


def _camera(view, label):
    result = {}
    for key, shape in (("K", (3, 3)), ("R", (3, 3)), ("C", (3,))):
        value = np.asarray(view[key], dtype=np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{label}.{key} must be finite with shape {shape}")
        result[key] = value
    if (result["K"][0, 0] <= 0 or result["K"][1, 1] <= 0
            or not np.allclose(result["K"][2], (0, 0, 1))):
        raise ValueError(f"{label}.K must contain calibrated pinhole intrinsics")
    try:
        np.linalg.inv(result["K"])
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{label}.K must be invertible") from error
    return result


def condition_holes(prior, view, mask, expected, endpoints, endpoint_views,
                    endpoint_depths, endpoint_masks, endpoint_expected, *, t=None):
    """Warp actual endpoint RGB into holes supported by both endpoint views.

    ``expected`` supplies a hypothesized camera-space depth, not a measurement
    of missing geometry. It is used only to warp RGB; this function never adds
    points or modifies depths/masks. Endpoint support uses nearest depth/mask
    pixels, while actual endpoint RGB is sampled with bilinear interpolation.
    A measured endpoint depth must agree within 3%; an unmeasured endpoint hole
    must have a supported expected depth agreeing within 15%. Proposals in
    observed free space, behind an observed surface, or outside either endpoint
    image are left unchanged. Black RGB is valid image content.

    ``t`` is the last endpoint's blend weight in [0, 1]. If omitted, distance
    from the target camera to each endpoint determines the weight; coincident
    cameras use equal weights. Returns a new uint8 RGB image and JSON-safe
    diagnostics. Inputs are never modified.
    """
    arrays = {
        "prior": (prior, (HEIGHT, WIDTH, 3), np.uint8),
        "mask": (mask, (HEIGHT, WIDTH), np.bool_),
        "expected": (expected, (HEIGHT, WIDTH), None),
        "endpoints": (endpoints, (2, HEIGHT, WIDTH, 3), np.uint8),
        "endpoint_depths": (endpoint_depths, (2, HEIGHT, WIDTH), None),
        "endpoint_masks": (endpoint_masks, (2, HEIGHT, WIDTH), np.bool_),
        "endpoint_expected": (endpoint_expected, (2, HEIGHT, WIDTH), None),
    }
    checked = {}
    for name, (value, shape, dtype) in arrays.items():
        value = np.asarray(value)
        if value.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {value.shape}")
        if dtype is not None and value.dtype != np.dtype(dtype):
            raise ValueError(f"{name} must have dtype {np.dtype(dtype)}")
        if dtype is None and not np.issubdtype(value.dtype, np.number):
            raise ValueError(f"{name} must contain numeric depths")
        if dtype is None and np.iscomplexobj(value):
            raise ValueError(f"{name} must contain real depths")
        checked[name] = value
    prior, mask, expected = (checked[name] for name in ("prior", "mask", "expected"))
    endpoints, endpoint_depths, endpoint_masks, endpoint_expected = (
        checked[name] for name in ("endpoints", "endpoint_depths", "endpoint_masks", "endpoint_expected"))
    if len(endpoint_views) != 2:
        raise ValueError("endpoint_views must contain exactly two cameras")
    view = _camera(view, "view")
    endpoint_views = [_camera(camera, f"endpoint_views[{i}]") for i, camera in enumerate(endpoint_views)]
    if t is None:
        distances = [np.linalg.norm(view["C"]-camera["C"]) for camera in endpoint_views]
        total = sum(distances)
        t = distances[0]/total if total > 1e-12 else .5
    t = float(t)
    if not np.isfinite(t) or not 0 <= t <= 1:
        raise ValueError("t must be a finite blend weight in [0, 1]")

    valid_mask = mask & np.isfinite(expected) & (expected > 0)
    y, x, xyz = hole_positions(valid_mask, expected, view)
    count = len(x)
    supported = np.ones(count, dtype=bool)
    samples = []
    stats = {
        "policy": "both_endpoint_calibrated_rgb_warp_for_unmeasured_hole_conditioning",
        "target_hole_pixels": int(mask.sum()),
        "valid_expected_pixels": count,
        "conditioned_pixels": 0,
        "conditioned_fraction": 0.,
        "endpoint_supported_pixels": [],
        "endpoint_observed_support_pixels": [],
        "endpoint_prior_support_pixels": [],
        "endpoint_free_space_conflicts": [],
        "blend_t": t,
    }
    for i, camera in enumerate(endpoint_views):
        camera_xyz = (xyz-camera["C"]) @ camera["R"].T
        z = camera_xyz[:, 2]
        homogeneous = camera_xyz @ camera["K"].T
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            uv = homogeneous[:, :2]/z[:, None]
        valid = np.isfinite(uv).all(axis=1) & np.isfinite(z) & (z > 0)
        valid &= (uv[:, 0] >= 0) & (uv[:, 0] <= WIDTH-1)
        valid &= (uv[:, 1] >= 0) & (uv[:, 1] <= HEIGHT-1)
        ids = np.flatnonzero(valid)
        px, py = np.rint(uv[ids]).astype(np.intp).T
        observed = endpoint_depths[i, py, px]
        estimate = endpoint_expected[i, py, px]
        measured = np.isfinite(observed) & (observed > 0)
        with np.errstate(invalid="ignore", over="ignore"):
            observed_support = measured & (np.abs(z[ids]-observed) <= .03*observed)
            prior_support = (~measured & endpoint_masks[i, py, px]
                             & np.isfinite(estimate) & (estimate > 0)
                             & (np.abs(z[ids]-estimate) <= .15*estimate))
            free_space = measured & (z[ids] < observed*.97)
        endpoint_supported = np.zeros(count, dtype=bool)
        endpoint_supported[ids] = (observed_support | prior_support) & ~free_space
        supported &= endpoint_supported
        stats["endpoint_supported_pixels"].append(int(endpoint_supported.sum()))
        stats["endpoint_observed_support_pixels"].append(int(observed_support.sum()))
        stats["endpoint_prior_support_pixels"].append(int(prior_support.sum()))
        stats["endpoint_free_space_conflicts"].append(int(free_space.sum()))

        # Full-size maps also support masks larger than OpenCV's 32767-pixel
        # limit for one remap dimension, without flattening pixels into a row.
        map_x = np.full((HEIGHT, WIDTH), -1, dtype=np.float32)
        map_y = np.full((HEIGHT, WIDTH), -1, dtype=np.float32)
        usable = np.flatnonzero(endpoint_supported)
        map_x[y[usable], x[usable]] = uv[usable, 0]
        map_y[y[usable], x[usable]] = uv[usable, 1]
        warped = cv2.remap(endpoints[i], map_x, map_y, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
        samples.append(warped[y, x].astype(np.float32))

    result = prior.copy()
    blended = (1-t)*samples[0][supported] + t*samples[1][supported]
    result[y[supported], x[supported]] = np.rint(blended).clip(0, 255).astype(np.uint8)
    stats["conditioned_pixels"] = int(supported.sum())
    stats["conditioned_fraction"] = stats["conditioned_pixels"]/max(stats["target_hole_pixels"], 1)
    return result, stats
