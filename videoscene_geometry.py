"""Fixed-camera geometry and conservative hole gates for VideoScene proposals."""

import math

import cv2
import numpy as np

from dense_surface_filter import positions, vertex_chunks
from ply_io import PLY_TYPES, parse_ply_header, validate_ply


WIDTH, HEIGHT, FRAME_COUNT = 720, 480, 49


def quaternion_rotation(q):
    q = np.asarray(q, dtype=float)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-8:
        raise ValueError("Invalid camera quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def rotation_quaternion(rotation):
    vector = cv2.Rodrigues(np.asarray(rotation, dtype=float))[0].ravel()
    angle = np.linalg.norm(vector)
    return np.r_[math.cos(angle/2), vector * (math.sin(angle/2)/angle if angle else .5)]


def read_cameras(model):
    """Read COLMAP's undistorted text model, not the distorted source cameras."""
    cameras = {}
    for line in (model / "cameras.txt").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        cid, kind, width, height, *params = line.split()
        p = list(map(float, params))
        if kind == "PINHOLE" and len(p) == 4:
            fx, fy, cx, cy = p
        elif kind == "SIMPLE_PINHOLE" and len(p) == 3:
            fx, cx, cy = p
            fy = fx
        else:
            raise ValueError("Completion requires an undistorted PINHOLE model")
        width, height = int(width), int(height)
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
        if width <= 0 or height <= 0 or min(fx, fy) <= 0 or not np.isfinite(k).all():
            raise ValueError("Invalid pinhole intrinsics")
        k[0] *= WIDTH / width
        k[1] *= HEIGHT / height
        cameras[int(cid)] = (k, width, height)
    views = []
    with (model / "images.txt").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            p = line.split(maxsplit=9)
            if len(p) != 10:
                raise ValueError("Malformed COLMAP image pose")
            cid = int(p[8])
            r = quaternion_rotation(list(map(float, p[1:5])))
            t = np.array(list(map(float, p[5:8])))
            if not np.isfinite(t).all() or cid not in cameras:
                raise ValueError("Invalid camera translation/reference")
            k, width, height = cameras[cid]
            views.append({"name": p[9].strip(), "camera_id": cid, "K": k,
                          "R": r, "C": -r.T @ t, "size": (width, height)})
            if handle.readline() == "":
                raise ValueError("Missing COLMAP image observation line")
    if len({v["name"] for v in views}) != len(views):
        raise ValueError("Duplicate camera names")
    return views


def interpolate_views(first, last, count=FRAME_COUNT):
    delta = cv2.Rodrigues(last["R"] @ first["R"].T)[0]
    return [{"K": (1-a)*first["K"]+a*last["K"],
             "R": cv2.Rodrigues(a*delta)[0] @ first["R"],
             "C": (1-a)*first["C"]+a*last["C"]} for a in np.linspace(0, 1, count)]


def project(xyz, view):
    camera = (xyz - view["C"]) @ view["R"].T
    z = camera[:, 2]
    p = camera @ view["K"].T
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = p[:, :2] / z[:, None]
    good = np.isfinite(uv).all(1) & np.isfinite(z) & (z > 1e-6)
    good &= (uv[:, 0] >= 0) & (uv[:, 0] < WIDTH-.5) & (uv[:, 1] >= 0) & (uv[:, 1] < HEIGHT-.5)
    pixel = np.zeros((len(xyz), 2), dtype=np.int32)
    pixel[good] = np.rint(uv[good]).astype(np.int32)
    return pixel, z, good


def render_cloud(cloud, view):
    """Full-cloud z-buffer; point sampling must not manufacture the hole mask."""
    depth = np.full((HEIGHT, WIDTH), np.inf, dtype=np.float32)
    color = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for chunk in vertex_chunks(cloud):
        if not {"red", "green", "blue"}.issubset(chunk.dtype.names):
            raise ValueError("VideoScene requires a colored observed point cloud")
        pixels, z, good = project(positions(chunk), view)
        pixels, z = pixels[good], z[good]
        rgb = np.column_stack([chunk[n][good] for n in ("red", "green", "blue")])
        flat = pixels[:, 1] * WIDTH + pixels[:, 0]
        order = np.lexsort((z, flat))
        flat, z, rgb = flat[order], z[order], rgb[order]
        first = np.r_[True, flat[1:] != flat[:-1]] if len(flat) else np.zeros(0, bool)
        flat, z, rgb = flat[first], z[first], rgb[first]
        closer = z < depth.ravel()[flat]
        depth.ravel()[flat[closer]] = z[closer]
        color.reshape(-1, 3)[flat[closer]] = rgb[closer]
    return color, depth


def hole_gate(depth, max_span=96):
    """Propose missing pixels bracketed by coherent observed surface on two axes.

    Connectivity to the image exterior is not evidence that a local gap is
    unbounded. Inverse-depth interpolation supports slanted surfaces but is
    only a validation prior, never geometry appended to the cloud.
    """
    known = np.isfinite(depth) & (depth > 0)
    mask = np.zeros(depth.shape, bool)
    expected = np.zeros(depth.shape, np.float32)
    if known.sum() < 100:
        return mask, expected
    density = cv2.boxFilter(known.astype(np.float32), -1, (5, 5), borderType=cv2.BORDER_CONSTANT)
    support = known & (density >= .4)
    estimates, valid_axes = [], []
    for axis in (0, 1):
        values, valid = np.moveaxis(depth, axis, -1), np.moveaxis(support, axis, -1)
        n = values.shape[-1]
        coordinate = np.broadcast_to(np.arange(n), values.shape)
        left = np.maximum.accumulate(np.where(valid, coordinate, -1), axis=-1)
        right = np.minimum.accumulate(np.where(valid, coordinate, n)[..., ::-1], axis=-1)[..., ::-1]
        lo = np.take_along_axis(values, left.clip(0, n-1), axis=-1)
        hi = np.take_along_axis(values, right.clip(0, n-1), axis=-1)
        span = right-left
        bracket = (left >= 0) & (right < n) & (span > 0) & (span <= max_span)
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = (coordinate-left) / np.maximum(span, 1)
            estimate = 1. / ((1-fraction)/lo + fraction/hi)
            bracket &= np.isfinite(estimate) & (estimate > 0) & (np.abs(hi-lo) <= .15*estimate)
        estimates.append(np.moveaxis(np.where(bracket, estimate, 0), -1, axis))
        valid_axes.append(np.moveaxis(bracket, -1, axis))
    a, b = estimates
    mean = (a+b)*.5
    mask = ~known & valid_axes[0] & valid_axes[1] & (np.abs(a-b) <= .08*mean)
    mask[:4] = mask[-4:] = False
    mask[:, :4] = mask[:, -4:] = False
    expected[mask] = mean[mask]
    return mask, expected


def hole_positions(mask, expected, view):
    y, x = np.nonzero(mask)
    rays = np.column_stack([x, y, np.ones(len(x))]) @ np.linalg.inv(view["K"]).T
    xyz = (rays * expected[y, x, None]) @ view["R"] + view["C"]
    return y, x, xyz


def supported_holes(views, depths, masks, expected):
    """Require agreement across views and both calibrated endpoint boundaries.

    These are projections of observed geometry, not independent measurements
    of the hole interior. Retain that distinction in all completion reports.
    """
    result = np.zeros_like(masks, dtype=bool)
    for index, view in enumerate(views):
        y, x, xyz = hole_positions(masks[index], expected[index], view)
        votes = np.zeros(len(x), np.int16)
        anchors = np.zeros(len(x), np.int16)
        conflict = np.zeros(len(x), bool)
        for j, other in enumerate(views):
            pixels, z, good = project(xyz, other)
            ids = np.flatnonzero(good)
            px, py = pixels[good].T
            observed = depths[j][py, px]
            prior = expected[j][py, px]
            measured = np.isfinite(observed) & (observed > 0)
            consistent = (measured & (np.abs(z[ids]-observed) <= .03*observed))
            consistent |= masks[j][py, px] & (prior > 0) & (np.abs(z[ids]-prior) <= .15*prior)
            votes[ids] += consistent
            if j in (0, len(views)-1):
                anchors[ids] += consistent
            conflict[ids] |= measured & (z[ids] < observed*.97)
        result[index, y, x] = (votes >= 3) & (anchors == 2) & ~conflict
    return result


def completion_composite(prior, generated, depth, mask, expected, context_radius=6):
    """Blend a narrow, depth-coherent context for MVS, not for PLY promotion."""
    if not mask.any():
        return prior.copy()
    kernel = (2*context_radius+1,)*2
    weight = cv2.boxFilter(mask.astype(np.float32), -1, kernel, normalize=False)
    local = cv2.boxFilter(np.where(mask, expected, 0).astype(np.float32), -1, kernel, normalize=False)
    local /= np.maximum(weight, 1)
    distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    context = np.isfinite(depth) & (weight > 0) & (np.abs(depth-local) <= .08*local)
    alpha = np.where(context, np.clip(1-distance/(context_radius+1), 0, 1), 0)
    alpha[mask] = 1
    return np.rint(prior*(1-alpha[..., None]) + generated*alpha[..., None]).astype(np.uint8)


def gate_points(xyz, views, depths, masks, expected, *, diagnostics=False):
    """Require a target hole and three consistent views of candidate geometry.

    A missing point need not project into a hole in every supporting view:
    measured surface depth also provides geometric support. Hole membership
    identifies where a point can contribute; it is not independent geometric
    evidence. Endpoint agreement and observed free space remain mandatory.
    """
    votes = np.zeros(len(xyz), dtype=np.int16)
    support = np.zeros(len(xyz), dtype=np.int16)
    anchors = np.zeros(len(xyz), dtype=np.int16)
    conflict = ~np.isfinite(xyz).all(1)
    for index, (view, depth, mask, estimate) in enumerate(zip(views, depths, masks, expected)):
        pixels, z, good = project(xyz, view)
        ids = np.flatnonzero(good)
        x, y = pixels[good].T
        observed = depth[y, x]
        # New points may be behind occluders, but never in measured free space.
        conflict[ids] |= np.isfinite(observed) & (z[ids] < observed*.97)
        hole_vote = mask[y, x] & (estimate[y, x] > 0) & (np.abs(z[ids]-estimate[y, x]) < .15*estimate[y, x])
        measured = np.isfinite(observed) & (np.abs(z[ids]-observed) <= .03*observed)
        votes[ids] += hole_vote
        support[ids] += hole_vote | measured
        if index in (0, len(views)-1):
            anchors[ids] += hole_vote | measured
    eligible = (votes >= 1) & (support >= 3)
    endpoint_supported = eligible & (anchors == 2)
    keep = endpoint_supported & ~conflict
    if diagnostics:
        return keep, {"raw_points": len(xyz), "three_hole_views": int((votes >= 3).sum()),
                      "any_hole_views": int((votes >= 1).sum()),
                      "three_supported_views": int((support >= 3).sum()),
                      "eligible_target_points": int(eligible.sum()),
                      "endpoint_supported": int(endpoint_supported.sum()),
                      "rejected_endpoint_support": int((eligible & (anchors != 2)).sum()),
                      "rejected_free_space": int((endpoint_supported & conflict).sum()),
                      "free_space_conflicts": int(conflict.sum()), "accepted_points": int(keep.sum())}
    return keep


def write_fixed_model(folder, views, database_images=None):
    folder.mkdir(parents=True, exist_ok=True)
    cameras, images = {}, []
    for index, view in enumerate(views, 1):
        k, r, c = view["K"], view["R"], view["C"]
        name = f"{index:03d}.png"
        image_id, camera_id = database_images[name] if database_images is not None else (index, index)
        camera = f"{camera_id} PINHOLE {WIDTH} {HEIGHT} {k[0,0]:.17g} {k[1,1]:.17g} {k[0,2]:.17g} {k[1,2]:.17g}\n"
        if camera_id in cameras and cameras[camera_id] != camera:
            raise ValueError("Database camera group has inconsistent fixed intrinsics")
        cameras[camera_id] = camera
        values = " ".join(f"{v:.17g}" for v in np.r_[rotation_quaternion(r), -r@c])
        images.append(f"{image_id} {values} {camera_id} {name}\n\n")
    (folder / "cameras.txt").write_text("".join(cameras.values()), encoding="ascii")
    (folder / "images.txt").write_text("".join(images), encoding="ascii")
    (folder / "points3D.txt").write_text("", encoding="ascii")


def write_vertices(path, properties, chunks, count):
    dtype = np.dtype([(name, "<"+PLY_TYPES[kind][0]) for name, kind in properties])
    with path.open("wb") as handle:
        header = ["ply", "format binary_little_endian 1.0", f"element vertex {count}"]
        header += [f"property {kind} {name}" for name, kind in properties]
        handle.write(("\n".join([*header, "end_header"]) + "\n").encode("ascii"))
        written = 0
        for chunk in chunks:
            out = np.zeros(len(chunk), dtype=dtype)
            for name in dtype.names:
                if name in chunk.dtype.names:
                    out[name] = chunk[name]
            out.tofile(handle)
            written += len(out)
    if written != count:
        raise ValueError("PLY vertex count changed while writing")
    validate_ply(path)


def merge_clouds(observed, additions, output):
    with observed.open("rb") as handle:
        header = parse_ply_header(handle)
    if any(e["count"] for e in header["elements"][1:]):
        raise ValueError("Completion cannot merge into a mesh")
    supported = {"x", "y", "z", "nx", "ny", "nz", "red", "green", "blue"}
    if not {p[0] for p in header["vertex_properties"]}.issubset(supported):
        raise ValueError("Refusing to discard unknown PLY properties")
    count = validate_ply(additions)["vertex"]

    def chunks():
        yield from vertex_chunks(observed)
        yield from vertex_chunks(additions)

    write_vertices(output, header["vertex_properties"], chunks(), header["vertex_count"] + count)
