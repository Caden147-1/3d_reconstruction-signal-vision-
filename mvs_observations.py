"""Read calibrated COLMAP observations without changing the reconstruction.

Only undistorted perspective models are accepted. Image IDs are never treated as
array indices. Depth and normal maps use COLMAP's planar, Fortran-order layout.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import struct

import numpy as np

from colmap_workspace_check import CAMERA_MODELS, contained_path, read_exact
from reconstruction_metrics import Snapshot


def undistorted_intrinsics(kind, params):
    """Accept perspective distortion models only when distortion is exactly zero.

    COLMAP 3.13 keeps the original model name for IsUndistorted() cameras.
    Zero-coefficient fisheye models are not perspective pinhole projections.
    """
    layouts = {"SIMPLE_PINHOLE": (3, 3), "PINHOLE": (4, 4),
               "SIMPLE_RADIAL": (4, 3), "RADIAL": (5, 3),
               "OPENCV": (8, 4), "FULL_OPENCV": (12, 4)}
    count, intrinsic_count = layouts.get(kind, (0, 0))
    p = np.asarray(params, dtype=float)
    if (not count or p.shape != (count,) or not np.isfinite(p).all()
            or np.any(p[intrinsic_count:] != 0)):
        raise ValueError(f"Undistorted perspective camera required; got {kind} with parameters {p.tolist()}. "
                         "Run image_undistorter for cameras with nonzero distortion.")
    if intrinsic_count == 3:
        f, cx, cy = p[:3]
        return f, f, cx, cy
    return tuple(p[:4])


def rotation(q):
    q = np.asarray(q, float)
    if not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Invalid camera quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def read_model(model: Path, snapshot=None):
    snapshot = snapshot or Snapshot()
    cameras, images, points = {}, [], {}
    binary = (model / "cameras.bin").is_file()
    suffix = ".bin" if binary else ".txt"
    for stem in ("cameras", "images", "points3D"):
        if snapshot.identity(model / (stem + suffix)) is None:
            raise ValueError(f"Missing undistorted model file: {stem}{suffix}")
    for stem in ('frames', 'rigs'):
        snapshot.identity(model/(stem+suffix))
    if binary:
        with (model / "cameras.bin").open("rb") as f:
            for _ in range(struct.unpack("<Q", read_exact(f, 8))[0]):
                cid, kind, w, h = struct.unpack("<iiQQ", read_exact(f, 24))
                name, count = CAMERA_MODELS[kind]
                cameras[cid] = (name, w, h, struct.unpack("<"+"d"*count, read_exact(f, 8*count)))
        with (model / "images.bin").open("rb") as f:
            for _ in range(struct.unpack("<Q", read_exact(f, 8))[0]):
                p = struct.unpack("<idddddddi", read_exact(f, 64))
                name = bytearray()
                while (byte := read_exact(f, 1)) != b"\0":
                    name.extend(byte)
                    if len(name) > 65536:
                        raise ValueError("Invalid image name")
                n = struct.unpack("<Q", read_exact(f, 8))[0]
                obs = np.frombuffer(read_exact(f, n*24), dtype=[("x", "<f8"), ("y", "<f8"), ("id", "<i8")])
                images.append((p[0], p[1:5], p[5:8], p[-1], name.decode("utf8"), obs["id"].copy()))
        with (model / "points3D.bin").open("rb") as f:
            for _ in range(struct.unpack("<Q", read_exact(f, 8))[0]):
                p = struct.unpack("<QdddBBBd", read_exact(f, 43))
                n = struct.unpack("<Q", read_exact(f, 8))[0]
                track = np.frombuffer(read_exact(f, n*8), dtype="<i4").reshape(-1, 2)
                points[p[0]] = (np.array(p[1:4]), p[-1], track[:, 0].copy())
    else:
        def lines(path):
            return [line.strip() for line in path.read_text(encoding="utf8").splitlines()
                    if line.strip() and not line.lstrip().startswith("#")]
        for line in lines(model / "cameras.txt"):
            cid, kind, w, h, *params = line.split()
            cameras[int(cid)] = (kind, int(w), int(h), list(map(float, params)))
        with (model / "images.txt").open(encoding="utf8") as f:
            for line in f:
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                p = line.split(maxsplit=9)
                obs = f.readline()
                if len(p) != 10 or obs == "":
                    raise ValueError("Malformed COLMAP image/observation record")
                fields = obs.split()
                if len(fields) % 3:
                    raise ValueError("Malformed image observations")
                images.append((int(p[0]), list(map(float, p[1:5])), list(map(float, p[5:8])),
                               int(p[8]), p[9].strip(), np.array(list(map(int, fields[2::3])), dtype=np.int64)))
        for line in lines(model / "points3D.txt"):
            p = line.split()
            if len(p) < 8 or (len(p)-8) % 2:
                raise ValueError("Malformed sparse point")
            points[int(p[0])] = (np.array(list(map(float, p[1:4]))), float(p[7]), np.array(list(map(int, p[8::2]))))
    views = []
    for iid, q, t, cid, name, ids in images:
        kind, w, h, params = cameras[cid]
        fx, fy, cx, cy = undistorted_intrinsics(kind, params)
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.]])
        r, t = rotation(q), np.asarray(t)
        if min(w, h, fx, fy) <= 0 or not np.isfinite(k).all() or not np.isfinite(t).all():
            raise ValueError("Invalid camera calibration")
        name = name.replace("\\", "/")
        contained_path(model.parent / "images", name)
        ids = np.array(sorted({int(i) for i in ids if i >= 0 and int(i) in points}), dtype=np.int64)
        errors = [points[int(i)][1] for i in ids]
        # One pixel is the sampling floor, not a scene-unit distance threshold.
        noise = max(1., float(np.median(errors))) if errors else 1.
        views.append(dict(id=iid, name=name, K=k, R=r, C=-r.T @ t, size=(w, h),
                          point_ids=ids, noise_px=noise))
    if not views or len({v['id'] for v in views}) != len(views) or len({v['name'] for v in views}) != len(views):
        raise ValueError("Missing/duplicate calibrated views")
    if any(not np.isfinite(p[0]).all() or not np.isfinite(p[1]) or p[1] < 0 for p in points.values()):
        raise ValueError("Invalid sparse geometry")
    snapshot.verify_unchanged()
    return sorted(views, key=lambda v: v['name']), points


def project(xyz, view, size=None):
    camera = (xyz-view['C']) @ view['R'].T
    z = camera[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        uv = (camera @ view['K'].T)[:, :2] / z[:, None]
    w, h = view['size']
    if size is not None:
        uv *= np.array(size) / [w, h]
        w, h = size
    good = np.isfinite(uv).all(1) & np.isfinite(z) & (z > 0)
    good &= (uv[:, 0] >= .5) & (uv[:, 0] < w-.5) & (uv[:, 1] >= .5) & (uv[:, 1] < h-.5)
    return uv, z, good


def read_map(path, channels):
    with path.open('rb') as f:
        header = bytearray()
        while header.count(b'&') < 3 and len(header) < 80:
            byte = f.read(1)
            if not byte:
                break
            header.extend(byte)
    if header.count(b'&') != 3:
        raise ValueError(f"Invalid stereo header: {path}")
    w, h, c = map(int, header[:-1].split(b'&'))
    if min(w, h) <= 0 or c != channels or path.stat().st_size != len(header)+w*h*c*4:
        raise ValueError(f"Invalid/truncated stereo map: {path}")
    data = np.memmap(path, dtype='<f4', mode='r', offset=len(header), shape=(w, h, c), order='F')
    return data.transpose(1, 0, 2)


class DepthStore:
    """Bounded map cache; authenticate each used map against its MVS receipt."""
    def __init__(self, dense, snapshot=None, expected=None, max_open=12):
        self.dense, self.snapshot = Path(dense), snapshot or Snapshot()
        self.expected, self.max_open, self.cache = expected, max_open, OrderedDict()

    def get(self, view):
        name = view['name']
        if name in self.cache:
            self.cache.move_to_end(name)
            return self.cache[name]
        maps = []
        for folder, channels in [('depth_maps', 1), ('normal_maps', 3)]:
            rel = 'stereo/'+folder+'/'+name+'.geometric.bin'
            path = contained_path(self.dense, rel)
            identity = self.snapshot.identity(path)
            if identity is None:
                if self.expected is not None and rel in self.expected:
                    raise ValueError(f"Missing authenticated observation: {rel}")
                return None
            if self.expected is not None and identity != self.expected.get(rel):
                raise ValueError(f"Observation differs from MVS receipt: {rel}")
            maps.append(read_map(path, channels))
        depth, normal = maps[0][:, :, 0], maps[1]
        h, w = depth.shape
        iw, ih = view['size']
        if normal.shape != (h, w, 3) or w > iw or h > ih or abs(w/iw-h/ih) > max(1/iw, 1/ih):
            raise ValueError("Stereo map dimensions do not match calibration")
        self.cache[name] = (depth, normal)
        while len(self.cache) > self.max_open:
            self.cache.popitem(last=False)
        return depth, normal

    def close(self):
        # Windows keeps mapped files locked even when an exception traceback
        # retains array references. Close mappings explicitly after assessment.
        for pair in self.cache.values():
            for array in pair:
                mapping = getattr(array, '_mmap', None)
                if mapping is not None and not mapping.closed:
                    mapping.close()
        self.cache.clear()


def pair_depth_support(first, second, points, minimum_tracks=30, minimum_angle=2.):
    """Measure translation support from common tracks in calibrated cameras."""
    shared = np.intersect1d(first['point_ids'], second['point_ids'], assume_unique=True)
    result = dict(shared_tracks=len(shared), usable=False, median_angle_degrees=0.)
    if len(shared) < minimum_tracks:
        return result
    selected = shared[np.linspace(0, len(shared)-1, min(256, len(shared)), dtype=int)]
    xyz = np.array([points[int(pid)][0] for pid in selected])
    a, b = xyz-first['C'], xyz-second['C']
    norms = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1)
    valid = (norms > 1e-12) & ((a @ first['R'][2]) > 0) & ((b @ second['R'][2]) > 0)
    if valid.sum() < minimum_tracks:
        return result
    cosines = np.sum(a[valid]*b[valid], axis=1) / norms[valid]
    angles = np.degrees(np.arccos(np.clip(np.abs(cosines), 0, 1)))
    median = float(np.median(angles))
    return dict(shared_tracks=len(shared), median_angle_degrees=median,
                usable=minimum_angle <= median <= 60., sampled_tracks=int(valid.sum()))


def depth_source_plan(views, points, metadata, max_sources=20):
    """Prefer cross-stop sources, requiring actual parallax even within a stop."""
    from itertools import combinations

    if not isinstance(max_sources, int) or max_sources < 2:
        raise ValueError('Depth estimation requires at least two source views')
    candidates = {v['name']: [] for v in views}
    for first, second in combinations(views, 2):
        evidence = pair_depth_support(first, second, points)
        if not evidence['usable']:
            continue
        a, b = metadata.get(first['name'], {}), metadata.get(second['name'], {})
        same_stop = bool(a.get('session') and a.get('global_point_index') not in (None, '')
                         and a.get('session') == b.get('session')
                         and a.get('global_point_index') == b.get('global_point_index'))
        for ref, src in ((first, second), (second, first)):
            candidates[ref['name']].append(dict(source=src['name'], same_checkpoint=same_stop, **evidence))
    stationary_refs = {
        name for name in candidates
        if metadata.get(name, {}).get('image_association_basis') == 'ordered_checkpoint_same_location'
        and metadata.get(name, {}).get('session')
        and metadata.get(name, {}).get('global_point_index') not in (None, '')
    }
    active = set(candidates)
    no_translation_sources = set()
    while active:
        weak = set()
        for name in active:
            available = [row for row in candidates[name] if row['source'] in active]
            if len(available) < 2:
                weak.add(name)
            if name in stationary_refs and not any(not row['same_checkpoint'] for row in available):
                weak.add(name)
                no_translation_sources.add(name)
        if not weak:
            break
        active -= weak
    selected = {name: sorted((row for row in rows if row['source'] in active),
                            key=lambda r: (r['same_checkpoint'], -r['shared_tracks'], r['source']))[:max_sources]
                for name, rows in candidates.items() if name in active}
    weak = sorted(set(candidates) - active)
    return dict(status='ready' if len(active) >= 3 else 'insufficient_parallax',
                minimum_shared_tracks=30, minimum_median_angle_degrees=2.,
                max_sources=max_sources, view_count=len(views), depth_reference_count=len(active),
                weak_views=weak, sources=selected,
                stationary_views_without_external_source=sorted(no_translation_sources),
                stationary_source_policy='Known same-location capture requires a geometrically usable source outside its checkpoint; unknown source locations are not certified translations.',
                interpretation='Internal geometric support only; low reprojection error does not establish true scene shape.')


def main(argv=None):
    import argparse
    import csv
    import json
    from contextlib import nullcontext
    from colmap_workspace_check import write_provenance
    from workspace_lock import workspace_lock

    parser = argparse.ArgumentParser(description='Inspect real parallax and select explicit PatchMatch sources.')
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--apply', action='store_true', help='Write sources only before any stereo maps exist.')
    parser.add_argument('--summary', type=Path, help='Optional separate diagnostic JSON; inspection defaults to console output only.')
    args = parser.parse_args(argv)
    workspace = args.workspace.resolve()
    with workspace_lock(workspace) if args.apply else nullcontext():
        dense = workspace / 'dense'
        if args.apply and (any((dense / 'stereo/depth_maps').rglob('*.bin'))
                           or any((dense / 'stereo/normal_maps').rglob('*.bin'))):
            raise ValueError('Existing depth maps must not be reinterpreted with different source views')
        snapshot = Snapshot()
        snapshot.identity(Path(__file__))
        views, points = read_model(dense / 'sparse', snapshot)
        manifest = workspace / 'metadata/images_manifest.csv'
        snapshot.identity(manifest)
        with manifest.open(encoding='utf-8-sig', newline='') as handle:
            metadata = {r['image_file']: r for r in csv.DictReader(handle)}
        plan = depth_source_plan(views, points, metadata)
        plan['inputs'] = {str(p): identity for p, (identity, _) in snapshot.files.items()}
        snapshot.verify_unchanged()
        summary = args.summary or (dense / 'depth_source_summary.json' if args.apply else None)
        if summary is not None:
            summary = summary.resolve()
            if summary.is_relative_to(dense / 'sparse') or summary in snapshot.files or summary.suffix != '.json':
                raise ValueError('Depth source report must be a separate JSON file outside model inputs')
            if not args.apply and summary == (dense / 'depth_source_summary.json').resolve():
                raise ValueError('Inspection must not overwrite the applied depth-source receipt')
        if args.apply and plan['status'] != 'ready':
            raise ValueError('Insufficient connected depth sources; inspect without --apply first')
        if summary is not None:
            summary.parent.mkdir(parents=True, exist_ok=True)
            write_provenance(summary, plan)
        print(json.dumps({k: plan[k] for k in ('status', 'view_count', 'depth_reference_count', 'weak_views')}), flush=True)
        if args.apply:
            lines = []
            for name, rows in plan['sources'].items():
                lines.extend((name, ', '.join(row['source'] for row in rows)))
            target = dense / 'stereo/patch-match.cfg'
            staged = target.with_suffix('.pending')
            staged.write_text('\n'.join(lines)+'\n', encoding='utf-8')
            staged.replace(target)
            fusion = dense / 'stereo/fusion.cfg'
            staged = fusion.with_suffix('.pending')
            staged.write_text('\n'.join(plan['sources'])+'\n', encoding='utf-8')
            staged.replace(fusion)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
