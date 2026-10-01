"""Estimate fuzzy capture extent and rank visual wall hypotheses with WiFi.

Scores are relative support, not calibrated occupancy probabilities. Interior
RSSI samples cannot identify an exterior wall by themselves. All output stays
in the authenticated reconstruction's relative frame; no vertices are cropped,
camera poses moved, or metric PDR calibration inferred.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import uuid

import numpy as np

from dense_surface_filter import statistics_sample
from mvs_observations import read_model
from ply_io import validate_ply
from reconstruction_diagnostics import stationary_group_center
from reconstruction_evaluation import Snapshot
from wifi_observations import fingerprint_confidence, fingerprint_similarity, visible_fingerprint
from workspace_lock import WorkspaceBusyError, locked_workspace


def fit_planes(points, threshold, max_planes=6, iterations=256, min_support=300):
    remaining = np.asarray(points, dtype=float)
    rng = np.random.default_rng(0)
    planes = []
    for _ in range(max_planes):
        if len(remaining) < min_support:
            break
        evaluation = remaining[rng.choice(len(remaining), min(len(remaining), 12000), replace=False)]
        best_count, best = 0, None
        # Explicit elementwise operations avoid large BLAS thread startup overhead.
        for _ in range(iterations):
            triple = remaining[rng.choice(len(remaining), 3, replace=False)]
            normal = np.cross(triple[1]-triple[0], triple[2]-triple[0])
            length = np.linalg.norm(normal)
            if length < 1e-10:
                continue
            normal /= length
            offset = -float(normal @ triple[0])
            count = int((np.abs(np.sum(evaluation * normal, axis=1)+offset) <= threshold).sum())
            if count > best_count:
                best_count, best = count, (normal, offset)
        if best is None:
            break
        normal, offset = best
        good = np.abs(np.sum(remaining * normal, axis=1)+offset) <= threshold
        if good.sum() < min_support:
            break
        for _ in range(3):
            support = remaining[good]
            center = support.mean(axis=0)
            _, _, vt = np.linalg.svd(support-center, full_matrices=False)
            normal = vt[-1]
            if normal[np.argmax(np.abs(normal))] < 0:
                normal = -normal
            offset = -float(normal @ center)
            good = np.abs(np.sum(remaining * normal, axis=1)+offset) <= threshold
            if good.sum() < min_support:
                break
        if good.sum() < min_support:
            break
        support = remaining[good]
        origin = support.mean(0)
        origin -= (origin @ normal + offset)*normal
        # Orthogonal plane basis, independent of an assumed world-up direction.
        axis = np.eye(3)[np.argmin(np.abs(normal))]
        u = np.cross(normal, axis); u /= np.linalg.norm(u)
        v = np.cross(normal, u)
        basis = np.array([u, v])
        projected = (support-origin) @ basis.T
        planes.append(dict(normal=normal, offset=offset, origin=origin, basis=basis,
            lower=projected.min(0), upper=projected.max(0),
            sampled_support_count=int(good.sum()),
            fit_rms_scene_units=float(np.sqrt(np.mean((support @ normal+offset)**2)))))
        remaining = remaining[~good]
    return planes


def checkpoint_anchors(rows, views):
    """One RF observation and one robust camera center per capture stop."""
    by_name = {v['name']: v for v in views}
    grouped = defaultdict(list)
    for row in rows:
        if row.get('source_type') == 'wifi_anchor_photo' and row.get('image_file') in by_name:
            grouped[(row['session'], row['global_point_index'])].append(row)
    anchors = []
    for (session, point), group in sorted(grouped.items()):
        row = group[0]
        confidence = fingerprint_confidence(row)
        if confidence <= 0:
            continue
        center, _, scatter = stationary_group_center([by_name[r['image_file']]['C'] for r in group])
        anchors.append(dict(session=session, checkpoint=point, center=center,
                            scatter=scatter, confidence=confidence, fingerprint=row))
    return anchors


def relative_frame(anchors):
    """Fit a capture plane; it is a floor proxy, not a gravity measurement."""
    centers = np.array([a['center'] for a in anchors])
    if len(centers) < 3:
        raise ValueError('Need at least three registered WiFi checkpoints')
    origin = np.median(centers, axis=0)
    _, singular, axes = np.linalg.svd(centers - origin, full_matrices=False)
    if singular[1] <= max(singular[0] * 1e-3, 1e-9):
        raise ValueError('Collinear checkpoints cannot define a 2D boundary frame')
    for i in range(2):
        if axes[i, np.argmax(np.abs(axes[i]))] < 0:
            axes[i] *= -1
    axes[2] = np.cross(axes[0], axes[1])
    xy = (centers - origin) @ axes[:2].T
    distances = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
    distances[distances <= 1e-9] = np.inf
    spacing = float(np.median(np.min(distances, axis=1)))
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Distinct checkpoint positions are required')
    return origin, axes, xy, spacing, float(singular[2] / singular[1])


def radio_links(anchors, xy, spacing):
    """RF-similar, locally adjacent stops may connect the fuzzy support region.

    Do not connect sessions, reused scans, or remotely similar fingerprints.
    Similarity is not converted to a physical distance or wall thickness.
    """
    links = []
    for i, left in enumerate(anchors):
        for j in range(i + 1, len(anchors)):
            length = float(np.linalg.norm(xy[i] - xy[j]))
            if not 1e-9 < length <= 3 * spacing:
                continue
            evidence = stable_radio_similarity(left['fingerprint'], anchors[j]['fingerprint'])
            if evidence['score'] > 0:
                links.append(dict(a=i, b=j, score=float(evidence['score']),
                                  full_score=evidence['full_score'],
                                  ap_dropout_stability=evidence['ap_dropout_stability'],
                                  shared_aps=evidence['shared_aps'], rssi_rms_db=evidence['rssi_rms_db']))
    return links


def stable_radio_similarity(left, right):
    """Conservative AP-dropout sensitivity, not a statistical confidence bound.

    Reuse the retrieval validity rules, including session and scan independence.
    A link must survive removing each shared AP; two-AP links cannot qualify.
    This check is local to boundary estimation and does not change visual pairs.
    """
    evidence = fingerprint_similarity(left, right)
    full_score = evidence['score']
    scores = [full_score]
    if full_score > 0:
        shared = visible_fingerprint(left).keys() & visible_fingerprint(right).keys()
        for ap in sorted(shared):
            a, b = dict(left), dict(right)
            a.pop(ap)
            b.pop(ap)
            scores.append(fingerprint_similarity(a, b)['score'])
    score = min(scores)
    return {**evidence, 'score': score, 'full_score': full_score,
            'ap_dropout_stability': score / full_score if full_score else 0.}


def support_field(points, xy, confidence, links, bandwidth):
    """Spatial evidence decays to unknown; RF similarity affects connections.

    Max aggregation avoids counting overlapping pair evidence as independent
    measurements. Missing fingerprints supply no positive or negative evidence.
    """
    points = np.asarray(points, dtype=float)
    result = np.zeros(len(points))
    for center, weight in zip(xy, confidence):
        d2 = np.sum((points - center) ** 2, axis=1)
        result = np.maximum(result, weight * np.exp(-d2 / (2 * bandwidth ** 2)))
    for edge in links:
        a, b = xy[edge['a']], xy[edge['b']]
        delta = b - a
        length2 = float(delta @ delta)
        if length2 <= 1e-18:
            continue
        t = np.clip((points - a) @ delta / length2, 0, 1)
        d2 = np.sum((points - a - t[:, None] * delta) ** 2, axis=1)
        result = np.maximum(result, edge['score'] * np.exp(-d2 / (2 * bandwidth ** 2)))
    return result


def joint_wall_score(visual, wireless, wifi_weight):
    """RF can promote a visual hypothesis, but cannot create a wall alone."""
    return float(visual + wifi_weight * wireless * visual * (1 - visual))


def density_boundary_prior(workspace, snapshot, *, mode='radio', seed=0):
    """Build capture support and its local reliability as separate quantities.

    Controls keep camera positions and per-stop quality fixed. Positions-only
    omits RF links; permuted shuffles complete fingerprints within each session.
    Neither control provides geometric ground truth.
    """
    if mode not in {'radio', 'positions-only', 'permuted'}:
        raise ValueError('Unknown WiFi density prior mode')
    workspace = Path(workspace).resolve()
    manifest = workspace / 'metadata/images_manifest.csv'
    if snapshot.identity(manifest) is None:
        return None, dict(status='unavailable', reason='missing_wifi_manifest')
    views, _ = read_model(workspace / 'dense/sparse', snapshot)
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        anchors = checkpoint_anchors(list(csv.DictReader(handle)), views)
    try:
        origin, axes, xy, spacing, tilt = relative_frame(anchors)
    except ValueError as error:
        return None, dict(status='unavailable', reason=str(error))
    if tilt > .25:
        return None, dict(status='unavailable', reason='nonplanar_capture_positions')
    confidence = np.array([a['confidence'] / (1 + (a['scatter'] / spacing)**2) for a in anchors])
    graph_anchors = [dict(a) for a in anchors]
    permutation = list(range(len(anchors)))
    if mode == 'permuted':
        rng = np.random.default_rng(seed)
        sessions = defaultdict(list)
        for i, anchor in enumerate(anchors):
            sessions[anchor['session']].append(i)
        for indices in sessions.values():
            for i, j in zip(indices, rng.permutation(indices)):
                permutation[i] = int(j)
                graph_anchors[i]['fingerprint'] = anchors[j]['fingerprint']
    links = [] if mode == 'positions-only' else radio_links(graph_anchors, xy, spacing)
    for edge in links:
        i, j = edge['a'], edge['b']
        # RF shape and local measurement quality have different roles. Avoid
        # treating a uniformly weak fingerprint as evidence of empty space.
        quality = np.sqrt(fingerprint_confidence(graph_anchors[i]['fingerprint']) *
                          fingerprint_confidence(graph_anchors[j]['fingerprint']))
        edge['score'] = float(np.clip(edge['score'] / quality, 0, 1))

    bandwidth = .6 * spacing
    influence_radius = 3 * spacing

    def field(xyz):
        points = (xyz-origin) @ axes[:2].T
        support = support_field(points, xy, np.ones(len(anchors)), links, bandwidth)
        reliability = np.zeros(len(points))
        for center, weight in zip(xy, confidence):
            distance2 = np.sum((points-center)**2, axis=1)
            taper = np.maximum(0., 1-distance2/influence_radius**2)**2
            reliability = np.maximum(reliability, weight*taper)
        return {'support': support, 'confidence': reliability}

    return field, dict(status='available', checkpoint_count=len(anchors), link_count=len(links),
                       algorithm='confidence_gated_ap_dropout_v2', mode=mode,
                       permutation_seed=seed if mode == 'permuted' else None,
                       fingerprint_assignment=permutation,
                       spacing_scene_units=spacing, bandwidth_scene_units=bandwidth,
                       influence_radius_scene_units=influence_radius,
                       links=links, checkpoint_confidence=confidence.tolist(),
                       coordinate_frame='relative_visual_capture_plane',
                       interpretation=('Fuzzy capture support; weak or distant RF evidence reduces the '
                           'penalty. Positions-only retains per-stop quality but omits RSSI links. '
                           'The radius and bandwidth are heuristic, not learned wall distances.'))


def wall_candidates(sample, origin, axes, xy, confidence, links, spacing, wifi_weight,
                    *, diagnostics=None):
    span = float(np.linalg.norm(np.ptp(xy, axis=0)))
    tolerance = max(span * .003, 1e-8)
    minimum = max(100, len(sample) // 250)
    if diagnostics is None:
        diagnostics = {}
    rejected = dict(orientation=0, extent=0, surface_coverage=0)
    diagnostics.update(status='insufficient_observed_geometry', sampled_point_count=len(sample),
                       minimum_sample_count=300, plane_count=0, accepted_count=0,
                       rejected=rejected, plane_tolerance_scene_units=tolerance,
                       minimum_plane_support=minimum)
    if len(sample) < 300:
        return []
    planes = fit_planes(sample, tolerance, max_planes=8, iterations=128, min_support=minimum)
    diagnostics.update(status='no_supported_planes', plane_count=len(planes))
    candidates = []
    for plane in planes:
        normal = plane['normal']
        if abs(normal @ axes[2]) > np.sin(np.deg2rad(20)):
            rejected['orientation'] += 1
            continue
        inliers = sample[np.abs(sample @ normal + plane['offset']) <= tolerance]
        local = (inliers - origin) @ axes.T
        horizontal_normal = axes[:2] @ normal
        horizontal_normal /= np.linalg.norm(horizontal_normal)
        direction = np.array([-horizontal_normal[1], horizontal_normal[0]])
        along = local[:, :2] @ direction
        lo, hi = np.quantile(along, [.05, .95])
        bottom, top = np.quantile(local[:, 2], [.05, .95])
        if hi - lo < .5 * spacing or top - bottom < .1 * span:
            rejected['extent'] += 1
            continue
        # A line/starburst should not pass merely by containing many points.
        # Require spread along the candidate and across its height.
        occupied, _, _ = np.histogram2d(along, local[:, 2], bins=(10, 4),
                                        range=((lo, hi), (bottom, top)))
        coverage = float(np.mean(occupied >= 2))
        if coverage < .35:
            rejected['surface_coverage'] += 1
            continue
        offset = float(np.median(local[:, :2] @ horizontal_normal))
        endpoints = np.array([lo * direction, hi * direction]) + offset * horizontal_normal
        query = np.linspace(endpoints[0], endpoints[1], 24)
        plus = support_field(query + .5 * spacing * horizontal_normal, xy, confidence, links, .6 * spacing)
        minus = support_field(query - .5 * spacing * horizontal_normal, xy, confidence, links, .6 * spacing)
        wireless = float(max(plus.mean(), minus.mean()))
        visual = coverage * np.exp(-plane['fit_rms_scene_units'] / tolerance)
        visual *= min(1., len(inliers) / (2 * minimum))
        joint = joint_wall_score(visual, wireless, wifi_weight)
        candidates.append(dict(endpoints=endpoints.tolist(), height_range=[float(bottom), float(top)],
            visual_score=float(visual), wifi_support=wireless, joint_score=joint,
            wifi_score_delta=joint-float(visual), sampled_support_count=len(inliers),
            surface_coverage=coverage, fit_rms_scene_units=plane['fit_rms_scene_units'],
            interpretation='Visual plane near wireless-supported capture space; wall semantics unverified'))
    diagnostics.update(accepted_count=len(candidates),
                       status='candidates_available' if candidates else
                       'all_planes_rejected' if planes else 'no_supported_planes')
    return sorted(candidates, key=lambda row: (-row['joint_score'], -row['sampled_support_count']))


def write_viewer(path, payload):
    """Self-contained top view with a live WiFi ablation and support contour."""
    template = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>WiFi 辅助范围与墙体候选</title><style>
body{margin:24px;background:#101825;color:#e5edf8;font:15px system-ui}h1{font-size:23px}
.layout{display:grid;grid-template-columns:minmax(500px,1fr) 300px;gap:24px}
canvas{width:100%;background:#192536;border-radius:10px}label{display:block;margin:18px 0}
input[type=range]{width:100%}.muted{color:#b6c6da;line-height:1.6}li{margin:12px 0}
</style><h1>WiFi 辅助范围与墙体候选</h1>
<p class="muted">模糊候选 · 相对坐标 · 未标定。低无线支持表示未知；采集点外围不等于墙体。当前重建误差会影响本图。</p>
<div class="layout"><canvas id="map" width="1000" height="850"></canvas><aside>
<label>无线支持轮廓 <span id="levelText"></span><input id="level" type="range" min="0.05" max="0.8" step="0.01" value="0.2"></label>
<label>WiFi 对墙体评分的影响 <span id="weightText"></span><input id="weight" type="range" min="0" max="1" step="0.05"></label>
<label><input id="points" type="checkbox" checked> 显示视觉点云投影</label>
<p class="muted">青色：无线支持区域<br>蓝圈：WiFi 采集点<br>橙线：视觉墙体候选，粗细随融合评分变化</p>
<p id="status" class="muted"></p><ol id="walls"></ol></aside></div>
<script>const data=__DATA__;
const canvas=document.getElementById('map'),ctx=canvas.getContext('2d');
const level=document.getElementById('level'),weight=document.getElementById('weight');weight.value=data.wifi_weight;
const [xmin,ymin,xmax,ymax]=data.bounds,scale=Math.min(920/(xmax-xmin),770/(ymax-ymin));
const xy=p=>[40+(p[0]-xmin)*scale,810-(p[1]-ymin)*scale];
function draw(){ctx.clearRect(0,0,1000,850);const n=data.grid_size,dx=(xmax-xmin)/n,dy=(ymax-ymin)/n;
for(let y=0;y<n;y++)for(let x=0;x<n;x++){let v=data.field[y*n+x],p=xy([xmin+x*dx,ymin+(y+1)*dy]);
ctx.fillStyle=`rgba(24,202,189,${Math.min(.6,v*.65)})`;ctx.fillRect(...p,dx*scale+1,dy*scale+1);
if(v>=+level.value && (x===0||y===0||x===n-1||y===n-1||data.field[y*n+x-1]<+level.value||data.field[(y-1)*n+x]<+level.value||data.field[y*n+x+1]<+level.value||data.field[(y+1)*n+x]<+level.value)){ctx.fillStyle='#5be5d4';ctx.fillRect(...p,dx*scale+1,dy*scale+1);}}
if(document.getElementById('points').checked){ctx.fillStyle='#9ba9bd65';for(let p of data.points){let q=xy(p);ctx.fillRect(...q,1.5,1.5)}}
ctx.font='13px system-ui';for(let a of data.anchors){let p=xy(a.xy);ctx.fillStyle='#85c2ff';ctx.beginPath();ctx.arc(...p,5,0,Math.PI*2);ctx.fill();ctx.fillText(a.checkpoint,p[0]+7,p[1]-7)}
const scored=data.walls.map((w,i)=>({...w,id:i+1,score:w.visual_score+(+weight.value)*w.wifi_support*w.visual_score*(1-w.visual_score)})).sort((a,b)=>b.score-a.score);
for(let w of scored){let a=xy(w.endpoints[0]),b=xy(w.endpoints[1]);ctx.strokeStyle='#ffb76b';ctx.lineWidth=1+6*w.score;ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();ctx.fillStyle='#ffcf96';ctx.fillText('W'+w.id,(a[0]+b[0])/2,(a[1]+b[1])/2)}
document.getElementById('levelText').textContent=level.value;document.getElementById('weightText').textContent=weight.value;
document.getElementById('status').textContent=`${data.anchors.length} 个无线采集点，${data.links.length} 条相似指纹连接。支持值与评分均不是概率或真实贡献百分比。`;
document.getElementById('walls').replaceChildren(...scored.map(w=>{let li=document.createElement('li');li.textContent=`W${w.id}：视觉 ${w.visual_score.toFixed(3)} → 融合 ${w.score.toFixed(3)}；无线支持 ${w.wifi_support.toFixed(3)}`;return li}));}
for(let id of ['level','weight','points'])document.getElementById(id).addEventListener('input',draw);draw();
</script></html>'''
    # Prevent source names in embedded JSON from terminating the script element.
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    path.write_text(template.replace('__DATA__', encoded), encoding='utf-8')


def boundary_source(workspace, binding, snapshot):
    """Prefer authenticated observed support, including when final is in a viewer."""
    from reconstruction_stages import verified_final

    dense = workspace / 'dense'
    receipt = snapshot.read_json(dense / 'point_filter_summary.json')
    layers = receipt.get('observation_layers', receipt.get('layers', {}))
    layer = layers.get('supported', {})
    if receipt.get('reconstruction_binding') == binding and layer.get('vertex_count', 0) > 0:
        path = Path(layer['path']).resolve()
        if not path.is_relative_to(dense.resolve()):
            raise ValueError('Observation support must be inside the dense workspace')
        if snapshot.identity(path) != layer.get('identity'):
            raise ValueError('Supported cloud changed since observation assessment')
        return path, 'observation_supported_points'
    final = verified_final(workspace, binding)
    if final is None:
        raise ValueError('Authenticated observed geometry is required')
    if final.get('source_kind') == 'videoscene_completed_point_cloud':
        return Path(final['observed_source']), 'observed_points_before_completion'
    return dense / 'final.ply', 'observed_final_points'


@locked_workspace('workspace')
def estimate_boundary(workspace, *, wifi_weight=.5, grid_size=96):
    from reconstruction_stages import verified_context

    if not np.isfinite(wifi_weight) or not 0 <= wifi_weight <= 1:
        raise ValueError('WiFi weight must be finite and between zero and one')
    if not isinstance(grid_size, int) or not 24 <= grid_size <= 256:
        raise ValueError('Grid size must be an integer between 24 and 256')
    workspace = Path(workspace).resolve()
    binding = verified_context(workspace)
    snapshot = Snapshot()
    source, source_kind = boundary_source(workspace, binding, snapshot)
    source_identity = snapshot.identity(source)
    views, _ = read_model(workspace / 'dense/sparse', snapshot)
    manifest = workspace / 'metadata/images_manifest.csv'
    if snapshot.identity(manifest) is None:
        raise ValueError('Workspace image/WiFi manifest is missing')
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        anchors = checkpoint_anchors(list(csv.DictReader(handle)), views)
    origin, axes, xy, spacing, tilt = relative_frame(anchors)
    confidence = np.array([a['confidence'] / (1 + (a['scatter'] / spacing)**2) for a in anchors])
    links = radio_links(anchors, xy, spacing)
    # Camera scatter weakens a connection as well as its endpoint kernels.
    for edge in links:
        i, j = edge['a'], edge['b']
        edge['score'] *= float(np.sqrt(confidence[i]*confidence[j] /
                                       (anchors[i]['confidence']*anchors[j]['confidence'])))
    count = validate_ply(source)['vertex']
    sample, stride, finite_count = statistics_sample(source, count, max_samples=80000)
    local = (sample - origin) @ axes.T
    # Bounded analysis window only; points outside it are not classified/deleted.
    lower = xy.min(0) - 2 * spacing
    upper = xy.max(0) + 2 * spacing
    inside = ((local[:, :2] >= lower) & (local[:, :2] <= upper)).all(1)
    sample, local = sample[inside], local[inside]
    xs = lower[0] + (np.arange(grid_size)+.5) * (upper[0]-lower[0])/grid_size
    ys = lower[1] + (np.arange(grid_size)+.5) * (upper[1]-lower[1])/grid_size
    xx, yy = np.meshgrid(xs, ys)
    query = np.column_stack((xx.ravel(), yy.ravel()))
    field = support_field(query, xy, confidence, links, .6 * spacing)
    unlinked = support_field(query, xy, confidence, [], .6 * spacing)
    print(f'[boundary] {len(anchors)} WiFi stops; fitting visual wall candidates...', flush=True)
    wall_diagnostics = {}
    walls = wall_candidates(sample, origin, axes, xy, confidence, links, spacing, wifi_weight,
                            diagnostics=wall_diagnostics)
    decision = snapshot.read_json(workspace / 'pdr_alignment_decision.json')
    warnings = ['Uncalibrated support scores, not occupancy probabilities or surveyed walls.',
                'Reconstruction poses determine this frame; geometry errors propagate to boundary hypotheses.',
                'Interior samples do not locate the exterior boundary; low support remains unknown.']
    if decision.get('status') != 'aligned':
        warnings.append('PDR alignment unavailable; no PDR coordinates were overlaid on the visual model.')
    if tilt > .25:
        warnings.append('Capture positions are not close to a plane; the top view may not represent a floor.')
    if not walls:
        warnings.append('No supported wall candidates: ' + wall_diagnostics['status'] +
                        '. The WiFi capture extent remains available; this does not imply no walls exist.')
    payload = dict(schema_version=1, status='unverified_boundary_hypotheses',
        coordinate_frame='relative_visual_capture_plane', units='scene_units_not_calibrated_metres',
        origin=origin.tolist(), axes=axes.tolist(), capture_plane_nonplanarity=tilt,
        wifi_weight=wifi_weight, spacing_scene_units=spacing, bandwidth_scene_units=.6*spacing,
        bounds=[*lower.tolist(), *upper.tolist()], grid_size=grid_size,
        field=field.round(5).tolist(),
        radio_link_effect=dict(increased_cell_count=int(np.count_nonzero(field > unlinked+1e-6)),
                               max_support_increase=float(np.max(field-unlinked))),
        anchors=[dict(session=a['session'], checkpoint=a['checkpoint'], xy=p.tolist(),
                      confidence=float(c), camera_scatter_scene_units=a['scatter'])
                 for a,p,c in zip(anchors,xy,confidence)],
        links=links, walls=walls, wall_diagnostics=wall_diagnostics, warnings=warnings,
        source=dict(path=str(source), kind=source_kind, identity=source_identity, vertex_count=count,
                    finite_vertex_count=finite_count, sample_stride=stride,
                    analysis_sample_count=len(sample)),
        reconstruction_binding=binding,
        score_semantics='RF confidence kernels plus similar-fingerprint corridors; visual planes promoted by nearby RF support',
        changes_observed_geometry=False,
        points=local[::max(1, (len(local)+9999)//10000), :2].round(5).tolist())
    payload['code_identity'] = {name: snapshot.identity(Path(__file__).with_name(name)) for name in
                              ('wifi_boundary.py', 'wifi_observations.py',
                               'reconstruction_diagnostics.py', 'mvs_observations.py')}
    payload['inputs'] = {str(p): identity for p, (identity, _) in snapshot.files.items()}
    snapshot.verify_unchanged()
    if verified_context(workspace) != binding:
        raise ValueError('Reconstruction changed during boundary estimation')
    output = workspace / 'wifi_boundary' / uuid.uuid4().hex
    output.mkdir(parents=True)
    (output / 'boundary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    write_viewer(output / 'viewer.html', payload)
    latest = workspace / 'wifi_boundary/latest.json'
    staged = latest.with_suffix('.pending')
    staged.write_text(json.dumps(dict(status=payload['status'], viewer=str(output/'viewer.html'),
                                     summary=str(output/'boundary.json')), indent=2)+'\n', encoding='utf-8')
    staged.replace(latest)
    print(f'[boundary] {len(walls)} wall candidates; viewer: {output / "viewer.html"}', flush=True)
    return payload, output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('colmap_workspace'))
    parser.add_argument('--wifi-weight', type=float, default=.5,
                        help='RF promotion of visual wall scores; zero gives the visual-only scores.')
    parser.add_argument('--grid-size', type=int, default=96)
    args = parser.parse_args(argv)
    try:
        estimate_boundary(args.workspace, wifi_weight=args.wifi_weight, grid_size=args.grid_size)
    except (OSError, ValueError, KeyError, WorkspaceBusyError) as exc:
        print(f'[boundary error] {exc}', flush=True)
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
