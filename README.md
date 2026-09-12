# 3D Reconstruction Signal Vision Pipeline

这个项目把手机采集到的 WiFi、IMU、点位照片和路线视频整理成一条可运行的室内 3D 重建流水线。

核心目标：

- 用照片/视频抽帧给 COLMAP 生成点云输入
- 用 IMU/PDR 给图像提供粗位置约束
- 用 WiFi 点位生成指纹库，用于后续定位
- 用 WiFi/PDR 采样点生成粗边界和信号覆盖模型
- 用 constrained matching 减少室内重复纹理导致的 COLMAP 乱匹配

- Preprocess raw `.sess` WiFi/IMU logs into selected AP tables and PDR trajectories.
- Bind ordered checkpoint photos to WiFi/PDR sampling points.
- Extract route-video keyframes as weak visual bridge frames.
- Export a COLMAP-friendly image dataset and metadata tables.
- Build a P2 route-contour candidate graph from WiFi/PDR checkpoint order, then let P1 visual evidence select evaluated COLMAP pairs.
- Build WiFi fingerprint localization models and signal-boundary summaries.
- Generate sparse/dense COLMAP batch scripts, density-preserving sampled point clouds, optional dense meshes, and quality reports without synthetic scene completion.

## Claim Boundary

The main technical claim is an explainable signal-motion prior for indoor reconstruction, not signal-only 3D reconstruction. IMU/PDR supplies a relative route, heading seeds, an ordered route-contour constraint, and PDR alignment references. WiFi supplies checkpoint labels, fingerprint assets, and signal-boundary summaries. The WiFi-point hull is only an RSSI interpolation envelope, not a room outline. COLMAP remains the geometric reconstruction backbone, while `match_graph_edges.csv`, `pdr_trajectory_contour.csv`, `camera_pose_seed.csv`, `pdr_alignment_refs.txt`, and the quality reports make the WiFi/IMU contribution inspectable.

This README describes the reusable workflow rather than one fixed dataset result. Current-run numbers belong in `colmap_workspace/QUALITY_REPORT.md`, `colmap_workspace/quality_report.json`, and the evaluation section of the project summary. If the quality report says `needs_colmap_rerun`, use a clean sparse+dense rerun before treating reconstruction metrics as final.

## 1. 目录结构

推荐把数据放成这样：

```text
.
|-- main.py                         # Main launcher for preprocessing and optional COLMAP run
|-- preprocess_pipeline.py          # End-to-end preprocessing pipeline
|-- image_mapping.py                # Photo-to-WiFi point mapping helpers
|-- video_frames.py                 # Route video frame extraction and filtering
|-- reconstruction_export.py        # COLMAP dataset export
|-- colmap_bridge.py                # COLMAP workspace and batch-script generation
|-- reconstruction_quality.py       # Sparse/dense quality report and model selection
|-- image_similarity.py             # P1 visual evidence used to select evaluated COLMAP match pairs
|-- trajectory_contour.py           # P2 continuous IMU/PDR route-contour builder
|-- point_cloud_filter.py           # Conservative far/isolated dense outlier filter
|-- ply_io.py                       # Shared PLY vertex IO helpers
|-- wifi_positioning.py             # WiFi fingerprint localization model generation
|-- localize_wifi.py                # Localize a new WiFi scan
|-- run_colmap.py                   # Wrapper for generated COLMAP batch scripts
|-- pic/                            # Input checkpoint photos, ordered by filename
|-- video/                          # Input route videos
|-- *.sess/                         # Raw phone WiFi/IMU/movement session folders
|-- processed/                      # Generated preprocessing outputs
|-- reconstruction_dataset/         # Generated COLMAP-ready dataset
|-- colmap_workspace/               # Generated COLMAP workspace and scripts
```

输入数据：

- `pic/`：点位照片，按文件名数字顺序排列
- `video/`：路线视频，推荐放 1 段连续视频，例如 `route.mp4`
- `*.sess/`：手机采集的 WiFi、IMU、movement 数据
- `colmap-x64-windows-cuda/`：COLMAP 程序目录，里面需要有 `bin/colmap.exe`

默认采集规则：

```text
1 个 WiFi 点位 = 20 张照片
```

这 20 张照片会被强绑定到对应 WiFi 指纹点。`video/` 里的视频会被抽帧，作为弱绑定的视觉基底，帮助 COLMAP 把相邻点位连接起来。

如果 `pic/` 里照片数量超过 `WiFi 点位数 * images-per-wifi-point`，多出的照片不会被悄悄当成 WiFi 锚点使用；流水线会写入 `processed/unassigned_images.csv`，方便在报告里说明或回头重新分组。

## 2. 一键预处理

在 PowerShell 或 VS Code Terminal 里进入项目目录：

```powershell
cd C:\Users\ASUS\Desktop\3d_reconstruction-signal-vision-
```

运行：

```powershell
python main.py
```

`main.py` now defaults to the `final` reconstruction profile with `--colmap-load-profile low-peak`. This keeps the high-quality processing path enabled, but caps supported COLMAP stages to fewer threads and inserts short cooldown waits between heavy stages.

这一步会生成：

```text
processed/
reconstruction_dataset/
colmap_workspace/
```

如果看到：

```text
[done] preprocessing complete
```

At the start of `full`, `run_colmap.py` prints a rough stage-by-stage estimate for reset, sparse reconstruction, dense PatchMatch, fusion, filtering, meshing, and reports. It then asks whether to run the complete COLMAP pipeline. Old COLMAP outputs are still removed automatically after you confirm; cleanup itself does not ask for another prompt.

To run without the upfront prompt:

```powershell
python run_colmap.py full --yes
```

## 3. 流水线逻辑

### Step 1: 解析 WiFi 原始数据

输入：

```text
*.sess/wifiScanner.dat
```

During `full` and `dense` runs, the live estimator prints an immediate model preview as soon as COLMAP reports the dense view count. It then prints the first target-precision estimate as soon as the sample and model agree within the default 30-minute error target, or after 60 seconds at the latest:

```powershell
python run_colmap.py full --runtime-estimate-sample-views 3
python run_colmap.py full --runtime-estimate-first-update-seconds 60
python run_colmap.py full --runtime-estimate-target-error-minutes 30
python run_colmap.py full --no-runtime-estimate
python run_colmap.py full --no-upfront-estimate
```

## Pipeline

1. WiFi parsing
   - Reads raw WiFi scans from `.sess` data.
   - Selects stable access points.
   - Writes long/wide WiFi feature tables.

2. IMU/PDR parsing
   - Reads accelerometer, rotation, and movement logs.
   - Estimates steps, heading, and relative PDR coordinates.
   - Produces a time-ordered `pdr_x` / `pdr_y` centerline and bounded route corridor.

3. WiFi/PDR fusion
   - Assigns WiFi scans to PDR positions.
   - Treats WiFi as sparse fingerprints/checkpoint anchors on the route, not as room geometry.

4. Photo mapping
   - Groups ordered photos into checkpoint slots.
   - Binds each checkpoint photo to a WiFi/PDR point.
   - Writes image-to-WiFi mapping summaries.

5. Video frame extraction
   - Extracts route-video frames.
   - Filters blurry or near-duplicate frames.
   - Weakly assigns frames along the PDR route.

6. Reconstruction dataset export
   - Copies and renames COLMAP input images.
   - Writes `images_manifest.csv`, `camera_pose_seed.csv`, WiFi fingerprint tables, and boundary seeds.

7. WiFi positioning
   - Builds a KNN-style WiFi fingerprint localizer.
   - Writes localization evaluation, RSSI grid, and signal boundary summaries.

8. COLMAP workspace generation
   - Writes the raw P2 graph as `match_pairs_route.txt`, route plus visual-recovery candidates as `match_pairs_unfiltered.txt`, and the selected list as `match_pairs.txt`, plus `match_graph_edges.csv` and batch scripts.
   - Generates sparse, dense, automatic, GUI, and reset scripts.
   - Writes P1 visual evidence as `image_similarity_edges.csv`, `image_similarity_summary.json`, and the keep/drop audit `match_pair_filter.csv`.
   - Uses P2 route topology (adjacent anchors plus bounded along-route distance) as the primary checkpoint relation when trajectory metadata is present.

## Reconstruction Profiles

Use `--reconstruction-profile` to control speed/quality tradeoffs:

```powershell
python main.py --reconstruction-profile preview
python main.py --reconstruction-profile balanced
python main.py --reconstruction-profile final
python main.py --reconstruction-profile rtx4080
python main.py --reconstruction-profile complete
```

- `preview`: faster checks with lower image/feature limits.
- `balanced`: moderate image/feature limits.
- `final`: quality-oriented settings for moderate compute.
- `rtx4080`: recommended default for RTX 4080; balances dense resolution, feature count, and PatchMatch cost.
- `complete`: highest-completeness profile, using more frames/features and the `complete` dense profile.

The current default is `final` plus the `low-peak` load profile.

Load profiles control peak load without changing the reconstruction profile:

```powershell
python main.py --colmap-load-profile low-peak
python main.py --colmap-load-profile standard
python main.py --colmap-load-profile max
python main.py --colmap-num-threads 2 --colmap-stage-cooldown-s 20
```

P2 describes the observed route topology. It first forms a candidate graph from adjacent anchors on the same IMU/PDR contour, bounded along-route distance, nearby angular photo slots, and nearby video frames. A missing contour annotation only triggers fallback among unannotated anchors; it does not remove known contour breaks. `match_pairs_route.txt` preserves this route graph.

P1 evaluates route candidates with actual ORB matches: hash or color similarity alone cannot establish feature overlap or a near duplicate. Evaluated `weak_visual_support` and `no_feature_overlap` pairs are excluded; unreviewed or unreadable route pairs remain candidates. An all-rejected graph stays empty and stops constrained workspace preparation instead of silently restoring rejected pairs.

Bounded visual recovery also retrieves up to 12 non-route candidates per image by appearance, without hard route/slot restrictions. Only candidates passing ORB fundamental-matrix RANSAC support enter the selected list; COLMAP still performs its own matching and geometric verification. `--visual-recovery-neighbors 0` disables this recovery for ablation. `--image-similarity-max-pairs` caps route evaluations only; recovery has its separate neighbor budget. `--skip-image-similarity` disables both P1 and recovery, retaining the base route/order graph.

`match_pairs_unfiltered.txt` contains both candidate sources, and `match_pair_filter.csv` records the selection decisions. These counts describe prepared pairs, not actual verified COLMAP matches or measured reconstruction gains. The default `constrained` matcher imports the selected list. `--colmap-matcher hybrid` explicitly adds exhaustive visual matches outside this gate.

```powershell
python image_similarity.py --workspace colmap_workspace
python image_similarity.py --workspace colmap_workspace --max-pairs 200
python main.py --skip-image-similarity
```

P2 writes `processed/pdr_trajectory_contour.csv` and `pdr_trajectory_edges.csv`.
This is the observed IMU/PDR capture-route corridor: WiFi checkpoints attach to it
as sparse fingerprint anchors, and the matching graph follows its predecessor/
successor relations plus along-route distance instead of treating a WiFi-point
convex hull as a physical room contour. P2 never adds synthetic surfaces to
`dense/final.ply`.

Route-video frames inherit an interpolated position on that observed contour only
as weak matching evidence. They remain excluded from PDR model alignment, so an
unsynchronized video cannot masquerade as a calibrated IMU/PDR measurement.

When both checkpoint photos and extracted video frames are present, the generated
workspace places them under `images/anchors/` and `images/video/`. COLMAP then uses
one shared camera per source folder instead of forcing their different aspect ratios
and intrinsics into a single camera. Dense preflight requires database coverage of
the current image list, complete sparse image/camera/point records, and registered
image names belonging to the current inputs. Partial image registration is allowed.

Reflective surfaces are not completed synthetically. For a known mirror or glass
region, provide a manual `--colmap-mask-dir` with masks matching the generated
relative names, such as `anchors/point_001_slot_01_DSC00001.png` or
`video/video_base_000001_IMG_1735_frame_000078.png`.

## Useful Commands

Regenerate preprocessing outputs:

```powershell
python main.py
```

Run only COLMAP sparse:

```powershell
python run_colmap.py sparse
```

Run only COLMAP dense:

```powershell
python run_colmap.py dense
```

Open COLMAP GUI:

```powershell
python run_colmap.py gui
```

Reset COLMAP outputs without rebuilding:

```powershell
python run_colmap.py reset
```

This asks you to type `YES` before deleting old COLMAP outputs.

Localize a new WiFi scan from JSON:

```powershell
python localize_wifi.py --scan-json new_scan.json
```

Localize a new WiFi scan from CSV:

```powershell
python localize_wifi.py --scan-csv new_scan.csv
```

## Common Options

Change photos per WiFi checkpoint:

```powershell
python main.py --images-per-wifi-point 16
```

Skip video frames:

```powershell
python main.py --skip-video-frames
```

Use visual-only COLMAP preparation without PDR-distance fallback or PDR model alignment:

```powershell
python main.py --visual-only
```

Reduce dense compute cost:

```powershell
python main.py --colmap-dense-max-image-size 1600
```

Use a different dense profile:

```powershell
python main.py --colmap-dense-profile balanced
python main.py --colmap-dense-profile rtx4080
python main.py --colmap-dense-profile complete
```

P2 is the IMU/PDR trajectory-contour constraint, not a mesh-completion step.

Tighten same-point angular matching:

```powershell
python main.py --constrained-slot-window 1
```

Loosen neighboring-point matching:

```powershell
python main.py --constrained-point-window 3
```

## Key Outputs

Preprocessing outputs:

```text
processed/wifi_long.csv
processed/ap_stats.csv
processed/selected_aps.json
processed/wifi_wide_selected.csv
```

作用：

- 展开每一次 WiFi scan
- 筛选稳定 AP
- 生成 selected AP 的 RSSI 特征表

### Step 2: 解析 IMU 并生成 PDR 轨迹

输入：

```text
*.sess/7.dat
*.sess/8.dat
*.sess/11.dat
*.sess/movements.dat
```

输出：

```text
processed/pdr_steps.csv
processed/pdr_path.csv
processed/movement_events.csv
processed/movement_segments.csv
```

作用：

- 用线性加速度检测步伐
- 用旋转向量估计 heading
- 生成相对 PDR 坐标 `pdr_x`, `pdr_y`

### Step 3: 融合 WiFi 和 PDR

输出：

```text
processed/wifi_pdr_features.csv
```

作用：

- 每条 WiFi scan 绑定一个 PDR 位置
- 后续 WiFi 指纹库、边界和图像位置种子都基于这个表

### Step 4: 绑定点位照片和 WiFi 点

输入：

```text
pic/
processed/wifi_pdr_features.csv
```

输出：

```text
processed/image_wifi_points.csv
processed/image_wifi_long.csv
processed/image_wifi_summary.json
```

作用：

- 按照片编号顺序分组
- 每 20 张照片绑定到一个 `CHECKPOINT_PAUSE` WiFi 点位
- 点位照片是强 WiFi 标签，进入 WiFi 指纹库

### Step 5: 抽取视频基底帧

输入：

```text
video/
processed/image_wifi_points.csv
```

输出：

```text
processed/video_frames/
processed/video_frames_manifest.csv
processed/video_frames_summary.json
```

作用：

- 默认每 1 秒抽 1 帧
- 默认每段视频最多抽 240 帧
- 抽帧会按路线顺序弱绑定到 WiFi 点位之间
- 视频帧只作为 COLMAP 视觉基底，不进入 WiFi 指纹库

如果没有 `video/` 目录，代码会自动跳过，不会报错。

视频抽帧优先使用系统里的 `ffmpeg`。如果没有 `ffmpeg`，代码会尝试使用 Python OpenCV；两者都没有时会继续跑照片/WiFi 流程，但不会生成视频帧。

### Step 6: 导出 3D 重建数据集

输出：

```text
reconstruction_dataset/images/
reconstruction_dataset/images_manifest.csv
reconstruction_dataset/camera_pose_seed.csv
reconstruction_dataset/wifi_fingerprint_library.csv
reconstruction_dataset/wifi_boundary_seed.csv
```

作用：

- 把点位照片和视频帧复制并重命名成 COLMAP 友好的格式
- 给点位照片写入 PDR 初始位置和 yaw seed
- 给视频帧写入弱 PDR 插值位置
- 生成 WiFi 指纹库和粗边界

### Step 7: 生成 WiFi 定位模型和信号边界

输出：

```text
reconstruction_dataset/wifi_localization_model.json
reconstruction_dataset/wifi_localization_eval.csv
reconstruction_dataset/wifi_positioning_summary.json
reconstruction_dataset/wifi_signal_grid.csv
reconstruction_dataset/wifi_signal_boundary.csv
```

作用：

- `wifi_localization_model.json`：KNN WiFi 指纹定位模型
- `wifi_signal_grid.csv`：采样区域内的 RSSI 插值网格
- `wifi_signal_boundary.csv`：带信号强度统计的边界点

### Step 8: 准备 COLMAP workspace

输出：

```text
colmap_workspace/images/
colmap_workspace/image_list.txt
colmap_workspace/match_pairs.txt
colmap_workspace/pdr_alignment_refs.txt
colmap_workspace/run_colmap_sparse.bat
colmap_workspace/run_colmap_dense.bat
QUALITY_REPORT.md
quality_report.json
```

作用：

- `images/`：COLMAP 输入图像，包含点位照片和视频基底帧
- `match_pairs.txt`：受限匹配对，来自 WiFi 点位顺序、拍摄方向、PDR 距离和视频帧顺序
- `pdr_alignment_refs.txt`：点位照片的 PDR 位置种子，用于 sparse 模型对齐
- `metadata/`：WiFi 指纹、边界、定位模型、视频帧清单等

## 4. 运行 COLMAP

如果之前已经跑过 COLMAP，先清理旧数据库和旧模型：

```powershell
python run_colmap.py reset
```

它会要求输入：

```text
colmap_workspace/dense/final.ply
colmap_workspace/dense/final_output_summary.json
colmap_workspace/dense/point_filter_summary.json
```

`final.ply` is the dense model you normally open. The pipeline prefers a valid `fused_filtered.ply`. Filtering now measures voxel occupancy on all observed points, selects the dominant density-connected component, and retains original points in a small neighborhood around it. This removes many peripheral streaks and disconnected fragments before any optional sampling. Voxels are selection masks, not replacement vertices: positions, normals and colors are not averaged or invented. The filter uses NumPy, not SciPy. The generated dense batch checks this dependency before reconstruction and stops before final promotion if filtering fails.

Finalization checks the complete PLY payload, finite XYZ values, and mesh indices, stages a verified copy, and replaces `final.ply` before cleaning derived intermediates. Sampling audits compare the output with the point budget available **after geometry filtering**, so intentional noise removal is not mistaken for an underfilled sampler. Raw `fused.ply` is retained for comparison and recovery, as are unrelated PLY files. Manual finalization still permits raw fallback when a filtered source is unavailable. A failed promotion preserves source files and the previous final. This protection applies to finalization, not the deliberate output reset at the start of a full rebuild.

Meshing is disabled by default. `python main.py --colmap-meshing` explicitly enables Poisson/Delaunay inspection meshes and retains the intermediate files; `final.ply` still prefers the observed point cloud. Meshes can interpolate unobserved surfaces and are not evidence of wireless or visual observations. For mesh-oriented inspection, run `python finalize_dense_outputs.py --dense-dir colmap_workspace/dense --prefer-mesh --keep-intermediate` while the mesh artifact is retained. Outside mesh mode, `$env:KEEP_INTERMEDIATE_PLY = "1"` before running the generated batch also retains intermediates.

The `complete` dense profile, also used by the default `final` reconstruction profile, now retains all points that pass geometry filtering (`--max-points 0`) and writes binary PLY in bounded-memory chunks. The `balanced` and `rtx4080` profiles uniformly sample up to 0.6/0.9 million points **after** filtering. Full-resolution output reduces sampling gaps but costs more viewer memory. Binary PLY is supported by MeshLab; it is not intended for text-editor inspection.

This is an observed-density heuristic, not a semantic room mask or a calibration fix. Weak real surfaces can be removed, and dense reflection artifacts can survive. Ambiguous large components or removal of more than half the finite source points cause an explicit failure rather than silent acceptance. WiFi/IMU still constrain the capture route and alignment; the filter never treats that route as a room boundary and adds no mirror, floor or wall patches.

To rerun only filtering without changing the current final, write an inspection candidate:

```powershell
python point_cloud_filter.py --input colmap_workspace/dense/fused.ply --output colmap_workspace/dense/filtered_candidate.ply --summary colmap_workspace/dense/filtered_candidate_summary.json
```

Inspect it before promotion. `--core-quantile` (default `0.15`) controls the point-weighted dense-core threshold; larger values are stricter. `--support-halo` (default `1`) retains nearby observed detail in voxel-cell units. `--surface-cell-ratio` (default `0.0025`) sets the cell width relative to the robust scene diagonal. These distances are model units, not verified physical meters. `--method legacy` retains the previous weak bbox/radial/neighborhood implementation for comparisons.

## Quality Reports

Run:

```powershell
python reconstruction_quality.py --workspace colmap_workspace --summary quality_report.json --markdown QUALITY_REPORT.md
```

The report checks:

- Input image count.
- COLMAP database image count.
- Sparse component count and registered-image ratio.
- Final dense model availability and vertex/face counts.
- Intermediate dense point cloud, filtered point cloud, and mesh availability when those files are kept.
- Whether P2's route-contour constraint was available and how many actual candidate edges it contributed.
- P1 prepared-pair selection, separate P2/recovery counts, and whether the configured matcher enforces that pair list; these are not measured reconstruction gains.
- Match-pair reason counts.

If the report says `database image count does not match image_list.txt`, rebuild with:

```powershell
python run_colmap.py sparse
python run_colmap.py dense
```

The launcher clears stale COLMAP outputs before rebuilding sparse. A previous `final.ply` can still exist as diagnostic evidence, but it should not be treated as the final clean-run result while the report status is `needs_colmap_rerun`.

## Troubleshooting

### Existing COLMAP outputs are detected

Use the Python launcher for normal rebuilds:

```powershell
python run_colmap.py full
```

It automatically removes stale `database.db`, sparse, dense, mesh, and report outputs before rebuilding. If you run the generated `.bat` scripts directly and they report existing outputs, run `colmap_workspace\reset_colmap_outputs.bat` first; it asks you to type `YES` before deleting old outputs.

### Dense reconstruction says sparse model is missing

Run sparse first:

```powershell
python run_colmap.py sparse
```

成功后运行 dense：

```powershell
python run_colmap.py dense
```

也可以完整运行：

```powershell
python run_colmap.py full
```

或者直接用 bat：

```powershell
.\run_colmap_full.bat
```

Open `colmap_workspace/dense/final.ply`, and inspect `point_filter_summary.json` plus `final_output_summary.json` if the final model still looks noisy. If noise is still high, try tighter matching or lower dense image size:

```powershell
python main.py --constrained-slot-window 1
```

放宽相邻点位约束：

```powershell
python main.py --constrained-point-window 2
```

降低 dense 计算量：

```powershell
python main.py --colmap-dense-max-image-size 1600
```

## 7. 推荐完整运行顺序

### Video frames are missing

最终查看：

```text
colmap_workspace/dense/final.ply
```

## 8. WiFi 定位怎么用

指纹库：

```text
reconstruction_dataset/wifi_fingerprint_library.csv
```

定位模型：

```text
reconstruction_dataset/wifi_localization_model.json
```

新 WiFi scan 可以保存成 JSON：

```json
[
  {"mac": "1C946885EEB1", "dbm": -50},
  {"mac": "1C946885EEB3", "dbm": -53}
]
```

运行：

```powershell
python localize_wifi.py --scan-json new_scan.json
```

CSV 也可以，列名包含 `mac,dbm` 或 `mac,rssi`：

```powershell
python localize_wifi.py --scan-csv new_scan.csv
```

## 9. 采集指南

下一轮采集请看：

```text
DATA_COLLECTION_GUIDE.md
```

最重要的原则：

- 每个 WiFi 点位拍 20 张，覆盖一圈方向
- 全程录一段连续路线视频，放进 `video/`
- 点位照片负责 WiFi 指纹，视频负责连接视觉结构
- 相邻点位之间也要有足够画面重叠
- 尽量形成闭环路线
