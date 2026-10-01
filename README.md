# 3D Reconstruction With WiFi, IMU and Vision

This project reconstructs indoor scenes from checkpoint photos and route videos,
using WiFi observations and IMU/PDR trajectories as supporting constraints.
COLMAP supplies the observed geometry. VideoScene can propose additions inside
locally bounded, multiview-supported gaps after point-cloud filtering.

## Source layout

| Location | Responsibility |
| --- | --- |
| `main.py` | Main preprocessing and reconstruction entry |
| `run_colmap.py` | COLMAP stages and recovery commands |
| Root helper modules | Capture inputs, geometry, provenance and native runners |
| `experiments/` | Generated local results; not source code |

Use `main.py` for the main pipeline and `run_colmap.py` for individual stages.
All reconstruction quality profiles default to COLMAP's automatic thread settings
and zero inter-stage cooldown. The launcher no longer asks about a low-peak mode.
Explicit thread/cooldown overrides remain available for manual configuration.
New defaults apply when workspace scripts are generated; an active reconstruction
continues with its original settings.
WiFi parsing and PDR/trajectory preparation now run in two owned processes, then
join before fingerprint/position fusion. Set `--preprocess-workers 1` for the
serial path. `processing_summary.json` records each branch's elapsed time and the
combined wall time. A failed branch or Ctrl-C stops both workers before the
workspace lock is released; worker diagnostics are retained in `processed/logs/`.
Preprocessing also locks its output and exported dataset directories. Concurrent
runs need distinct `--output`, `--reconstruction-output` and `--colmap-workspace`
directories, so separate workspaces cannot accidentally overwrite shared tables.

Image similarity prepares image features in parallel, with one ORB instance per
worker. Automatic mode uses up to eight workers on machines with at least 16
logical CPUs, and up to four on smaller machines. Pair verification and
RANSAC remain ordered. Use `--image-similarity-workers 1` for serial execution,
`0` for automatic selection, or `2` through `8` for an explicit worker count.
This preserves the feature settings and geometry acceptance thresholds.
Completed image tasks immediately free a slot for the next image, while final
results retain the input order. Thread counts are never changed from worker
threads. On cancellation, pending work is cancelled and already running native
image operations finish before their thread pool exits.
The dependent reconstruction stages still wait for complete inputs:
`sparse -> undistort -> PatchMatch -> fusion -> filtering`. These changes improve
CPU preprocessing; they do not promise a proportional speedup for a saturated
single-GPU PatchMatch pass or modify an already running workspace.
Historical experiment reports retain their original commands and hashes;
those commands may refer to optional tools that have been removed.

## Scope

- IMU/PDR provides a relative walking trajectory, continuous route-contour
  constraints where observations support continuity, and alignment references.
- Fresh WiFi RSSI fingerprints provide optional revisit/matching candidates.
  Candidates still require visual geometric verification; checkpoint order alone
  is not a wireless measurement.
- WiFi also estimates a fuzzy capture-support region and promotes nearby visual
  wall hypotheses. RSSI similarity connects local support regions; confidence and
  checkpoint camera scatter control their strength. These outputs use the visual
  reconstruction's relative frame and do not require a successful PDR alignment.
- Neither the walking contour nor the WiFi interpolation envelope is a room wall.
- VideoScene outputs are synthetic hypotheses, not recovered wireless or visual
  measurements. This project uses an observed COLMAP prior rather than the
  paper's NoPoSplat prior; it is not the complete paper implementation.

## Fuzzy WiFi extent and wall candidates

After a verified dense reconstruction, run:

```powershell
python run_colmap.py boundary --workspace colmap_workspace
```

The command prints a standalone `viewer.html` path under
`colmap_workspace/wifi_boundary/`; `latest.json` points to the newest view and
its `boundary.json` evidence. Newly generated sensor-assisted dense scripts run
this step automatically. Existing reconstructions need no MVS rerun.

The viewer shows a fuzzy wireless-support field, checkpoint positions, and
vertical visual plane candidates. Adjust the support contour and compare wall
scores with the WiFi slider at zero. For a saved alternative, use
`python wifi_boundary.py --workspace colmap_workspace --wifi-weight 0`.
Zero disables RF promotion of wall scores; it does not rebuild the upstream
reconstruction without WiFi or hide the separate radio-support field.
The estimator prefers the authenticated `supported.ply` observation layer;
if unavailable, it uses verified observed final geometry, never generated points.

Interior, uncalibrated RSSI samples do not directly locate outer walls. Low
support means unknown, not outside the room. Field width is a modeling choice
relative to checkpoint spacing; scores are not calibrated probabilities. WiFi
can favor visual planes near the sampled area but cannot create a wall without
visual evidence. The capture plane is estimated from checkpoint camera centers,
not certified by IMU gravity; no metre scale, closed floor plan, or accurate wall
semantics is claimed. Distorted visual poses produce distorted hypotheses.
The source cloud and camera poses stay unchanged.

## Requirements

The full runner currently targets **Windows with an NVIDIA CUDA GPU**. It uses
Windows COLMAP executables and generated `.bat` scripts, not a Linux/macOS runner.
Use Python 3.10-3.12; the current environment was checked with Python 3.10.
Install Git with Git LFS and a CUDA-enabled COLMAP distribution separately.
FFmpeg on PATH is optional; OpenCV also handles native video extraction.

From the project directory, create the main environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

An existing compatible Python environment can be used instead. Always install
`requirements.txt` with the same interpreter used to launch `main.py`.
VideoScene dependencies are installed separately under `VideoScene/.venv`,
without replacing packages in the main environment. COLMAP subprocesses launched
through `run_colmap.py` inherit the launching Python environment.
Activate the main environment before using the `python` commands below:

```powershell
.\.venv\Scripts\Activate.ps1
```

VideoScene is disabled by default; use `--scene-completion videoscene`
to opt in. Its code, model assets and
isolated environment must already be prepared under the ignored `VideoScene/`
directory to enable `--scene-completion videoscene`. The project does not include
an automatic installer or downloader.

The adapter expects the official `THU-SI/VideoScene` code revision
`de23c631fdb0663e5e732317c4642f38e4bd295f` and `hanyang/VideoScene` weight revision
`cf6075132cb3d7e1b42b6e7e382748b424387f19`. Place the weight files under
`VideoScene/checkpoints/checkpoints/`, preserving their relative paths in
`videoscene_runner.py` (`WEIGHT_FILES` and `CONFIG_FILES`). Both the distilled
checkpoint and full base transformer are required, together with the VAE,
prompt embedding and configs (approximately 27.8 GB total).

For an existing pinned checkout, prepare its separate environment with:

```powershell
py -3.10 -m venv VideoScene/.venv
.\VideoScene\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
.\VideoScene\.venv\Scripts\python.exe -m pip install -r requirements-videoscene.txt
python videoscene_completion.py --check
```

The completion check launches the isolated interpreter, imports dependencies
and the pinned official code, and runs a tiny CUDA/BF16 attention operation with
a 90-second timeout. Reconstruction entry points perform runtime preflight
before rebuild cleanup. Inference also verifies model file hashes. These checks
do not establish generated output quality. Sequential CPU offload, VAE tiling
and a CUDA allocator cap limit memory usage but do not guarantee sufficient
VRAM or runtime. BF16-capable CUDA is required.

## Inputs

```text
project/
  main.py
  requirements.txt
  pic/                            checkpoint photos, ordered by filename
  video/                          continuous route videos
  recording.sess/                 phone WiFi/IMU/movement logs
  colmap-x64-windows-cuda/
    bin/colmap.exe
```

The default is fuzzy ordered grouping with 20 photos per `CHECKPOINT_PAUSE` as
a soft reference, set by `--images-per-wifi-point`. It reads JPEG EXIF
`DateTimeOriginal` and relative pauses between photos; the camera clock does not
need to match the WiFi/PDR clock. Dynamic programming finds contiguous groups
within the count tolerance (default half the reference: 10--30 for a reference
of 20), balancing distinct pauses against count changes. Set
`--photo-group-tolerance 5` to limit groups to 15--25, or `--photo-grouping fixed`
to retain strict 20-photo grouping. These settings describe the capture protocol,
not any particular dataset.

Missing/unusable EXIF, reversed camera times, unsupported boundaries or nearly
tied alternatives retain fixed grouping with an explanation in
`image_wifi_summary.json`. No filesystem modification times or sensor timestamp
matching are used. A capture pause can also occur at the same location: inferred
boundaries remain estimates, with score margins, ignored pauses and count-limit
warnings available for review. Trailing photos remain unassigned. An explicit
`--photo-checkpoint-map photo_checkpoints.csv` overrides automatic inference.
The UTF-8 CSV columns are `session,point_index,image_name`: use the `.sess`
folder name, the 1-based checkpoint index within that session, and the exact
filename in `--pic-dir`. Each row assigns one photo; row order within a point
defines its view order. For example:

```csv
session,point_index,image_name
example.sess,1,DSC00001.JPG
example.sess,1,DSC00002.JPG
example.sess,2,DSC00005.JPG
```

For an explicit map, duplicate assignments, unknown
checkpoints and missing files fail validation; omitted photos remain unassigned,
and omitted checkpoints retain their sensor rows without photo anchors. The map
hash is recorded in `image_wifi_summary.json`. Export counts and default view
steps use each point's own group size. Circular view order remains a candidate
pairing/pose-seed heuristic, not a measured heading. When inference falls back to
fixed counts, a missing or extra photo can still shift later groups.
Readable, unique unassigned photos are now included as `visual_only_photo` inputs
by default. `unassigned_image_audit.csv` records exclusions for missing/corrupt
files, paths outside the capture directory, and exact duplicates. These photos
receive no checkpoint, PDR pose or WiFi fingerprint. Visual retrieval and geometric
verification determine usable overlap; passing the file audit does not establish
scene membership. Use `--exclude-unassigned-images` to exclude them. Extra-photo
camera folders separate image resolutions from checkpoint/video groups; sharing
intrinsics within a folder assumes a common camera and lens setting.
Missing fresh WiFi observations do not remove photos or valid PDR coordinates.
Post-reconstruction PDR reports also contain `photo_grouping`, which does not
require valid PDR coordinates. It compares robust camera-group centers, within-
group spread and distances to other groups in the same session. The report lists
photos much closer to another center, overlapping groups (including possible
revisits), and insufficient support. These are review flags; no assignment,
camera, or observation is changed. `priority_review_checkpoint_count` excludes
groups flagged only because another group's extent overlaps them. Thresholds
are relative to local spacing, so changing scene scale or coordinate axes does
not change their meaning. `--capture-mode off` disables this stationary check.
Checkpoint photos are grouped in capture order: finish acquisition at one point
before moving to the next. Photo timestamps need not match sensor timestamps.
WiFi aggregation uses all finite raw AP observations in `wifi_long.csv`, including
APs omitted by the global selection filter. Repeated AP/timestamp observations
count once. Each measurement belongs to at most one checkpoint; recorded stays
receive higher confidence than nearest-checkpoint assignments (bounded to 120 s).
Delayed logger observations and uncertain point associations receive continuous
age/time penalties. Per-AP weighted medians and RSSI dispersion produce a robust
fingerprint and confidence; missing APs are not synthesized. Points with at least
two observed APs can propose matches with their confidence carried downstream.

WiFi retrieval ranks candidates by shared-AP RSSI, reliability, overlap and point
confidence. Each point nominates up to four neighboring points, with a 1.5
priority multiplier. Each nominated stop pair has at most 64 image pairs by
default (the historical eight-by-eight budget), now allocated across all photos
at both stops. Already proposed edges are excluded before allocation; underused
photos receive priority, with deterministic tie breaking independent of assumed
camera yaw. The legacy `images_per_point` argument controls the pair budget, not
a fixed subset of participating images. Coverage counts are recorded per stop.
When visual verification has a pair budget, one quarter is reserved for WiFi
proposals and shared across stops and photos before RF priority breaks ties.
Every added pair still needs geometric
verification. These weights affect retrieval/verification and WiFi positioning;
they are not COLMAP bundle-adjustment weights. A point without RF evidence retains
its photos and PDR position but is not given an invented fingerprint.

After the first visual checks, recovery gives weak images and small disconnected
components a second, bounded search against the supported main component. It
excludes previously considered pairs before selecting the top candidates, and
uses available WiFi similarity as a ranking bonus. This additional search is
limited to `recovery_neighbors` pairs per target and uses the same geometric gate.
PDR alignment failures report the best candidate consensus and per-checkpoint
fit residuals for diagnosis; those values do not relax alignment acceptance.
Without measured video/sensor synchronization, video frames remain visual-only.

## Run

```powershell
python main.py
```

Confirm the full COLMAP run when prompted. A confirmed full rebuild intentionally
resets old reconstruction outputs. Back up a result you want to retain first.
Declining the prompt runs preprocessing/workspace preparation only when there
are no reconstruction outputs. An existing workspace is protected: declining
or running noninteractively exits before preprocessing can change its images,
matching pairs, configuration or reports. Use `python run_colmap.py resume` to
continue existing results, or choose a new `--colmap-workspace` for new inputs.
Direct preprocessing and workspace-preparation calls enforce the same guard.

For completed or interrupted reconstructions, confirming a full rebuild resets
old COLMAP outputs before preprocessing, after checking required VideoScene files
and the COLMAP executable. This restarts reconstruction, not a checkpoint resume.
Declining the full rebuild leaves the interrupted workspace untouched and exits
before preprocessing. The workspace lock rejects a second independent run before
prompts or cleanup; never remove a live lock file to bypass this protection.

Sparse reconstruction now checks PDR after visual mapping, using one vote per
checkpoint, robust fitting, held-out checkpoint validation and camera scatter.
`--pdr-capture-mode auto` (default) uses only photo groups explicitly tagged as
`ordered_checkpoint_same_location`. Use `stationary` to confirm that protocol
for older metadata; use `off` to disable this alignment stage for moving capture.
This switch controls PDR alignment, not photo association or WiFi retrieval.

Each stop now estimates a robust camera center with continuous Cauchy weights.
Every finite camera remains in the group. A stop's contribution is reduced by
its raw camera spread relative to the median spacing to up to three nearby
distinct stops; taking more photos does not multiply the stop's alignment weight.
The weighted fit is applied as one global similarity transform only after the
existing consensus, held-out and residual checks pass. It does not pull each
camera toward a stop, change bundle-adjustment weights, or turn RSSI into a
distance measurement. Diagnostic JSON retains per-camera and per-stop weights
even when alignment is withheld. No new sparse rebuild is needed just to inspect
these diagnostics; existing certified reconstructions are not automatically
realigned when the code changes.

Existing models can be checked without starting reconstruction or alignment:

```powershell
python pdr_alignment.py --inspect-only --model colmap_workspace/sparse/0 --manifest colmap_workspace/metadata/images_manifest.csv --report pdr_inspection.json --grouping-review-csv photo_grouping_review.csv
```

The optional grouping CSV preserves every checkpoint photo, including unregistered
ones. It shows original filenames, current session-local and global checkpoint
IDs, group flags, geometric proximity evidence, and available observation counts
and 8x8 image coverage. `nearer_point_index` is the session-local ID of the other
group; it is not an automatic correction. `not_registered` means no camera
position is available, and `no_flag` does not mean the assignment is verified.
Missing support metrics stay blank rather than becoming zero. The JSON report
records the CSV hash and source-model/manifest identities. Inspection does not
run `main.py` or modify the model or photo assignments.

For a dataset export's manifest, also pass `--exported-manifest` to translate
image names into the workspace namespace. The report must be outside the input
model. Local motion checks order stops by their recorded event times, retain
contour-break edges as unsupported, and compare displacement scale and unsigned
turning angles only within a continuous contour. Scale comparisons exclude the
edge being examined and require at least three other usable edges. These checks
expose disagreement; they do not change grouping, camera poses or alignment gates.
Only a reliable subset from one capture session can be used for alignment.
Rejected PDR keeps the raw visual coordinate frame. Alignment writes to a staging
directory and is selected only after the complete model and residuals pass
validation. The decision is saved in `pdr_alignment_decision.json`, bound into
sparse provenance, and included in evaluations. PDR and dense reconstruction
select the valid sparse component with the most registered images, breaking ties
by sparse point count and then folder name. An aligned model is used only when
authenticated to that component. Dense caches bind the selected model's path and
hashes; older caches are rebuilt, while existing PLY files remain until export.
Historical provenance still authenticates unchanged sparse files, so correcting
model selection does not require feature extraction or matching again. Existing
PDR decisions for a different component are not applied to the selected model.
Old sparse scripts without this gate are
refused before rebuild cleanup; `main.py` can regenerate them during a confirmed
full rebuild, or prepare a separate new workspace.

Newly launched subprocess trees are owned by a Windows Job Object: the child is
created suspended, attached, then resumed. Exceptions and timeouts terminate and
wait for descendants before returning to the workspace lock. Closing the owner
also terminates its job children (see [Windows Job Objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects)).
This does not retrofit protection onto an already running launcher.

The default workflow is:

1. Parse WiFi/IMU logs and construct supported PDR route contours.
2. Associate checkpoint photos and extract route-video keyframes.
3. Compare nearby native frames, retaining the sharper original or candidate.
   If all initial candidates remain below the blur threshold, search further
   within that sampling bin (at most 2 seconds or 45% of the interval).
4. Prepare visually verified matching candidates using route and WiFi constraints.
5. Run COLMAP sparse reconstruction. After undistortion, select explicit depth
   sources using at least 30 common tracks and a median triangulation angle of
   2--60 degrees; prefer sources from different capture stops. Dense execution
   omits unsupported depth references while preserving all sparse cameras and
   photos. `dense/depth_source_summary.json` records excluded views; the cache
   and fusion validate the same explicit subset. Fewer than three mutually
   supported views stops dense execution.
   These are conservative internal checks, not a guarantee of correct geometry.
6. Classify observed points using calibrated depth support and export `dense/final.ply`.
   Unknown points remain available; only repeated, diverse free-space conflicts
   without support are excluded. Density-only filtering is an explicit legacy option.
7. Optionally run VideoScene and save gated synthetic hypotheses separately.
   `main.py` defaults to `--scene-completion off`; its checkpoint model
   uses `--pdr-capture-mode stationary` for rotation-at-stop capture.
8. Write JSON quality and provenance reports.

Native frame selection does not synthesize or sharpen pixels. Its fallback adds
at most eight seeks per sample and never lowers the blur threshold. An entirely
blurred selection stays empty. Video reports include retained-bin fraction and
maximum temporal gap, including clip boundaries; these are sampling diagnostics,
not 3D surface coverage. Use `--no-video-refine-coverage-search` for the narrow
search, or `--video-refine-window-s 0` to disable refinement. The retired FlashVSR
replacement branch remains excluded. WiFi fingerprint model generation remains
in the main workflow.

Useful options:

```powershell
python main.py --scene-completion videoscene
python main.py --video-refine-window-s 0
python main.py --video-refine-budget-hours 2
python main.py --colmap-exe "C:\Tools\COLMAP\bin\colmap.exe"
python main.py --help
```

VideoScene defaults to up to three distinct 49-frame 720x480 proposals, processed
sequentially within a four-hour completion budget, not an end-to-end runtime
guarantee. Selection prioritizes novel supported gap coverage along compatible
route segments. Missing pixels need coherent observed depth brackets and
actual shared-track parallax between endpoints at different checkpoints, plus
agreement across views, including both calibrated endpoints. A narrow generated
texture context assists local MVS, but only validated hole points can be added;
observed geometry is never replaced. Reports include projected target coverage,
not just a successful inference flag. Large unsupported regions and reflective
content are not automatically filled. Failure or no accepted proposal does not
authorize fabricated replacement geometry. Observed backups and accepted
synthetic additions have separate provenance under `dense/videoscene/<run>/`.

`--skip-wifi-recovery` disables both explicit RF pair retrieval and RF scoring
in targeted recovery. Both arms still prioritize cross-checkpoint recovery and
require visual verification; this switch does not disable the IMU/PDR route.

The root `run_colmap_dense.bat` uses `resume` and follows the configuration
originally recorded in the workspace, preserving existing reconstruction inputs.
It no longer forces VideoScene on. Previously prepared workspaces retain their
recorded settings. Use `python run_colmap.py resume --with-videoscene` to opt in
for a workspace that recorded completion as off.
Completed compatible hypothesis runs are reused rather than inferred again.

Quality reports mark retained unknown observation points as `dense_needs_review`;
successful execution does not certify geometry. The WiFi boundary output includes
`wall_diagnostics` with plane rejection counts. If observed geometry is insufficient
for walls, the fuzzy capture extent is still exported, without inventing walls.

Completion conditions supported holes with RGB reprojected from both calibrated
endpoint photos. The depth estimates used for that warp are hypotheses, not new
measurements; original rendered colors remain the MVS context outside the hole
and its narrow blend boundary. Accepted points must project into at least one
target hole, agree with at least three views (observed depth or supported hole
prior), have support at both endpoints, and avoid observed free-space conflicts.
Requiring holes in all three views would discard points whose other views already
have useful observed depth. `conditioning.json` and `geometry_validation.json`
record conditioning coverage and the separate validation losses.

`dense/videoscene/<run>/observed.ply` is the **before-completion backup**;
`generated_additions.ply` and `completed.ply` are separate hypothesis/preview
layers. The observed `dense/final.ply` stays unchanged by default. Explicit
`python videoscene_completion.py --merge-generated` enables the previous merge
behavior and records generated provenance. To compare with the original
unconditioned rendering, use `python videoscene_completion.py --prior-conditioning observed`.
The default is `--prior-conditioning endpoints`. Increasing `--max-clips` only
helps when more distinct, supported gaps exist; it does not recover surfaces
with no useful observations.

## Outputs and Recovery

COLMAP stages now use compact console output by default. Interactive terminals
update the PatchMatch progress on one line; redirected output prints progress
at most every 15 seconds, plus pass and stage changes. Warnings and errors remain
visible. `ETA/pass` estimates only the current photometric/geometric pass from
completed views; it does not include later passes, fusion or completion.
Complete subprocess output is saved under `colmap_workspace/logs/` (or the selected
workspace's `logs/`), with the log path printed at startup and exit.
Use `python main.py --colmap-log-level verbose` or
`python run_colmap.py dense --log-level verbose` for native console output.
Output changes take effect on the next launch; an already running reconstruction
keeps its current console. Reconstruction resolution and iteration counts are
unchanged. Dense output validation reads its shared PatchMatch settings once per
validation instead of rereading the JSON for every view.

- `processed/`: WiFi/PDR tables, photo associations, video frames and metadata.
- `reconstruction_dataset/`: COLMAP-ready images, route seeds and WiFi models.
- `colmap_workspace/`: matching inputs, generated runners, sparse/dense models.
- `colmap_workspace/dense/final.ply`: filtered observed cloud; synthetic additions
  enter this file only when `--merge-generated` is explicitly supplied.
- `final_output_summary.json`, `videoscene_summary.json` and `quality_report.json`:
  identities, provenance and quality status within the workspace.

Raw `fused.ply` is retained. Binary PLY files are intended for tools such as
MeshLab, not text editors. Meshing is off by default; `--colmap-meshing` enables
inspection meshes, but does not make their interpolated surfaces observations.

Final export stages both the PLY and its provenance, keeps rollback copies until
both commits succeed, and only then removes optional intermediates. If rollback
itself fails, recovery copies are retained and their paths are reported. Cleanup
receipt failures leave the committed identity valid and mark cleanup as pending.

Stage-specific entry points are retained because they support recovery without
rerunning everything:

```powershell
python run_colmap.py full
python run_colmap.py resume --dry-run
python run_colmap.py resume
python run_colmap.py dense
python run_colmap.py filter
python run_colmap.py completion --max-clips 3
python run_colmap.py report
python reconstruction_quality.py --workspace colmap_workspace --check-freshness
```

Use these only with current generated scripts and matching inputs. Dense caches
are reused only when provenance, configuration and recorded output hashes match.
Certification checks the undistorted sparse model, registered images, PatchMatch
view coverage, and each geometric depth/normal map's dimensions, byte count and
finite payload. Old cache receipts without these output checks are recomputed.
Completion requires a
filtered final with its hash and reconstruction binding. Old exports without
that binding require a new filtered export from corresponding verified dense
data; do not fabricate a receipt for an old cloud. `--prepare-only` allows preview
preparation without merging. Do not delete an active `.pipeline.lock` file.

`run_colmap_dense.bat` and the legacy workspace `run_colmap_dense_high.bat` now
delegate to the locked `resume` entry. The legacy name no longer invokes removed
floor/dark-surface scripts or overrides the configured reconstruction profile.
Newly generated workspaces include this compatibility entry too.

Resume keeps a verified final and runs only missing completion/report stages.
After a completed sparse stage, an interrupted MVS stage can continue through
the canonical dense runner if no existing PLY would be removed. Unverified or
interrupted sparse results require an explicit rebuild. Existing PLY files with
mismatched provenance are preserved and cause a refusal. Filtering verifies MVS
hashes, uses a proven fused source or re-fuses into an isolated staging directory,
and retains previous finals before publishing a changed result. `--dry-run`
checks identities and prints the planned stages without running reconstruction.

New dense runners checkpoint undistorted inputs before PatchMatch. With matching
inputs, an interrupted run retains validated per-view depth/normal pairs and
recomputes missing or damaged pairs. A damaged photometric dependency invalidates
the geometric pass; other valid photometric views remain reusable. Completed
receipts skip both undistortion and PatchMatch, including repeated interruptions
before fusion. Older partial caches without an input checkpoint are recomputed
once because their inputs cannot be authenticated.

Quality reports bind their input images, model, configuration and PLY files by
SHA-256, and are written atomically. `--check-freshness` exits with code 2 for an
old or changed report. Completion and the stage runner refresh reports;
`python run_colmap.py report` refreshes them manually. Reports stay in their
reconstruction workspace by default; no compatibility copy is written to the
project root. Experimental workspaces use their own dataset metadata and reports.

The selected sparse model's quality report now includes per-camera linked
observation counts, out-of-image coordinates, 8x8 feature occupancy, bounding-box
spread and a spatial eigenvalue ratio. Unlinked features do not count as 3D
support. Views with features in at most two grid cells are listed for inspection;
this is not an automatic rejection rule or a surface-area estimate.
`reconstruction_evaluation.py compare` also lists per-camera observation and
grid-coverage losses relative to the first report, so aggregate point gains do
not hide local losses. Binary and text models use the same calculations; absent
metrics in older reports are reported as unavailable.

The default compact console shows stage progress, warnings and a short final
summary. Full preprocessing statistics are saved to `processed/processing_summary.json`;
COLMAP output and stage timings remain under the workspace's `logs/` directory.
Repeated identical warnings are counted, routine metadata stays in the log, and
per-stage timing tables are omitted from the compact console. Use
`python main.py --colmap-log-level verbose` (or `run_colmap.py --log-level verbose`)
to display full details.

Subprocess monitoring records the active stage, elapsed time and time since the
last complete output line every 60 seconds. The console heartbeat appears when
there has been no visible update for that interval, including during silent bundle
adjustment. On Windows, a batch shell left without any child process for 30
seconds fails with a diagnostic instead of waiting indefinitely. An exited
launcher whose output pipe stays open gets a 5-second drain period before its
remaining descendants are terminated. Logs are flushed on timer ticks and timing
reports include the exit code and any monitoring failure.

Silence alone does not establish a hang: long computations remain allowed by
default. Set `COLMAP_OUTPUT_TIMEOUT_SECONDS` to a positive number to opt into a
no-output deadline (`0` disables it). Ctrl-C remains responsive while waiting for
output. Generated batch runners recognize all nonzero exit codes, including
negative Windows crash codes.

Sparse mapping saves recovery candidates every 50 registered frames under
`sparse_snapshots/<run-id>/`, separately from completed sparse models. These
snapshots are never automatically certified or selected for dense reconstruction;
they require validation before manual recovery. This does not change bundle
adjustment parameters or guarantee recovery from a numerical solver failure.

## Moving to Another Machine

1. Transfer source code and the required `pic/`, `video/` and `.sess` inputs.
   Raw data, binaries, generated outputs and weights are intentionally not in Git.
2. Recreate the main Python environment; do not copy virtual environments between
   machines. If using VideoScene, also prepare its pinned checkout and isolated
   environment as described above. Verified weights can be copied into the same
   relative checkpoint layout.
3. Install CUDA-enabled COLMAP in the relative directory above, provide
   `--colmap-exe`, or make `colmap` available on PATH.
4. Run `main.py` in the new location to regenerate workspace paths and scripts.
   Existing generated scripts/metadata can contain old absolute paths; do not
   reuse them unchanged after moving. Confirming a full run resets old outputs.

This project introduction documents the workflow. Runtime reports use CSV/JSON;
controlled comparisons also provide a small Markdown table. Documentation and tests bundled inside ignored third-party checkouts
are not modified by project source cleanup.
