# FYP Publication Summary

## Project Title

**Virtual Campus Tour System: Multimodal Indoor 3D Reconstruction with WiFi/IMU-Aided Positioning**

Authors: Yuxiao Long, Yuxuan Liu  
Degree: Bachelor of Science (Honours) in Computer Science and Technology  
Institution: Beijing Normal-Hong Kong Baptist University  
Date: May 2026

## Executive Summary

This Final Year Project develops a practical pipeline for building an indoor virtual campus tour from smartphone-collected visual, wireless, and inertial data. The system combines route videos, checkpoint photographs, WiFi scans, and IMU measurements to support indoor 3D reconstruction and signal-based localization. Instead of treating 3D reconstruction as a purely visual task, the project uses WiFi and pedestrian dead reckoning (PDR) derived from IMU data as supporting spatial constraints. These constraints help organize image data, generate localization metadata, reduce incorrect visual matching in repetitive indoor environments, and provide a reusable WiFi fingerprint library for later positioning.

The final system is designed around an end-to-end workflow. A user collects data with a phone by walking through the indoor environment, recording WiFi/IMU data, taking checkpoint photos, and filming a continuous route video. The project scripts then parse the raw session files, estimate a relative walking trajectory from IMU acceleration and heading readings, bind checkpoint images to WiFi/PDR points, extract video frames, export a reconstruction dataset, prepare a COLMAP workspace, and generate scripts for sparse and dense 3D reconstruction. When COLMAP dense reconstruction is run successfully, the dense point cloud can be opened in tools such as MeshLab or CloudCompare, while the generated WiFi localization model can estimate positions from new WiFi scans.

The current P0-quality preprocessing run uses the final reconstruction profile, enables PDR-assisted matching and PDR alignment references, includes route-video bridge frames, and records data-quality caveats explicitly. The generated workspace contains 277 COLMAP input images, 2,469 constrained match pairs, and 260 PDR alignment reference images. Existing COLMAP database and sparse outputs from an older run are detected and marked for reset before the next sparse/dense reconstruction run.

## Abstract

Indoor virtual tour systems require reliable spatial reconstruction, but indoor environments often create challenges for image-based reconstruction due to repeated textures, weak visual features, reflective surfaces, and limited GPS availability. This project proposes a multimodal pipeline for constructing an indoor virtual campus tour using smartphone videos, checkpoint photographs, WiFi scans, and IMU data. The system processes WiFi scans to generate fingerprint features and uses IMU linear acceleration, rotation vectors, and movement events to estimate a relative PDR trajectory. Checkpoint photographs are strongly associated with WiFi/PDR points, while route video frames are extracted as weakly constrained visual bridge frames for COLMAP reconstruction. The pipeline exports image manifests, camera pose seeds, WiFi fingerprint libraries, signal boundaries, constrained COLMAP matching pairs, and PDR alignment references. A KNN-based WiFi localization model is also produced for estimating positions from new WiFi scans. By combining visual reconstruction with wireless signal features and inertial motion metadata, the project improves the organization of indoor image data and reduces the risk of incorrect matching in repetitive spaces. The final system provides a reproducible workflow from raw mobile data collection to a PDR-enabled COLMAP workspace, dense reconstruction support, and indoor localization assets, forming a technical foundation for a virtual campus tour application.

## Problem Statement

A virtual campus tour needs more than static images. It requires a spatially coherent representation of indoor spaces so that users can understand where they are, how areas connect, and how to navigate between locations. Outdoor mapping can rely on GPS, but indoor environments do not provide stable GPS signals. Purely visual 3D reconstruction also faces several problems:

- Indoor corridors, classrooms, and walls often contain repeated or texture-poor surfaces.
- Photos taken at isolated points may not have enough overlap for stable reconstruction.
- COLMAP may incorrectly match visually similar but spatially distant areas.
- Mobile collection data includes useful WiFi and IMU signals, but these signals are not directly consumed by standard visual SfM tools.

This project addresses the gap by using WiFi and IMU/PDR data as auxiliary spatial information around the visual reconstruction workflow. The key idea is that WiFi and IMU do not directly create a point cloud; instead, they generate position seeds, matching constraints, signal boundaries, and localization models that make the visual reconstruction pipeline more structured and reusable.

## Objectives

The project aims to achieve the following objectives:

1. Build a complete preprocessing pipeline for smartphone-collected WiFi, IMU, checkpoint photo, and route video data.
2. Generate a WiFi fingerprint library from checkpoint-based WiFi scans.
3. Estimate a relative indoor walking path using IMU-based PDR.
4. Bind checkpoint photos to WiFi/PDR positions and extract route video frames as visual bridge frames.
5. Export a COLMAP-compatible reconstruction dataset with image manifests and camera pose seeds.
6. Generate constrained matching pairs to reduce incorrect COLMAP matching in indoor repetitive scenes.
7. Produce sparse and dense point cloud reconstruction outputs through COLMAP.
8. Provide a WiFi localization model that can estimate position from new WiFi scans.
9. Document a repeatable data collection and reconstruction workflow for future campus mapping.

## System Overview

The project is organized as an end-to-end pipeline:

```text
Raw phone data
  -> WiFi / IMU / movement event parsing
  -> IMU step detection and heading estimation
  -> PDR trajectory estimation
  -> WiFi-PDR feature fusion
  -> checkpoint image to WiFi point binding
  -> route video frame extraction
  -> reconstruction dataset export
  -> WiFi localization and signal boundary generation
  -> COLMAP workspace preparation
  -> sparse reconstruction
  -> dense reconstruction
  -> point cloud and virtual tour assets
```

The system treats the collected data as two complementary streams:

- **Visual stream:** checkpoint photos and extracted route video frames are used by COLMAP to reconstruct the indoor geometry.
- **Signal/motion stream:** WiFi scans provide fingerprint features, while IMU acceleration, rotation vectors, and movement events provide step detection, heading estimation, rough positions, route order, and matching constraints.

## Data Collection Design

The implemented data collection and preprocessing protocol follows the updated project guideline:

- Each WiFi checkpoint contains a short stop of around 3 to 5 seconds.
- Each checkpoint is marked using `CHECKPOINT_PAUSE`.
- Each WiFi checkpoint has 20 photos by default, covering approximately 360 degrees.
- Route video is recorded continuously while moving between checkpoints.
- IMU recording remains active during the whole route so that acceleration, orientation, and movement events share the same time base as WiFi scans.
- Adjacent checkpoints are designed to have sufficient visual overlap.
- The walking path should form a loop where possible to reduce drift and improve reconstruction continuity.

This design separates the roles of photos and videos:

- Checkpoint photos act as strong WiFi-labeled visual anchors.
- Video frames act as weak visual bridge frames between nearby checkpoints.
- IMU data acts as the motion backbone that connects checkpoints into a relative walking path.

## Methodology

### 1. WiFi Feature Processing

The pipeline reads raw WiFi scans from `wifiScanner.dat`, normalizes AP identifiers, filters useful access points, and exports RSSI feature tables. The generated WiFi fingerprint table is used for both localization and reconstruction metadata.

Main outputs include:

```text
processed/wifi_long.csv
processed/ap_stats.csv
processed/selected_aps.json
processed/wifi_wide_selected.csv
```

### 2. IMU Sensor Processing

The system parses IMU and movement files from the phone session. The IMU component is not optional background data; it provides the motion evidence used to build the relative indoor path. In the current implementation, the pipeline uses linear acceleration for step detection, rotation vector or game rotation vector readings for heading estimation, and movement events for segmenting the walking route.

Main IMU-related inputs include:

```text
*.sess/7.dat              linear acceleration samples
*.sess/8.dat              rotation vector samples
*.sess/11.dat             game rotation vector samples
*.sess/movements.dat      START / RESTART / CHECKPOINT_PAUSE / STOP events
```

The processing logic includes:

- Computing acceleration magnitude from three-axis IMU samples.
- Smoothing acceleration values with a moving average.
- Detecting step peaks using an adaptive threshold based on local acceleration statistics.
- Converting quaternion-based rotation readings into yaw/heading angles.
- Preferring game rotation vector heading when available, with rotation vector data as fallback.
- Splitting the route into movement segments using recorded movement events.

Main outputs include:

```text
processed/movement_events.csv
processed/movement_segments.csv
```

### 3. PDR Trajectory Generation

After IMU step detection and heading estimation, the system integrates steps into a relative PDR trajectory. Each detected step receives an estimated step length, heading angle, and accumulated relative position. The resulting coordinate system is not an absolute room map by itself; it is a relative motion trace that can later be aligned or used as a geometric prior.

Main outputs include:

```text
processed/pdr_steps.csv
processed/pdr_path.csv
processed/pdr_summary.json
processed/alignment_anchor_template.csv
```

The PDR trajectory contributes to the project in four ways:

- It provides rough `pdr_x` and `pdr_y` coordinates for WiFi scans.
- It gives checkpoint photos rough position seeds.
- It supports distance-based filtering when generating COLMAP matching pairs.
- It provides a possible alignment reference for sparse reconstruction outputs.

### 4. WiFi and PDR Fusion

Each WiFi scan is matched to the nearest PDR position, producing a table that connects signal features with relative spatial positions.

Main output:

```text
processed/wifi_pdr_features.csv
```

This table becomes the foundation for image binding, WiFi fingerprint export, boundary generation, and rough pose seeding.

### 5. Checkpoint Image Binding

Photos in `pic/` are sorted by filename and grouped by checkpoint. By default, every 20 photos are assigned to one WiFi checkpoint. These images receive strong WiFi labels and can be used as anchor images.

Main outputs include:

```text
processed/image_wifi_points.csv
processed/image_wifi_long.csv
processed/image_wifi_summary.json
```

### 6. Route Video Frame Extraction

The route video is placed in `video/` and sampled into frames. These frames are not used as WiFi fingerprint samples. Instead, they act as visual bridge frames to help COLMAP connect adjacent checkpoint areas.

Main outputs include:

```text
processed/video_frames/
processed/video_frames_manifest.csv
processed/video_frames_summary.json
```

### 7. Reconstruction Dataset Export

The pipeline copies and renames checkpoint photos and video frames into a COLMAP-friendly dataset. It also writes image manifests, rough camera pose seeds, WiFi fingerprint data, and boundary seeds.

Main outputs include:

```text
reconstruction_dataset/images/
reconstruction_dataset/images_manifest.csv
reconstruction_dataset/camera_pose_seed.csv
reconstruction_dataset/wifi_fingerprint_library.csv
reconstruction_dataset/wifi_boundary_seed.csv
```

### 8. WiFi Localization Model

A KNN-based WiFi fingerprint localization model is generated from the fingerprint library. The system also supports leave-one-out evaluation and can estimate the position of a new WiFi scan provided as JSON or CSV.

Main outputs include:

```text
reconstruction_dataset/wifi_localization_model.json
reconstruction_dataset/wifi_localization_eval.csv
reconstruction_dataset/wifi_positioning_summary.json
reconstruction_dataset/wifi_signal_grid.csv
reconstruction_dataset/wifi_signal_boundary.csv
```

### 9. COLMAP Workspace Preparation

The project prepares a dedicated COLMAP workspace containing image lists, constrained matching pairs, PDR alignment references, metadata, and executable batch scripts.

Main outputs include:

```text
colmap_workspace/images/
colmap_workspace/image_list.txt
colmap_workspace/match_pairs.txt
colmap_workspace/pdr_alignment_refs.txt
colmap_workspace/metadata/
```

### 10. Sparse and Dense Reconstruction

After preprocessing, COLMAP can be run through the generated scripts. Sparse reconstruction estimates camera poses and sparse 3D points. Dense reconstruction generates a denser point cloud.

Expected final output:

```text
colmap_workspace/dense/fused.ply
```

## Implementation Summary

The project codebase is modular:

| File | Role |
| --- | --- |
| `main.py` | Convenient launcher for the full preprocessing pipeline |
| `preprocess_pipeline.py` | Main orchestration logic for WiFi parsing, IMU step/heading processing, PDR generation, images, video, export, and COLMAP workspace preparation |
| `image_mapping.py` | Binds checkpoint photos to WiFi/PDR points |
| `video_frames.py` | Extracts and filters route video frames |
| `reconstruction_export.py` | Exports COLMAP-friendly reconstruction datasets and metadata |
| `wifi_positioning.py` | Builds WiFi localization model, signal grid, and boundary data |
| `localize_wifi.py` | Estimates position from a new WiFi scan |
| `colmap_bridge.py` | Prepares COLMAP workspace, constrained match pairs, scripts, and metadata |
| `run_colmap.py` | Runs generated COLMAP batch scripts from the project root |

The default run command is:

```powershell
python main.py
```

The recommended COLMAP execution sequence is:

```powershell
python run_colmap.py reset
python run_colmap.py sparse
python run_colmap.py dense
```

## Key Technical Contributions

1. **Multimodal indoor reconstruction workflow**  
   The project integrates visual data, WiFi scans, and IMU/PDR data into a single repeatable workflow.

2. **Dedicated IMU/PDR motion estimation module**  
   Linear acceleration, rotation vectors, and movement events are processed into step detections, headings, movement segments, and a relative route trajectory.

3. **Checkpoint-based image and WiFi binding**  
   Each group of checkpoint images is strongly associated with a WiFi fingerprint point, giving each anchor image useful signal metadata.

4. **Video-based visual bridging**  
   Route video frames are used to improve visual continuity between checkpoint photo groups without polluting the WiFi fingerprint library.

5. **PDR-assisted spatial metadata**  
   IMU-based PDR provides rough relative positions and heading seeds for image metadata and reconstruction alignment.

6. **Constrained COLMAP matching**  
   The system generates `match_pairs.txt` based on route order, checkpoint adjacency, image slot direction, PDR distance, and video frame continuity. This reduces the chance of matching visually similar but spatially unrelated indoor images.

7. **WiFi fingerprint localization support**  
   The project exports a reusable WiFi localization model and evaluation files, allowing future localization from new WiFi scans.

8. **Reproducible project structure**  
   The pipeline produces standardized folders such as `processed/`, `reconstruction_dataset/`, and `colmap_workspace/`, making experiments easier to repeat and debug.

## Current Run Results

The current P0 run was regenerated with `python main.py` after changing the default launcher to use the `final` reconstruction profile and PDR-enabled COLMAP preparation. It should be treated as the current preprocessing/export result, not as a completed dense reconstruction result.

Current deliverables:

- A working preprocessing pipeline from raw phone session data to COLMAP-ready images and metadata.
- Processed IMU/PDR tables with movement events, detected steps, heading estimates, and relative path coordinates.
- Processed WiFi/PDR features with selected AP RSSI features and relative PDR coordinates.
- Checkpoint image mapping tables for 13 WiFi checkpoints, using 20 anchor photos per checkpoint.
- An explicit unassigned image list at `processed/unassigned_images.csv` for the 190 raw photos outside the checkpoint grouping protocol.
- Route video frame extraction with 17 retained video bridge frames from `video/IMG_1735.MOV`.
- A reconstruction dataset containing 277 exported images, pose seeds, WiFi fingerprints, signal boundaries, and video metadata.
- A PDR-enabled COLMAP workspace with constrained matching pairs and PDR alignment references.
- A WiFi localization model, leave-one-out evaluation, signal grid, and signal boundary outputs.
- COLMAP sparse/dense scripts ready to run after resetting older COLMAP outputs in the workspace.

Current result table:

| Metric | Current value |
| --- | --- |
| Collection sessions | 1 |
| Number of WiFi checkpoints | 13 |
| Photos per checkpoint | 20 |
| Raw photos in `pic/` | 450 |
| Mapped checkpoint photos | 260 |
| Extra unassigned photos | 190, listed in `processed/unassigned_images.csv` |
| Number of route videos | 1 |
| Video frames extracted | 240 |
| Video frames retained/exported | 17 |
| Detected PDR steps | 695 |
| Estimated PDR walking distance | 504.59 m |
| Movement events | 28 |
| Movement segments | 14 |
| WiFi/PDR feature rows | 1,652 |
| Selected WiFi APs | 15 |
| Reconstruction dataset images | 277 |
| Constrained COLMAP match pairs | 2,469 |
| PDR alignment reference images | 260 |
| COLMAP workspace status | `ready_needs_colmap_reset` |
| Sparse reconstruction registered images | Pending rerun after COLMAP reset |
| Sparse 3D points | Pending rerun after COLMAP reset |
| Dense point cloud file | Pending; expected path `colmap_workspace/dense/fused.ply` |
| WiFi localization mean error | 12.38 m |
| WiFi localization median error | 9.00 m |
| WiFi localization max error | 28.36 m |

P0 quality notes:

- The current launcher no longer defaults to visual-only mode. PDR-assisted distance matching and PDR alignment references are enabled by default.
- The 190 extra photos are not silently used as WiFi-labeled checkpoint anchors because they do not fit the fixed 20 photos/checkpoint protocol for the 13 detected checkpoints.
- The route video contributed bridge frames, but 223 of 240 extracted frames were rejected by the blur threshold. This is useful evidence for the report and a strong reason to recapture or tune video-frame filtering for a better visual reconstruction.
- Existing `database.db` and `sparse/` outputs are from an older COLMAP run. Before reporting sparse/dense metrics for this PDR-enabled workspace, run `python run_colmap.py reset`, then `python run_colmap.py sparse`, then `python run_colmap.py dense`.

## P1 Quality Addendum

After the P0 cleanup, the next quality layer is to make the project measurable and defensible rather than only runnable. P1 focuses on output freshness, localization interpretation, reconstruction diagnostics, ablation experiments, and reference-to-claim alignment.

### Output Freshness and Run Integrity

The preprocessing and export outputs are now regenerated from the current default command, while older COLMAP outputs are explicitly detected. This prevents the report from mixing a new `match_pairs.txt` or `pdr_alignment_refs.txt` with an old `database.db` or sparse model.

| Output group | Current status | Report treatment |
| --- | --- | --- |
| `processed/` | Regenerated with final profile and PDR-enabled defaults | Use as current preprocessing result |
| `processed/unassigned_images.csv` | Present, 190 extra raw photos listed | Explain as excluded from the fixed checkpoint-photo protocol |
| `processed/video_frames_manifest.csv` | Present, 17 retained bridge frames | Use as current video contribution, with blur rejection caveat |
| `reconstruction_dataset/` | Regenerated with 277 images | Use as current COLMAP input dataset |
| `wifi_localization_model.json` and evaluation files | Regenerated and copied to metadata | Use as lightweight WiFi baseline, not navigation-grade accuracy |
| `colmap_workspace/database.db` and `sparse/` | Existing older outputs detected | Do not report as final PDR-enabled reconstruction until reset and rerun |
| `colmap_workspace/dense/fused.ply` | Pending | Report as expected output path only |

### WiFi Localization Interpretation

The current WiFi positioning result uses 13 fingerprint points and 15 selected AP features. Leave-one-out evaluation gives a mean error of 12.38 m, median error of 9.00 m, and maximum error of 28.36 m. This should be presented as a baseline demonstrating that the pipeline can generate reusable WiFi localization assets, not as a finished high-accuracy indoor navigation system.

For a stronger P1 result, the report should explicitly state the causes and improvements:

- The fingerprint count is small for a long indoor route, so KNN has limited spatial resolution.
- Single-checkpoint fingerprints are vulnerable to RSSI noise, body occlusion, and device orientation.
- More repeated scans per checkpoint, denser checkpoints, AP stability filtering, and a graph-based model such as IndoorGNN are natural next steps.
- WiFi results should be evaluated separately from 3D reconstruction quality because RSSI localization and visual geometry fail for different reasons.

### Reconstruction Quality Report

The next COLMAP run should produce a compact quality report before claiming final sparse or dense reconstruction. The report should include the following metrics.

| Metric | Source | Current status |
| --- | --- | --- |
| Registered image count | COLMAP sparse model | Pending rerun after reset |
| Registered image ratio | Registered images / 277 input images | Pending rerun after reset |
| Sparse 3D point count | COLMAP `points3D` output | Pending rerun after reset |
| Sparse component count | `colmap_workspace/sparse/` model folders | Pending rerun after reset |
| Largest component size | COLMAP model analysis | Pending rerun after reset |
| PDR alignment reference count | `pdr_alignment_refs.txt` | 260 images available |
| PDR alignment residual | COLMAP `model_aligner` output/log | Pending rerun after reset |
| Dense point cloud exists | `colmap_workspace/dense/fused.ply` | Pending |
| Dense point count | PLY header or point-cloud tool | Pending |
| Match-pair count | `match_pairs.txt` / summary | 2,469 pairs available |
| Match-pair reason distribution | `match_graph_edges.csv` | Available for analysis |
| Video bridge coverage | `video_frames_manifest.csv` | 17 retained bridge frames |

### Ablation Experiments

P1 should include at least a small ablation table. The aim is not to exhaustively benchmark every setting, but to show that the design choices are testable.

| Experiment | Command/config idea | Purpose | Status |
| --- | --- | --- | --- |
| Visual-only baseline | `python main.py --visual-only` | Compare against PDR-enabled matching and alignment refs | Planned |
| PDR-enabled current run | `python main.py` | Main pipeline after P0 cleanup | Prepared |
| Without video bridge frames | `python main.py --skip-video-frames` | Measure whether route frames improve connectivity | Planned |
| Video blur threshold sweep | `--video-blur-threshold 0/40/60` | Check whether current filtering is too strict | Planned |
| Constrained vs exhaustive/sequential matching | `--colmap-matcher constrained/exhaustive/sequential` | Evaluate robustness vs compute cost | Planned |
| PDR distance window sweep | `--constrained-max-pdr-distance-m 4/8/12` | Check sensitivity to PDR drift and over-restriction | Planned |
| WiFi K sweep | `--wifi-localization-k 1/3/5` | Determine whether KNN weighting is stable | Planned |

The thesis can report these as a compact comparison table with columns for registered images, sparse points, dense file status, visual continuity, runtime, and WiFi localization error where applicable.

### Reference-to-Claim Alignment

The reference list should be used to support specific claims rather than appear as a generic bibliography.

| Project claim | Supporting references |
| --- | --- |
| COLMAP/SfM is the visual reconstruction backbone | [1], [2] |
| IMU/PDR is a weak spatial prior rather than a direct point-cloud generator | [3], [4], [5], [6] |
| WiFi RSSI fingerprinting is useful but noisy and data-dependent | [7], [8], [9] |
| Indoor reconstruction needs careful evaluation against indoor-scene benchmarks | [10] |
| Dense/surface reconstruction can later be improved with 3DGS or depth-prior methods | [11], [12], [13], [14], [15] |

This P1 framing keeps the final report honest: the current contribution is a reproducible multimodal pipeline and diagnostic COLMAP workspace, while the remaining quality gap is expressed through concrete metrics and experiments.
## Evaluation Plan

The project can be evaluated from four perspectives:

### Reconstruction Quality

- Whether COLMAP successfully registers the main checkpoint images and route frames.
- Whether the dense point cloud preserves the major indoor layout.
- Whether adjacent areas connect continuously rather than forming disconnected fragments.
- Whether constrained matching reduces incorrect matches in repeated indoor scenes.

### IMU/PDR Quality

- Whether step detection follows the actual walking rhythm without severe over-counting or under-counting.
- Whether heading estimation produces a route shape consistent with the real walking direction.
- Whether checkpoint events align with the expected positions along the PDR trajectory.
- Whether PDR distance constraints improve matching without removing necessary visual pairs.

### Localization Quality

- Leave-one-out WiFi fingerprint localization error.
- Stability of selected APs across checkpoints.
- Whether estimated WiFi positions remain inside or near the generated signal boundary.

### System Usability

- Whether a new data collection can be processed using `python main.py`.
- Whether the output folder structure is understandable.
- Whether the generated COLMAP scripts can be executed without manual reconstruction setup.
- Whether the documentation is clear enough for future data collection.

## Limitations

The project still has several limitations:

- IMU-based PDR coordinates are relative and may drift over long walking paths.
- Step detection can be affected by phone placement, walking style, pauses, and hand motion.
- Heading estimation may accumulate error when the phone orientation changes sharply or sensor readings become unstable.
- WiFi RSSI values are noisy and can change due to human movement, device orientation, and environmental changes.
- COLMAP reconstruction quality still depends heavily on image overlap, lighting, texture richness, and camera motion.
- The current WiFi localization model is lightweight and may not match the accuracy of more advanced probabilistic or learning-based methods.
- The system currently focuses on point cloud generation and metadata preparation; a complete interactive virtual tour frontend can be further developed.

## Future Work

Future improvements may include:

- Adding loop closure or map alignment to reduce PDR drift.
- Improving WiFi localization with probabilistic filtering or learned embeddings.
- Adding visual-inertial SLAM support for stronger pose estimation.
- Building an interactive web-based virtual campus tour viewer.
- Adding semantic labels for rooms, corridors, landmarks, and navigation points.
- Supporting multi-floor mapping and floor-level WiFi classification.
- Automating quality reports after each reconstruction run.

## References

[1] J. L. Schonberger and J.-M. Frahm, "Structure-from-Motion Revisited," IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.

[2] J. L. Schonberger, E. Zheng, M. Pollefeys, and J.-M. Frahm, "Pixelwise View Selection for Unstructured Multi-View Stereo," European Conference on Computer Vision (ECCV), 2016.

[3] S. Leutenegger, S. Lynen, M. Bosse, R. Siegwart, and P. Furgale, "Keyframe-Based Visual-Inertial Odometry using Nonlinear Optimization," International Journal of Robotics Research, 2015.

[4] T. Qin, P. Li, and S. Shen, "VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator," IEEE Transactions on Robotics, 2018.

[5] C. Campos, R. Elvira, J. J. Gomez Rodriguez, J. M. M. Montiel, and J. D. Tardos, "ORB-SLAM3: An Accurate Open-Source Library for Visual, Visual-Inertial, and Multi-Map SLAM," IEEE Transactions on Robotics, 2021.

[6] C. Forster, L. Carlone, F. Dellaert, and D. Scaramuzza, "IMU Preintegration on Manifold for Efficient Visual-Inertial Maximum-a-Posteriori Estimation," Robotics: Science and Systems, 2015.

[7] M. Youssef and A. Agrawala, "The Horus WLAN Location Determination System," ACM MobiSys, 2005.

[8] A. Haeberlen, E. Flannery, A. M. Ladd, A. Rudys, D. S. Wallach, and L. E. Kavraki, "Practical Robust Localization over Large-Scale 802.11 Wireless Networks," ACM MobiCom, 2004.

[9] "IndoorGNN: A Graph Neural Network based approach for Indoor Localization using WiFi RSSI," 2023.

[10] Y. Lyu et al., "ScanNet++: A High-Fidelity Dataset of 3D Indoor Scenes," 2023.

[11] B. Kerbl, G. Kopanas, T. Leimkuhler, and G. Drettakis, "3D Gaussian Splatting for Real-Time Radiance Field Rendering," ACM Transactions on Graphics, 2023.

[12] M. Sayed, J. Gibson, J. Watson, V. Prisacariu, M. Firman, and C. Godard, "SimpleRecon: 3D Reconstruction Without 3D Convolutions," European Conference on Computer Vision (ECCV), 2022.

[13] "LFM-3D: Learnable Feature Matching Across Wide Baselines Using 3D Signals," 2024.

[14] "MP-SfM: Monocular Surface Priors for Robust Structure-from-Motion," 2025.

[15] "A Review on 3D Gaussian Splatting for Sparse View Reconstruction," 2025.
## Thesis Chapter Mapping

This summary can be expanded into the following FYP thesis structure:

1. **Introduction**  
   Explain the motivation for indoor virtual tours and the limitations of GPS and pure visual reconstruction.

2. **Background and Related Work**  
   Discuss COLMAP/SfM, indoor positioning, WiFi fingerprinting, PDR, and multimodal mapping.

3. **System Design**  
   Present the overall architecture, data collection strategy, and module responsibilities.

4. **Implementation**  
   Describe preprocessing, IMU step detection, heading estimation, PDR generation, WiFi/PDR fusion, image binding, video frame extraction, dataset export, COLMAP workspace preparation, and localization.

5. **Experiments and Results**  
   Report reconstruction outputs, point cloud quality, localization evaluation, and comparison between unconstrained and constrained matching if available.

6. **Discussion**  
   Analyze strengths, failure cases, limitations, and practical deployment considerations.

7. **Conclusion and Future Work**  
   Summarize the achieved system and propose future improvements.

## Final Conclusion

This FYP demonstrates a practical approach to indoor virtual campus tour construction by combining visual reconstruction with WiFi and IMU-based spatial metadata. The project does not attempt to replace COLMAP with signal or inertial data; instead, it uses WiFi and IMU-derived PDR to organize images, constrain matching, generate rough spatial references, and support indoor localization. The final pipeline provides a complete route from smartphone data collection to a PDR-enabled COLMAP reconstruction workspace, dense point cloud generation scripts, and WiFi positioning assets. With further development into an interactive viewer, the system can become a usable foundation for virtual campus navigation, indoor mapping, and location-aware campus services.
