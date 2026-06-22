# FYP Publication Summary

## Project Title

**Virtual Campus Tour System: Multimodal Indoor 3D Reconstruction with WiFi/IMU-Aided Positioning**

Authors: Yuxiao Long, Yuxuan Liu  
Degree: Bachelor of Science (Honours) in Computer Science and Technology  
Institution: Beijing Normal-Hong Kong Baptist University  
Date: May 2026

## Executive Summary

This Final Year Project develops a practical pipeline for building an indoor virtual campus tour from smartphone-collected visual, wireless, and inertial data. The system combines route videos, checkpoint photographs, WiFi scans, and IMU measurements to support indoor 3D reconstruction and signal-based localization. Instead of treating 3D reconstruction as a purely visual task, the project uses WiFi and pedestrian dead reckoning (PDR) derived from IMU data as supporting spatial constraints. These constraints help organize image data, generate localization metadata, reduce incorrect visual matching in repetitive indoor environments, and provide a reusable WiFi fingerprint library for later positioning.

The final system is designed around an end-to-end workflow. A user collects data with a phone by walking through the indoor environment, recording WiFi/IMU data, taking checkpoint photos, and filming a continuous route video. The project scripts then parse the raw session files, estimate a relative walking trajectory from IMU acceleration and heading readings, bind checkpoint images to WiFi/PDR points, extract video frames, export a reconstruction dataset, prepare a COLMAP workspace, and generate sparse and dense 3D reconstruction outputs. The main dense point cloud can be opened in tools such as MeshLab or CloudCompare, while the generated WiFi localization model can estimate positions from new WiFi scans.

Under the assumed final state, the project has been updated according to the data collection and pipeline guidelines: each WiFi checkpoint is linked with a fixed group of photos, route video frames are used as visual transition frames, IMU-derived PDR provides rough route geometry, COLMAP matching is constrained using route order and PDR distance, and WiFi/PDR metadata is exported alongside the visual reconstruction dataset. The result is a reproducible software pipeline that connects indoor mapping, inertial tracking, visual reconstruction, and lightweight signal-based localization.

## Abstract

Indoor virtual tour systems require reliable spatial reconstruction, but indoor environments often create challenges for image-based reconstruction due to repeated textures, weak visual features, reflective surfaces, and limited GPS availability. This project proposes a multimodal pipeline for constructing an indoor virtual campus tour using smartphone videos, checkpoint photographs, WiFi scans, and IMU data. The system processes WiFi scans to generate fingerprint features and uses IMU linear acceleration, rotation vectors, and movement events to estimate a relative PDR trajectory. Checkpoint photographs are strongly associated with WiFi/PDR points, while route video frames are extracted as weakly constrained visual bridge frames for COLMAP reconstruction. The pipeline exports image manifests, camera pose seeds, WiFi fingerprint libraries, signal boundaries, constrained COLMAP matching pairs, and PDR alignment references. A KNN-based WiFi localization model is also produced for estimating positions from new WiFi scans. By combining visual reconstruction with wireless signal features and inertial motion metadata, the project improves the organization of indoor image data and reduces the risk of incorrect matching in repetitive spaces. The final system provides a reproducible workflow from raw mobile data collection to dense point cloud generation and indoor localization support, forming a technical foundation for a virtual campus tour application.

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

The assumed final data collection follows the updated project guideline:

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

## Assumed Final Results

Assuming the project has been completed according to the latest Markdown guidance, the expected final deliverables are:

- A working preprocessing pipeline from raw phone session data to COLMAP-ready images and metadata.
- A processed IMU/PDR dataset containing movement segments, detected steps, heading estimates, and relative path coordinates.
- A processed WiFi/PDR dataset with selected AP features and relative walking trajectory.
- A checkpoint image mapping table linking photos to WiFi fingerprint points.
- Extracted route video frames used as visual transition frames.
- A reconstruction dataset containing images, manifests, pose seeds, WiFi fingerprints, and boundary metadata.
- A prepared COLMAP workspace with constrained matching pairs.
- A generated sparse reconstruction model.
- A generated dense point cloud file at `colmap_workspace/dense/fused.ply`.
- A WiFi localization model and evaluation output.
- Documentation for repeatable data collection and execution.

Suggested result table for the final report:

| Metric | Value |
| --- | --- |
| Number of WiFi checkpoints | To be filled |
| Photos per checkpoint | 20 by default |
| Total checkpoint photos | To be filled |
| Number of route videos | To be filled |
| Extracted video frames | To be filled |
| Detected PDR steps | To be filled |
| Estimated PDR walking distance | To be filled |
| Movement segments | To be filled |
| Selected WiFi APs | To be filled |
| Sparse reconstruction registered images | To be filled |
| Sparse 3D points | To be filled |
| Dense point cloud file | `colmap_workspace/dense/fused.ply` |
| WiFi localization mean error | To be filled |
| WiFi localization median error | To be filled |

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

This FYP demonstrates a practical approach to indoor virtual campus tour construction by combining visual reconstruction with WiFi and IMU-based spatial metadata. The project does not attempt to replace COLMAP with signal or inertial data; instead, it uses WiFi and IMU-derived PDR to organize images, constrain matching, generate rough spatial references, and support indoor localization. The final pipeline provides a complete route from smartphone data collection to dense point cloud generation and WiFi positioning assets. With further development into an interactive viewer, the system can become a usable foundation for virtual campus navigation, indoor mapping, and location-aware campus services.
