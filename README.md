# 3D Reconstruction Signal Vision Pipeline

这个项目把手机采集到的 WiFi、IMU、点位照片和路线视频整理成一条可运行的室内 3D 重建流水线。

核心目标：

- 用照片/视频抽帧给 COLMAP 生成点云输入
- 用 IMU/PDR 给图像提供粗位置约束
- 用 WiFi 点位生成指纹库，用于后续定位
- 用 WiFi/PDR 采样点生成粗边界和信号覆盖模型
- 用 constrained matching 减少室内重复纹理导致的 COLMAP 乱匹配

注意：COLMAP 本身只直接使用图像。WiFi/IMU 不会直接变成点云，而是在前后处理里变成位置种子、匹配约束、边界和定位模型。

## 1. 目录结构

推荐把数据放成这样：

```text
FYP_3d/
  main.py
  preprocess_pipeline.py
  image_mapping.py
  video_frames.py
  reconstruction_export.py
  wifi_positioning.py
  colmap_bridge.py
  localize_wifi.py
  run_colmap.py
  run_colmap_sparse.bat
  run_colmap_dense.bat
  run_colmap_full.bat
  DATA_COLLECTION_GUIDE.md

  pic/
  video/
  2026-xx-xx-xx-xx-xx-xxx.sess/
  colmap-x64-windows-cuda/
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

## 2. 一键预处理

在 PowerShell 或 VS Code Terminal 里进入项目目录：

```powershell
cd C:\Users\ASUS\Desktop\FYP_3d
```

运行：

```powershell
python main.py
```

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

说明预处理和 COLMAP workspace 准备完成。

## 3. 流水线逻辑

### Step 1: 解析 WiFi 原始数据

输入：

```text
*.sess/wifiScanner.dat
```

输出：

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
colmap_workspace/metadata/
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
YES
```

然后运行 sparse：

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

## 5. 最终点云

最终 dense 点云：

```text
colmap_workspace/dense/fused.ply
```

可以用 MeshLab 或 CloudCompare 打开。

如果 PDR 对齐成功，还会有：

```text
colmap_workspace/sparse_pdr_aligned/
```

dense 阶段会优先使用 `sparse_pdr_aligned/0`。

## 6. 常用参数

默认每个 WiFi 点位 20 张照片。如果你临时不是 20 张，可以改：

```powershell
python main.py --images-per-wifi-point 16
```

视频每 0.5 秒抽 1 帧：

```powershell
python main.py --video-frame-interval-s 0.5
```

视频不限制最大抽帧数量：

```powershell
python main.py --video-max-frames 0
```

跳过视频抽帧：

```powershell
python main.py --skip-video-frames
```

收紧 COLMAP 匹配约束：

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

```powershell
cd C:\Users\ASUS\Desktop\FYP_3d
python main.py
python run_colmap.py reset
python run_colmap.py sparse
python run_colmap.py dense
```

最终查看：

```text
colmap_workspace/dense/fused.ply
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
