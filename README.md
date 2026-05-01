# 3D Reconstruction Signal Vision Pipeline

本项目把手机采集的 WiFi、IMU、照片数据整理成一条完整流水线，用于：

- 生成 COLMAP 点云输入
- 构建 WiFi 指纹库
- 用 WiFi/PDR 生成空间边界
- 用 IMU/PDR 约束照片位置
- 用受限匹配减少室内 COLMAP 乱匹配
- 后续支持新 WiFi scan 定位

COLMAP 本身只使用图像重建点云；本项目在 COLMAP 前后把 WiFi/IMU 变成可用的空间约束和定位模型。

## 1. 项目目录

项目根目录示例：

```text
FYP_3d/
  main.py
  preprocess_raw_data.py
  preprocess_pipeline.py
  image_mapping.py
  reconstruction_export.py
  wifi_positioning.py
  colmap_bridge.py
  localize_wifi.py
  run_colmap.py
  run_colmap_sparse.bat
  run_colmap_dense.bat
  run_colmap_full.bat
  open_colmap_gui.bat
  DATA_COLLECTION_GUIDE.md
  pic/
  2026-05-01-10-19-15-725.sess/
  colmap-x64-windows-cuda/
```

输入数据：

- `pic/`：照片目录
- `*.sess/`：手机采集的 WiFi、IMU、movement 数据
- `colmap-x64-windows-cuda/`：COLMAP 程序目录，内部需要有 `bin/colmap.exe`

默认采集规则：

```text
1 个 WiFi 点位 = 8 张照片
```

照片会按文件名数字顺序分组。

## 2. 一键生成预处理结果

在 PowerShell 或 VS Code Terminal 中进入项目目录：

```powershell
cd C:\Users\ASUS\Desktop\FYP_3d
```

运行：

```powershell
python main.py
```

这一步会自动完成：

```text
raw .sess + pic
  -> processed/
  -> reconstruction_dataset/
  -> colmap_workspace/
```

如果看到：

```text
[done] preprocessing complete
```

说明预处理流水线完成。

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
- 生成 selected AP RSSI 特征表

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

- 每条 WiFi scan 绑定一个 PDR 空间位置
- 后续 WiFi 指纹库和边界都基于这个坐标

### Step 4: 绑定照片和 WiFi 点位

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
- 每 8 张照片绑定到一个 `CHECKPOINT_PAUSE` WiFi 点位
- 每张照片携带 WiFi/PDR 信息

### Step 5: 导出 3D 重建数据集

输出：

```text
reconstruction_dataset/images/
reconstruction_dataset/images_manifest.csv
reconstruction_dataset/camera_pose_seed.csv
reconstruction_dataset/wifi_fingerprint_library.csv
reconstruction_dataset/wifi_boundary_seed.csv
```

作用：

- 把照片复制并重命名成 COLMAP 友好的格式
- 给每张照片生成 PDR 初始位置和 8 方向 yaw seed
- 生成 WiFi 指纹库
- 生成 WiFi/PDR 粗边界

### Step 6: 生成 WiFi 定位模型和信号边界

输出：

```text
reconstruction_dataset/wifi_localization_model.json
reconstruction_dataset/wifi_localization_eval.csv
reconstruction_dataset/wifi_positioning_summary.json
reconstruction_dataset/wifi_signal_grid.csv
reconstruction_dataset/wifi_signal_boundary.csv
```

作用：

- `wifi_localization_model.json`：加权 KNN WiFi 指纹定位模型
- `wifi_localization_eval.csv`：留一法定位误差评估
- `wifi_signal_grid.csv`：室内区域 RSSI 插值网格
- `wifi_signal_boundary.csv`：带信号强度指标的 WiFi/PDR 边界

当前模型是离线指纹定位：

```text
新 WiFi scan -> AP RSSI 向量 -> 匹配指纹库 -> 输出估计 pdr_x, pdr_y
```

### Step 7: 准备 COLMAP workspace

输出：

```text
colmap_workspace/images/
colmap_workspace/image_list.txt
colmap_workspace/match_pairs.txt
colmap_workspace/pdr_alignment_refs.txt
colmap_workspace/metadata/
```

作用：

- `images/`：COLMAP 输入照片
- `match_pairs.txt`：用 WiFi 点位顺序、拍摄方向、PDR 距离生成的受限匹配对
- `pdr_alignment_refs.txt`：用 PDR 坐标作为 sparse 模型对齐参考
- `metadata/`：复制 WiFi 指纹、信号边界、定位模型、PDR seed 等文件

默认 COLMAP 不再全量乱匹配，而是使用 constrained matching。

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

或者：

```powershell
.\run_colmap_sparse.bat
```

成功后运行 dense：

```powershell
python run_colmap.py dense
```

或者：

```powershell
.\run_colmap_dense.bat
```

也可以完整运行：

```powershell
python run_colmap.py full
```

或：

```powershell
.\run_colmap_full.bat
```

## 5. 点云输出

最终 dense 点云：

```text
colmap_workspace/dense/fused.ply
```

可以用 MeshLab 或 CloudCompare 打开。

稀疏模型：

```text
colmap_workspace/sparse/
```

如果 PDR 对齐成功，还会有：

```text
colmap_workspace/sparse_pdr_aligned/
```

dense 阶段会优先使用 `sparse_pdr_aligned/0`。

## 6. WiFi 边界如何使用

边界相关文件：

```text
reconstruction_dataset/wifi_boundary_seed.csv
reconstruction_dataset/wifi_signal_boundary.csv
reconstruction_dataset/wifi_signal_grid.csv
colmap_workspace/metadata/wifi_signal_boundary.csv
colmap_workspace/metadata/wifi_signal_grid.csv
```

含义：

- `wifi_boundary_seed.csv`：WiFi/PDR 点位凸包，表示采样空间粗边界
- `wifi_signal_boundary.csv`：边界点附带 RSSI 统计，例如强 AP 数量、最强 AP、top3 平均 RSSI
- `wifi_signal_grid.csv`：边界内部 1 米网格的 RSSI 插值和 inside confidence

这些文件用于后续：

- 点云裁剪
- 室内区域约束
- WiFi 覆盖区域可视化
- 定位结果合法性检查

## 7. WiFi 指纹定位如何使用

指纹库：

```text
reconstruction_dataset/wifi_fingerprint_library.csv
```

定位模型：

```text
reconstruction_dataset/wifi_localization_model.json
```

如果你有一条新的 WiFi scan，格式是 JSON 数组，例如：

```json
[
  {"mac": "1C946885EEB1", "dbm": -50},
  {"mac": "1C946885EEB3", "dbm": -53}
]
```

保存为：

```text
new_scan.json
```

运行：

```powershell
python localize_wifi.py --scan-json new_scan.json
```

如果是 CSV，列名包含 `mac,dbm` 或 `mac,rssi`：

```powershell
python localize_wifi.py --scan-csv new_scan.csv
```

输出会包含：

```text
estimated_x
estimated_y
confidence
neighbors
```

也可以保存 JSON：

```powershell
python localize_wifi.py --scan-json new_scan.json --output-json location_result.json
```

## 8. 常用参数

每个 WiFi 点位不是 8 张图时：

```powershell
python main.py --images-per-wifi-point 6
```

调整 WiFi 定位 KNN 的 K：

```powershell
python main.py --wifi-localization-k 5
```

调整 RSSI 信号网格大小：

```powershell
python main.py --wifi-grid-cell-size-m 0.5
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

## 9. 推荐完整运行顺序

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

## 10. 采集指南

下一轮采集请看：

```text
DATA_COLLECTION_GUIDE.md
```

核心建议：

- 不要只在点位原地转圈
- 点位之间也要拍照或录视频
- 相邻照片保持 60% 到 80% 重叠
- 每 1 到 3 米设置一个 WiFi 点位
- 每个 WiFi 点位停留 3 到 5 秒
- 每个点位拍 8 个方向
- 尽量形成闭环
