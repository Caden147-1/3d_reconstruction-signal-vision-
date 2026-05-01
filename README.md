# FYP 3D WiFi/IMU/Photo Reconstruction Pipeline

这个项目用于把手机采集的 `.sess` 数据、WiFi/IMU 信息和现场照片整理成可接入 COLMAP 的 3D 重建工作流。

## 目录要求

项目根目录应类似这样：

```text
FYP_3d/
  main.py
  preprocess_raw_data.py
  preprocess_pipeline.py
  image_mapping.py
  reconstruction_export.py
  colmap_bridge.py
  run_colmap_sparse.bat
  run_colmap_dense.bat
  run_colmap_full.bat
  run_colmap.py
  pic/
  2026-05-01-10-19-15-725.sess/
  colmap-x64-windows-cuda/
```

其中：

- `pic/` 存放采集照片。
- `.sess/` 文件夹存放手机采集的 WiFi、IMU、movement 数据。
- `colmap-x64-windows-cuda/` 是 COLMAP 程序目录，里面应有 `bin/colmap.exe`。

当前默认规则是：**一个 WiFi 点位对应 8 张照片**，照片按文件名数字顺序分组。

## 第一步：预处理数据

在 PowerShell 或 VS Code Terminal 中进入项目根目录：

```powershell
cd C:\Users\ASUS\Desktop\FYP_3d
```

运行：

```powershell
python main.py
```

这一步会自动生成：

```text
processed/
reconstruction_dataset/
colmap_workspace/
```

主要结果：

- `processed/image_wifi_points.csv`：一个 WiFi 点位一行，每行绑定 8 张图片。
- `processed/image_wifi_long.csv`：一张图片一行，带 WiFi/PDR 信息。
- `processed/wifi_pdr_features.csv`：WiFi 指纹和 PDR 轨迹融合结果。
- `reconstruction_dataset/wifi_fingerprint_library.csv`：WiFi 指纹库。
- `reconstruction_dataset/wifi_boundary_seed.csv`：WiFi/PDR 粗边界。
- `colmap_workspace/images/`：给 COLMAP 使用的图片。
- `colmap_workspace/metadata/`：WiFi、IMU、pose seed 等元数据。

如果运行成功，终端中应看到：

```text
[done] preprocessing complete
```

## 第二步：运行 COLMAP

第一次建议先跑稀疏重建：

```powershell
.\run_colmap_sparse.bat
```

如果稀疏重建成功，再跑 dense 点云：

```powershell
.\run_colmap_dense.bat
```

也可以直接完整运行：

```powershell
.\run_colmap_full.bat
```

注意：在 PowerShell 中运行 `.bat` 文件时要加 `.\`。

## 输出点云位置

COLMAP dense 点云输出在：

```text
colmap_workspace\dense\fused.ply
```

这个 `fused.ply` 可以用 MeshLab、CloudCompare 等工具打开、缩放、清理和进一步建 mesh。

稀疏模型输出在：

```text
colmap_workspace\sparse\
```

## WiFi 数据是否参与了流程

COLMAP 本身只使用图片生成点云，不会直接读取 WiFi/IMU。

本项目已经把 WiFi/IMU 用作空间元数据：

- 用 WiFi scan 生成指纹库。
- 用 WiFi 点位绑定照片。
- 用 PDR 给图片和 WiFi 点位提供空间坐标。
- 用 WiFi/PDR 点位生成粗边界。
- 将这些数据复制到 `colmap_workspace/metadata/`，用于后续点云对齐、定位、裁剪和边界约束。

## 常用命令

重新生成所有预处理和 COLMAP workspace：

```powershell
python main.py
```

只跑 COLMAP 稀疏模型：

```powershell
.\run_colmap_sparse.bat
```

只跑 COLMAP dense 点云：

```powershell
.\run_colmap_dense.bat
```

完整跑 COLMAP：

```powershell
.\run_colmap_full.bat
```

用 Python 启动 COLMAP 脚本：

```powershell
python run_colmap.py sparse
python run_colmap.py dense
python run_colmap.py full
```

打开 COLMAP GUI：

```powershell
.\open_colmap_gui.bat
```

## 常见问题

### PowerShell 提示找不到 bat

如果输入：

```powershell
run_colmap_full.bat
```

PowerShell 可能提示找不到命令。请改成：

```powershell
.\run_colmap_full.bat
```

### 出现 `Clamping features`

类似：

```text
Clamping features from 38157 to 32768
```

这是 COLMAP 的 warning，不是致命错误。它表示匹配点较多，COLMAP 按默认上限裁剪了一部分。一般可以先继续跑。

### dense 阶段提示找不到 sparse/0

说明稀疏重建还没成功，或者 COLMAP 生成的稀疏模型目录不是 `sparse/0`。

先运行：

```powershell
.\run_colmap_sparse.bat
```

确认 `colmap_workspace\sparse\0` 是否存在，再运行 dense。

### 修改照片数量或每点图片数

默认每个 WiFi 点位 8 张图。如果采集规则变了，可以运行：

```powershell
python main.py --images-per-wifi-point 6
```

## 推荐启动顺序

```powershell
cd C:\Users\ASUS\Desktop\FYP_3d
python main.py
.\run_colmap_sparse.bat
.\run_colmap_dense.bat
```

最终查看：

```text
colmap_workspace\dense\fused.ply
```
