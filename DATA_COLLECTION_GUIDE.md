# Data Collection Guideline for WiFi/IMU/3D Reconstruction

这份指南用于下一次采集数据时提高 COLMAP 点云质量，同时保留 WiFi/IMU 对定位和边界的价值。

## 核心原则

COLMAP 需要的是连续、有重叠、有平移视差的图像。WiFi/IMU 需要的是稳定的点位、时间顺序和空间覆盖。两者要一起采集，而不是只在点位原地转圈。

推荐采集方式：

1. 沿路线慢慢移动拍摄或录视频。
2. 到达 WiFi 点位时停下，手机保持稳定，采集 WiFi/IMU。
3. 在每个 WiFi 点位补拍 8 个方向照片。
4. 点位之间也要拍连续照片或视频帧，帮助 COLMAP 把点位连接起来。

## 推荐流程

### 1. 路线设计

- 先规划一条连续路线，尽量形成闭环。
- 相邻 WiFi 点位之间不要隔太远，建议 1 到 3 米一个点位。
- 如果空间较大，沿墙边、走廊、桌椅区域都走一遍。
- 回到起点附近拍一段闭环数据，有助于减少漂移。

### 2. WiFi/IMU 采集

- 手机开始采集 `.sess` 后不要频繁暂停。
- 每个点位停留 3 到 5 秒，让 WiFi 扫描更稳定。
- 每个点位触发一次 `CHECKPOINT_PAUSE`，之后再继续走。
- 尽量保持手机高度一致，例如胸口高度 1.3 到 1.6 米。
- 走路速度慢一点，避免 IMU 轨迹过度抖动。

### 3. 照片采集

每个 WiFi 点位建议拍 8 张：

```text
0, 45, 90, 135, 180, 225, 270, 315 degrees
```

同时注意：

- 每张照片之间尽量有 60% 到 80% 重叠。
- 不要只拍白墙、地板、天花板。
- 多包含门框、桌椅、墙角、海报、柱子、窗框等稳定特征。
- 避免过曝、欠曝、模糊和强反光。
- 同一个点位拍摄时尽量保持手机位置不变，只旋转方向。

### 4. 点位之间也要拍

只在每个 WiFi 点位原地拍 8 张，容易导致点云断开。更好的方式是：

- 从点位 A 走到点位 B 的过程中继续拍照，或者录视频。
- 如果用视频，后续可以每 0.5 到 1 秒抽一帧。
- 移动时让相机朝前或斜前方，不要剧烈甩动。

这样 COLMAP 能得到平移视差，3D 结构会稳定很多。

## 当前代码里的约束如何使用

当前 pipeline 已经把约束写进 COLMAP workspace：

- `colmap_workspace/match_pairs.txt`
  - 只让 COLMAP 匹配同一 WiFi 点位、相邻 WiFi 点位、相邻拍摄方向、PDR 距离较近的图片。
  - 目的是减少室内重复纹理导致的乱匹配。

- `colmap_workspace/pdr_alignment_refs.txt`
  - 给 COLMAP sparse 模型提供 PDR 坐标参考。
  - sparse 建完后会尝试用这些坐标做模型对齐。

- `colmap_workspace/metadata/wifi_fingerprint_library.csv`
  - WiFi 指纹库，可用于后续定位。

- `colmap_workspace/metadata/wifi_boundary_seed.csv`
  - WiFi/PDR 点位形成的粗边界，可用于后续点云裁剪或空间约束。

## 推荐运行方式

重新生成 workspace：

```powershell
python main.py
```

如果之前已经跑过 COLMAP，先清理旧数据库和旧模型：

```powershell
python run_colmap.py reset
```

然后重新跑：

```powershell
python run_colmap.py sparse
python run_colmap.py dense
```

或者：

```powershell
.\run_colmap_sparse.bat
.\run_colmap_dense.bat
```

## 如果点云仍然很乱

优先检查采集数据：

- 相邻点位之间是否有足够图片重叠。
- 是否只有原地旋转，缺少移动过程。
- 是否拍了太多白墙、玻璃、重复桌椅。
- 是否有大量模糊图片。
- 是否点位间隔太远。

然后可以尝试调整参数：

```powershell
python main.py --constrained-point-window 1 --constrained-slot-window 1
python run_colmap.py reset
python run_colmap.py sparse
python run_colmap.py dense
```

如果模型断开太多，可以放宽：

```powershell
python main.py --constrained-point-window 2 --constrained-slot-window 2
```

## 最推荐的下一轮采集方案

1. 手机开始记录 WiFi/IMU。
2. 从起点开始慢慢走，边走边录视频。
3. 每 1 到 3 米停一次作为 WiFi 点位。
4. 每个点位停 3 到 5 秒，拍 8 个方向照片。
5. 点位之间继续保持视频或照片采集。
6. 最后走回起点附近形成闭环。
7. 用 `python main.py` 处理数据，再用 constrained COLMAP 流程重建。
