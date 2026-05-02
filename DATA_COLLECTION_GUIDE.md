# Data Collection Guideline

这份指南用于下一轮采集数据时提高 COLMAP 点云质量，同时保留 WiFi/IMU 对定位和边界的价值。

## 核心原则

COLMAP 需要连续、有重叠、有平移视差的图像。WiFi/IMU 需要稳定的点位、时间顺序和空间覆盖。

推荐采集方式：

```text
WiFi 点位照片 = 强定位锚点
路线视频抽帧 = 视觉基底和过渡帧
```

也就是说：点位照片负责和 WiFi 指纹对应，视频负责让 COLMAP 更容易把不同点位连成完整结构。

## 推荐采集流程

1. 打开手机 WiFi/IMU 采集，开始记录 `.sess`。
2. 同时开始录一段连续路线视频，尽量不要中断。
3. 沿路线慢慢走，到达一个 WiFi 点位时停下。
4. 在该点位停留 3 到 5 秒，让 WiFi 扫描稳定。
5. 触发一次 `CHECKPOINT_PAUSE`。
6. 在同一个点位拍 20 张照片，覆盖一圈方向。
7. 继续边走边录视频，到下一个点位重复上面的步骤。
8. 最后尽量走回起点附近，形成闭环。

## 点位照片怎么拍

每个 WiFi 点位默认拍 20 张。可以理解成围绕身体转一圈：

```text
360 degrees / 20 photos = 每张约 18 degrees
```

注意：

- 同一个点位拍照时，手机位置尽量不变，只转方向。
- 不要只拍白墙、地板、天花板。
- 多包含门框、墙角、桌椅、窗框、海报、柜子等稳定特征。
- 相邻照片最好有 60% 到 80% 重叠。
- 避免过曝、欠曝、模糊、强反光。

这些照片会被代码强绑定到当前 WiFi 点位，进入：

```text
processed/image_wifi_long.csv
reconstruction_dataset/wifi_fingerprint_library.csv
```

## 视频怎么录

视频建议从起点开始一路录到终点，作为整个空间的视觉基底。

建议：

- 从 P1 走到 P2 的过程也保持录像。
- 移动时镜头朝前方或斜前方，不要剧烈甩动。
- 走慢一点，让画面有连续重叠。
- 尽量拍到墙角、门框、桌椅边缘等特征。
- 如果空间较大，可以录一段主路线视频，必要时再录补充路线视频。

代码默认会从 `video/` 目录抽帧：

```text
video/route.mp4
  -> processed/video_frames/
  -> processed/video_frames_manifest.csv
  -> reconstruction_dataset/images/
  -> colmap_workspace/images/
```

视频帧不会进入 WiFi 指纹库。它们只是弱绑定到路线顺序，用来帮助 COLMAP 连接相邻点位。

## 照片和信号怎么对应

当前代码的对应关系是：

```text
WiFi 点位 P1
  -> 20 张点位照片，强绑定 P1 的 WiFi 指纹
  -> P1 附近的视频帧，弱绑定到 P1/P2 之间

WiFi 点位 P2
  -> 20 张点位照片，强绑定 P2 的 WiFi 指纹
  -> P2 附近的视频帧，弱绑定到 P2/P3 之间
```

所以你不需要给视频帧单独标 WiFi。只要保证录视频的路线顺序和 WiFi 点位采集顺序一致，代码会按路线顺序把视频帧插值到点位之间。

## 路线设计建议

- 相邻 WiFi 点位间隔建议 1 到 3 米。
- 空间大时沿墙边、走廊、桌椅区域都走一遍。
- 尽量形成闭环，有助于减少漂移。
- 不要只在点位原地旋转，否则点云容易断开。
- 走动过程中也要保持视频画面稳定和连续。

## 当前代码里的约束

`python main.py` 会生成这些和约束有关的文件：

```text
colmap_workspace/match_pairs.txt
colmap_workspace/pdr_alignment_refs.txt
colmap_workspace/metadata/wifi_fingerprint_library.csv
colmap_workspace/metadata/wifi_boundary_seed.csv
colmap_workspace/metadata/video_frames_manifest.csv
```

含义：

- `match_pairs.txt`：限制 COLMAP 只优先匹配相邻 WiFi 点、相邻拍摄方向、PDR 距离近的图，以及连续视频帧。
- `pdr_alignment_refs.txt`：只使用点位照片的 PDR 坐标做模型对齐参考。
- `wifi_fingerprint_library.csv`：WiFi 指纹库，只来自点位照片对应的 WiFi 点。
- `wifi_boundary_seed.csv`：WiFi/PDR 点位形成的粗边界。
- `video_frames_manifest.csv`：视频基底帧清单，属于视觉辅助，不是强 WiFi 标签。

## 推荐运行方式

预处理：

```powershell
python main.py
```

重新跑 COLMAP：

```powershell
python run_colmap.py reset
python run_colmap.py sparse
python run_colmap.py dense
```

如果点云仍然断开，可以先试试放宽相邻点位匹配：

```powershell
python main.py --constrained-point-window 2
python run_colmap.py reset
python run_colmap.py sparse
python run_colmap.py dense
```

如果点云噪声很多，可以收紧同点位方向匹配：

```powershell
python main.py --constrained-slot-window 1
```

## 文件放置检查

采集回来以后，先检查：

```text
pic/        每个 WiFi 点位 20 张照片
video/      至少 1 段路线视频
*.sess/     WiFi/IMU 原始数据
```

如果有 14 个 WiFi 点位，`pic/` 里应该有：

```text
14 * 20 = 280 张照片
```

视频可以先只放一段主路线视频。跑通后如果发现某些区域点云断开，再补录覆盖那些区域的视频。
