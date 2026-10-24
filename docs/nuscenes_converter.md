# nuScenes 地图任务数据转换

`tools/dataset_converters/nuscenes_map_converter.py` 使用 nuScenes SDK，生成
MMEngine 标准的 `metainfo + data_list` PKL。它保存六路相机、当前帧
`LIDAR_TOP`、地图文件引用和时序关系，不读取 CAN bus、历史 LiDAR sweeps、
图像像素、点云内容或地图几何，不生成局部地图 GT。

## 使用方法

`tools/create_data.py` 是统一入口，通过数据集子命令分派到对应 converter。
目前仅提供 `nuscenes`，必须显式指定版本，每次只转换所选版本。在开发容器
的仓库根目录运行：

```bash
python tools/create_data.py nuscenes \
    --data-root /home/guyu/data/nuscenes/mini \
    --version v1.0-mini \
    --out-dir work_dirs/nuscenes_infos
```

输出：

```text
work_dirs/nuscenes_infos/mmhdmap_nuscenes_mini_infos_train.pkl
work_dirs/nuscenes_infos/mmhdmap_nuscenes_mini_infos_val.pkl
```

| `--version` | 输出 split | 文件名前缀（默认） |
| --- | --- | --- |
| `v1.0-mini` | train、val | `mmhdmap_nuscenes_mini_infos_` |
| `v1.0-trainval` | train、val | `mmhdmap_nuscenes_trainval_infos_` |
| `v1.0-test` | test | `mmhdmap_nuscenes_test_infos_` |

例如转换完整训练验证集或测试集时，分别执行：

```bash
python tools/create_data.py nuscenes \
    --data-root /home/guyu/data/nuscenes \
    --version v1.0-trainval \
    --out-dir work_dirs/nuscenes_infos

python tools/create_data.py nuscenes \
    --data-root /home/guyu/data/nuscenes \
    --version v1.0-test \
    --out-dir work_dirs/nuscenes_infos
```

`--data-root` 指向包含所选版本目录、samples 和 maps 的数据根目录。
统一入口也接受 MapTRv2 风格的 `--extra-tag`，它是 `--info-prefix` 的别名。
数据根目录参数统一使用 `--data-root`。不接受笼统的 `--version v1.0`，
也不会自动串联 trainval 和 test。可用
`python tools/create_data.py nuscenes --help` 查看数据集参数。
统一入口使用 `python tools/create_data.py nuscenes ...` 直接运行脚本。
原 converter 仍可独立运行。

默认采用官方场景划分，不继承 HRDX 的隐式 `FAIL_SCENES` 排除规则。
所有选中场景的六路图像、
当前点云和 `maps/expansion/<location>.json` 必须存在，否则报错并在写出前
停止。底图 PNG 不是必需文件。

`--info-prefix` 可以修改文件名前缀；已有文件默认不能覆盖，需要显式传入
`--overwrite`。自定义划分建议使用不同前缀或输出目录。

## 自定义划分

通过 `--split-file path/to/split.json` 传入场景清单，例如：

```json
{
  "name": "custom-mini",
  "version": "v1.0-mini",
  "splits": {
    "train": ["scene-0061"],
    "val": ["scene-0103"]
  }
}
```

清单中的 version 必须匹配数据集版本，场景必须存在且不能重复或跨 split
重叠。mini/trainval 要求非空的 train 和 val；test 版本只接受非空的 test。
未列出的场景不会导出。HRDX new split 可将其场景列表整理成这一格式，
明确排除不参与实验的场景。

## 输出协议 v1.0

顶层包含：

- `metainfo`：数据集名称和版本、协议版本、划分名称、实际场景清单、
  时间戳单位、参考传感器和相机顺序。
- `data_list`：每帧的 metadata 字典。它不是已经经过 pipeline 的
  `MapDataSample`。

单帧字段：

| 字段 | 含义 |
| --- | --- |
| `token` | SDK sample token |
| `scene_token`、`scene_name` | 场景标识 |
| `frame_idx` | 场景内从 0 开始的帧号 |
| `timestamp` | SDK sample 时间戳 |
| `prev_token`、`next_token` | 原始 sample 前后帧 token；边界为空字符串 |
| `location`、`map_path` | 地图区域名和完整地图 JSON 相对路径 |
| `ego2global` | 当前 LiDAR 采集时刻的 ego 到 global 变换 |
| `lidar2ego` | LiDAR 到 ego 的静态标定 |
| `lidar_points` | 当前点云相对路径、sample_data token、采集时间戳、点特征维数 5 |
| `images` | 以六个相机名称为键的相机信息 |

每个相机包含 `img_path`、`sample_data_token`、`timestamp`、`width`、
`height`、`cam2img`、`ego2cam`、`ego2global`。`ego2cam` 是静态外参；
相机内部的 `ego2global` 对应该相机的采集时刻，不是参考 LiDAR 时刻。

位姿采用 float64 的 4×4 矩阵、列向量、米；内参是 float64 的 3×3 矩阵。
所有时间戳保持 SDK 的整数微秒。所有文件路径相对于 data_root，与 PKL
所在目录无关。样本按场景名称和场景内 sample 链确定顺序，不依赖 SDK
表的存储顺序。

时间补偿可由这些基础矩阵推导：

```python
camera = sample['images']['CAM_FRONT']
global2cam = camera['ego2cam'] @ np.linalg.inv(camera['ego2global'])
lidar2cam = global2cam @ sample['ego2global'] @ sample['lidar2ego']
```

转换器不固定地图 ROI、局部地图参考系或地图投影高度。这些仍由地图提取
工具和 pipeline 决定。

## 与当前 Dataset 的关系

`NuscMapDataset` 现在复用 MMEngine `BaseDataset.load_data_list()` 读取新协议，
在 `parse_data_info()` 中拼接嵌套路径并验证矩阵。只接受 info_version 1.0，
不再兼容旧 HRDX `list[dict]`。`ann_path_prefix`、旧路径剥离逻辑及相关
TODO 已移除；地图路径直接以 data_root 为前缀，避免重复添加 maps/expansion。

`PackMapInputs` 保留 scene_token、frame_idx、prev_token、next_token 及完整
传感器元信息，输出仍为 `inputs + MapDataSample`。sample_idx 是当前
Dataset 索引，frame_idx 是原始场景内帧号；筛选数据后不能把前后帧 token
当作当前列表下标使用。

mini 配置已指向新文件，当前容器的 data_root 为
`/home/guyu/data/nuscenes/mini`。MapLocalVisualizer 使用相机时刻位姿投影，
同时以参考 LiDAR 时刻的 ego 高度定义所有相机共享的地图平面。现有 pipeline
只读取和打包图像，地图 GT 仍未生成；检查脚本在 pipeline 外提取地图用于显示。
