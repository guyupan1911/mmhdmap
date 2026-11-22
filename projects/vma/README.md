# VMA 迁移

当前已完成 SD 数据准备、`BevMapDataset`、原始矢量 GT 加载与打包，以及彩色/强度底图上的 GT 可视化。模型输入保持为单张彩色图；强度图仅在可视化时按需读取。VMA 模型、训练目标生成和 HRDX 训练尚未接入。

## 数据准备与目录

在仓库根目录执行：

```bash
# 首次准备：先恢复强度大图，再联合裁剪图像和 GT。
python projects/vma/tools/recover_sd_intensity.py --data-root /home/guyu/data/vma/sd_data
python tools/create_data.py sd --data-root /home/guyu/data/vma/sd_data
```

恢复脚本和 converter 分别支持 `--overwrite`，仅在需要重新生成各自产物时使用。Converter 默认输出 `DATA_ROOT/prepared`，可通过 `--out-dir` 修改。

```text
sd_data/
├── line/                         # box、freespace 具有相同结构
│   └── origin_data/
│       ├── image_data/*.jpg      # 原始 6000×6000 伪彩色大图
│       ├── trajectory_data/*.json
│       ├── line_data_6k.json
│       └── intensity_image/
│           ├── *.png             # 近似强度大图，uint8 单通道
│           └── recovery_manifest.json
└── prepared/
    └── line/                     # box、freespace 具有相同结构
        ├── images/*.jpg          # 1000×1000 彩色裁块
        ├── intensity_image/*.png # 相同窗口的无损强度裁块
        ├── sd_map_infos_train.pkl
        └── sd_map_infos_val.pkl
```

不再生成 `sd_image_infos.pkl`。图像和强度使用相同窗口、GT 过滤及集合划分；先恢复强度大图再裁剪，不对重新压缩的小图再次反查颜色。

Line/box 每条有效轨迹等弧长采样 7 个窗口，freespace 为 14 个；跳过无有效 GT 的窗口，保留重复与重叠窗口。按来源大图划分 train/val，前 `round(0.8*N)` 张用于训练，其余用于验证：line 保持 JSON 顺序并先排除没有有效裁块的大图；box/freespace 按图名排序且保留空大图参与划分计数。不随机打乱，同一大图的裁块不跨集合。

| 类型 | train 图块 / 实例 | val 图块 / 实例 |
| --- | --- | --- |
| line | 35 / 422 | 7 / 40 |
| box | 22 / 70 | 5 / 29 |
| freespace | 18 / 37 | 3 / 3 |

三类共 15 张原始大图、90 对裁块、601 个实例。不同来源大图若覆盖相同路段，仍需另行检查空间泄漏；HRDX 应按区域或路段划分。详细几何规则和 pickle 协议见 [SD converter](../../docs/vma/sd_converter.md)。

## Dataset、GT 与模型输入

当前 [sd_images.py](configs/sd_images.py) 的 pipeline：

```text
BevMapDataset
  → mmcv.LoadImageFromFile
  → LoadVectorMapAnnotations
  → PackBEVMapInputs
```

- Dataset 读取 `metainfo + data_list`，解析路径并保留原始 `instances`；不执行裁剪、强度恢复或目标采样。
- `label` 已由 converter 按 `metainfo.classes` 编码；SD line 的顺序为 `lane / curb / stopline`。原始 GT 可通过 `dataset.get_data_info(index)['instances']` 读取。
- 图像读取复用 MMCV 的 Pillow 后端，忽略 EXIF 旋转；打包保持 BGR，不归一化。
- `LoadVectorMapAnnotations` 保留原始点序和像素坐标，不重采样、不自动闭合、不编码属性。

```text
inputs: uint8 Tensor[3, H, W]       # 单个彩色图 tensor

data_samples: MapDataSample
  metainfo:
    sample_id, img_path, ori_shape, img_shape
    source_id, crop_origin
    intensity_img_path, source_intensity_img_path
  gt_instances: InstanceData
    labels: int64 Tensor[N]
    polylines: list[float32 ndarray[Ki, 2]]
    attributes: list[dict]         # 原始字符串属性
```

强度路径只保存在元信息中，模型 pipeline 不加载或打包强度像素。`has_intensity=True` 表示有配对文件，不代表模型使用强度输入。空 `instances=[]` 表示已标注且无目标；缺少 `instances` 表示无标注，推理配置应省略 GT 加载 transform。

GT 在小图像素坐标系中，原点在左上角，x 向右、y 向下，范围为 `[0,999]`。画在小图上直接使用这些坐标；恢复到来源大图才加 `crop_origin`。

## 初始化与可视化

[test.py](tools/test.py) 只负责 Dataset 初始化和 Visualizer 调用，不执行模型推理、额外 GT 校验或详细 JSON 统计。不指定可视化选项时，仅初始化并打印样本数，不读取图像：

```bash
python projects/vma/tools/test.py projects/vma/configs/sd_images.py --split train
python projects/vma/tools/test.py projects/vma/configs/sd_images.py --split val
```

指定 `--out-dir` 保存可视化；`--max-samples` 只限制可视化数量，省略时遍历该集合：

```bash
python projects/vma/tools/test.py projects/vma/configs/sd_images.py \
  --split val --max-samples 3 --draw-endpoints --draw-instance-ids \
  --out-dir work_dirs/vma/visualization/gt/line/val
```

通过 `--data-root /home/guyu/data/vma/sd_data/prepared/box` 或 `.../freespace` 切换类型。有显示设备时可加 `--show --wait-time 0.5`；`--wait-time 0` 等待交互。

脚本将彩色 tensor 从 BGR 转成 RGB，并在需要可视化时按 `intensity_img_path` 额外读取强度图。Visualizer 接收数组，不读取文件；强度图复制为三通道用于彩色 GT 绘制，不做 log 映射或对比度拉伸。

```text
OUT_DIR/
├── 000000_sample_5_0.png
└── intensity_image/
    └── 000000_sample_5_0.png
```

默认按类别着色并显示类别名称；不自动闭合或填充轮廓，预测绘制尚未实现。

| 参数 | 用途 |
| --- | --- |
| `--draw-points` | 显示原始顶点 |
| `--draw-endpoints` | 圆圈标起点，叉号标终点 |
| `--draw-instance-ids` | 显示裁块内实例编号 |
| `--draw-attributes` | 显示原始属性 |
| `--no-gt` | 只显示底图 |

当前 pipeline 无几何增强，配对底图可以直接使用相同 GT。以后若增加 resize、flip、crop，必须同步变换强度底图后再绘制，不能只按原始路径加载。

## 强度恢复与输入切换决策

强度恢复使用标准 OpenCV JET 色表：按 RGB 平方距离寻找最近条目，输出 0–255 索引。JPEG 压缩、未知色表差异及未知的原始归一化范围，使它只能表示近似强度，不能精确恢复原始点云 intensity。生成记录和原图哈希保存在 `recovery_manifest.json`。此前的单图强度分布与 log 映射实验见 [强度恢复记录](../../docs/vma/intensity_recovery.md)。

目前保留单图 tensor 接口。强度只用于可视化，因此不放进训练 pipeline，也不需要双图输入字典。MMCV `LoadImageFromFile` 固定读取 `img_path` 并输出 `img`；`KeyMapper` 可在 transform 前后映射字段名，但当前配置无需用它额外加载强度图。

后续若将强度作为模型输入，建议由配置选择输入路径，加载后统一使用 `img`。可以先复制灰度为三通道以复用 backbone 的结构和 checkpoint，再独立验证单通道首层及其权重适配。切换输入还需明确归一化参数并重新验证效果；输入选择 transform、单通道训练和双模态融合目前均未实现。

## 验证与后续工作

已完成的验证：

- 90 张彩色裁块的 JPEG 字节、GT 坐标、类别、属性和 split 与原版 VMA 输出对照一致。
- 15 张强度大图已生成，90 张强度裁块与对应大图区域逐像素一致。
- Dataset、GT 加载与打包、Visualizer 的 19 项相关测试通过。精简入口后，仅挂载 pickle 也能完成 Dataset 初始化；彩色和强度样例可视化与修改前一致。
- 曾对全部 90 个样本生成两种底图的 180 张 GT 可视化。`work_dirs` 属于可重新生成的运行产物，当前文件以实际执行结果为准。

后续先实现 VMA 固定点数采样、候选点序及属性编码，再进行模型迁移、checkpoint forward 对齐、SD 过拟合和 HRDX 训练。接口约定及目标设计见 [Dataset 设计](../../docs/vma/dataset_design.md)，原版实验入口见 [vma_legacy](../vma_legacy/README.md)。
