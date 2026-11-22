# BEV 图像矢量地图数据集设计

状态：已实现 SD 三类图像与 GT 联合裁剪、train/val infos、`BevMapDataset`、单图加载和打包，见 [SD converter](sd_converter.md) 和 [数据读取检查](../../projects/vma/README.md)。原始 GT 已进入 infos，并完成加载、DataSample 打包与可视化；模型目标生成尚未实现。

## 1. 范围与职责

一个样本是一张固定尺寸的 BEV 图像及其图像坐标系下的矢量 GT。不同数据源分别实现 converter，训练和推理共用 `BevMapDataset(BaseMapDataset)`，不为 SD、HRDX 分别实现 Dataset 子类，也不以模型名命名 Dataset。

当前 converter 支持 SD line、box、freespace 的边界矢量实例，随后接入 HRDX 点云底图和航拍图。模型专用目标由 transforms 生成；box、freespace 的完整区域语义和孔洞表示不在当前范围内。

```text
SD 图像和标注 ── SD converter ────┐
                                 ├─ 固定尺寸图像 + 统一 infos
HRDX 图像和标注 ─ HRDX converter ─┘
                                            ↓
                                  BevMapDataset
                                            ↓
                       加载图像/GT → 同步增强 → VMA 目标 → 打包
                                            ↓
                         图像 tensor + MapDataSample
                                            ↓
                             data preprocessor → 模型
```

| 层 | 职责 |
| --- | --- |
| Converter / 数据准备 | 生成或引用固定尺寸图像，转换和裁剪 GT，划分数据，写统一 infos |
| Dataset | 读取 infos，检查协议和样本字段，解析路径，调用 pipeline |
| 通用 transforms | 加载图像和原始矢量标注，同步变换图像与 GT，打包 |
| VMA transform | 采样固定点数、生成匹配点序、编码属性和归一化坐标 |
| Data preprocessor | 批处理、类型转换、图像归一化和设备搬运 |
| Metric / 大图推理工具 | 分别负责评估，以及窗口生成、坐标还原与预测合并 |

Dataset 不读取点云或轨迹，不执行重建、切块、补充采样和预测合并。数据准备可以调用独立的重建/栅格化工具，不要求所有操作都放进一个 converter 函数。

## 2. 文件组织与统一格式

建议输出结构：

```text
prepared_data/
├── images/
│   ├── sample_1_0.jpg
│   └── ...
├── bev_map_infos_train.pkl
├── bev_map_infos_val.pkl
└── bev_map_infos_test.pkl       # 有测试集时才生成
```

沿用 MMEngine 的 `metainfo + data_list` 格式，默认写 PKL；内容限定为可序列化的基础类型，不存 Shapely、Tensor 或 DataContainer。以下是结构示例，折线仅用两个顶点演示，不是完整 SD 样本：

```python
infos = dict(
    metainfo=dict(
        dataset_type='bev_image_map',
        info_version='1.0',
        dataset_name='sd_line',
        classes=['lane', 'curb', 'stopline'],
        image_size=[1000, 1000],  # [H, W]，同一份 infos 固定
        coordinate_system='image_pixel',
        attribute_schema={
            'lane_direction': dict(
                values=['unidirectional', 'bidirectional', 'unknown'],
                applies_to=['lane'])
        }),
    data_list=[
        dict(
            sample_id='sd_line/sample_1_0',
            img_path='images/sample_1_0.jpg',
            height=1000,
            width=1000,
            source_id='sd_line/sample_1',
            crop_origin=[2415, 0],  # [left, top]，可选
            instances=[
                dict(
                    label=0,
                    points=[[512.127, 0.0], [513.068, 64.88]],
                    attrs={'lane_direction': 'unidirectional'})
            ])])
```

字段约定：

| 字段 | 约定 |
| --- | --- |
| `dataset_type / info_version` | 协议标识和版本，不能用 nuScenes infos 冒充此协议 |
| `classes` | 有序类别表，`label` 是其从 0 开始的下标；泛用 Dataset 不内置 SD 类别 |
| `image_size` | 该份数据的固定 `[H, W]`；首版 SD 为 `[1000, 1000]`，不是类中写死的常量 |
| `attribute_schema` | 可为空；属性词表及适用类别。完整 SD converter 从原配置写出所有监督属性的词表 |
| `sample_id` | 稳定且唯一的样本标识，不依赖 DataLoader 顺序 |
| `img_path` | 相对图像根目录的路径；默认以 `data_root` 为根，允许通过 `data_prefix.img_path` 调整 |
| `height / width` | 实际保存图像的尺寸，必须与 `image_size` 一致 |
| `instances` | 训练/评估必需；空列表表示已标注且没有目标，缺少字段表示没有 GT，二者不能混用 |
| `label / points` | 实例类别及原始有序二维顶点，点数可变 |
| `attrs` | 属性名到字符串值的映射，无属性时为空字典；保留原始有用属性 |
| `source_id / crop_origin` | 可选追溯信息；SD converter 必须保留，供来源检查和大图坐标还原使用 |

`dataset_name` 仅用于记录来源，Dataset 不按其值分支执行源数据逻辑。配置覆盖 `classes`、属性词表或顺序时必须检查一致性；若需要重映射，应显式转换标签，不能只替换元信息。

SD 配对强度图通过可选 `intensity_img_path` 提供，独立的 `data_prefix.intensity_img_path` 默认指向 `data_root`。`has_intensity=True` 时每帧必须提供该路径字段，但它只表示存在配对底图，不要求模型使用强度图。当前 pipeline 只加载彩色图，输出单个 `inputs: Tensor[3,H,W]`；强度路径作为 DataSample 元信息保留。可视化工具需要时才读取强度底图，Visualizer 检查其与 DataSample 的尺寸并绘制。后续有几何增强时，可视化底图必须与 GT 同步变换；若切换模型输入，应通过配置选择图像来源并保持单图接口。

图像文件和 infos 可以放在不同位置，路径解析不以 PKL 所在目录为隐式基准。首版支持 SD 的三通道图像；航拍 RGB 使用相同接口。单通道输入可后续通过明确的加载和模型通道配置支持，不在 Dataset 中自动着色。

## 3. 坐标、几何与属性

### 图像坐标

- 原点为小图左上角，x 向右，y 向下，单位为像素，允许亚像素坐标。
- 第一版按像素中心约定，点坐标范围为 `0 <= x <= W-1`、`0 <= y <= H-1`。Converter 对边界做显式裁剪，不让 Dataset 静默修正异常 GT。
- `points` 形状为 `[K, 2]`，至少两个顶点，坐标有限，折线长度大于零。
- 保留裁剪规则产生的点序，不额外排序或按固定点数重采样；SD box 的 Arrow 沿用原版点序检查与反向逻辑。
- SD box/freespace 轮廓保留首尾重复点；SD line 闭合线沿用原版 buffer 和去掉重复闭合点的处理。这里保存边界矢量，不解释为带孔洞的填充区域。
- infos 保存未归一化坐标。VMA 的 `x/W`、`y/H` 和截断到 `0.999` 属于目标生成规则，不是原始数据协议。

`crop_origin` 只适用于未旋转、未缩放的平移裁块：大图坐标等于小图坐标加该偏移。转换器若旋转、缩放或重投影，应另保存明确的变换及坐标参考说明，不能只用该字段假装可以恢复。没有地理信息时不构造虚假的米制坐标或经纬度。

### 属性和缺失标注

- 已标注的 `unknown` 是一个真实类别，不等于缺失标注。
- 适用的属性缺失或为 `None` 表示未标注，后续训练应忽略或关闭相应分支。
- 属性不适用于该实例类别，由 `attribute_schema.applies_to` 判断。
- 原版 SD 的六个监督分支依次为 `lane_direction`、`lane_type`、`lane_properties`、`lane_flag`、`lane_width`、`curb_type`。训练目标配置显式指定顺序，不依赖字典插入顺序。
- SD 原始 `lane_color` 可以保留在 `attrs` 中，但不自动增加预测分支。
- 兼容原版 SD 时，不适用属性编码为对应词表长度，按原版 loss 的语义处理。不能将 HRDX 中缺失的适用属性直接当作原版已有监督标签；接入时需实现明确的 ignore 逻辑。

## 4. Dataset 与 pipeline 接口

类名为 `BevMapDataset`，继承现有 `BaseMapDataset`，默认 `modality=dict(use_bev_image=True)`。现有基类已允许新增 modality，不改变已有 nuScenes 默认输入。

Dataset 复用 MMEngine 的加载、序列化、子集和 pipeline 机制，仅增加本协议的元信息校验、样本校验和路径解析。初始化不解码全部图像；文件解码由 MMCV transform 完成，实际图像尺寸由 MMCV 加载器报告，converter 负责输出尺寸的一致性；`test.py` 只负责初始化 Dataset 和调用 Visualizer。

完整训练 pipeline 的目标接口如下。其中 `LoadVectorMapAnnotations` 已实现，`GenerateVMATargets` 尚未实现；当前配置包含图像/GT 加载和打包，见 [sd_images.py](../../projects/vma/configs/sd_images.py)：

```python
dataset = dict(
    type='BevMapDataset',
    data_root='/path/to/prepared_data',
    ann_file='bev_map_infos_train.pkl',
    modality=dict(use_bev_image=True),
    pipeline=[
        dict(type='mmcv.LoadImageFromFile', to_float32=False,
             imdecode_backend='pillow',
             color_type='color_ignore_orientation'),
        dict(type='LoadVectorMapAnnotations'),
        # 如需几何增强，在这里同步变换图像和原始折线。
        dict(type='GenerateVMATargets', profile='sd_line_legacy', num_points=50),
        dict(type='PackBEVMapInputs'),
    ])
```

`profile='sd_line_legacy'` 明确选择旧版点序和属性编码；其他数据源不能未经检查沿用此规则。目标生成使用变换后的实际图像尺寸。第一阶段不添加几何增强，完整复现旧版输入。

打包输出采用单图像检测接口，当前字段与后续训练字段区分如下：

```text
inputs: Tensor[C, H, W]
data_samples: MapDataSample
  metainfo:
    sample_id, img_path, ori_shape, img_shape
    可选 source_id, crop_origin, intensity_img_path
  gt_instances: InstanceData             # 无标注推理时不设置
    labels: LongTensor[N]
    polylines: list[float32 ndarray[Ki, 2]]  # 已实现
    attributes: list[dict]                  # 已实现，原始属性
    attrs: LongTensor[N, A]                 # 待实现，训练属性编码
    shifted_points: FloatTensor[N, S, P, 2] # 待实现，匹配候选点序
```

SD line 兼容模式使用 `A=6, P=50, S=49`，无效候选沿用旧版 `-10000` 填充。`attrs` 的实例维在首维，与旧版 `[A, N]` 不同，迁移 loss 时需显式适配。`S` 表示补齐后的候选容量，并不是每个实例都有 49 种有效点序。

原始折线应保留供评估和检查使用，不能仅保存匹配候选再猜测原始几何；当前已放入 `gt_instances.polylines`，为长度 N 的 float32 变长数组列表；原始属性字典列表保存在 `gt_instances.attributes`，训练编码后的属性另用 `attrs`。允许空 GT，各字段应有正确的零实例形状；SD 兼容预处理保留原版跳过空窗口的行为。

现有 `MapDataSample` / `InstanceData` 可复用。`PackMapInputs` 是多相机接口（`inputs['imgs']`）；已新增 `PackBEVMapInputs`，并更新 `MapDataSample` 文档说明两种输入形式。当前单图打包输出图像、元信息以及前置 transform 提供的 GT。无标注推理配置省略 `LoadVectorMapAnnotations`；标注为空时生成零实例 GT，缺少 `instances` 时不伪造空 GT。后续单图数据预处理将输入整理成 `[B, C, H, W]`；图像归一化仅执行一次。

无标注推理 pipeline 只加载图像并打包。验证 pipeline 保留 GT；`test_mode=True` 不代表必须丢掉标注。仅检查 backbone/neck/encoder 时也可以先使用不生成矢量目标的 pipeline。

当前复用 MMCV 的图像加载器，不再实现单独的 BEV 加载类。图像加载和打包保留 BGR；可视化前转换为 RGB，后续模型 data preprocessor 负责模型输入的 BGR→RGB 和归一化。

## 5. SD converter

`projects/vma/tools/sd_converter.py` 通过 `tools/create_data.py sd` 调度，从 SD 根目录统一读取 line、box、freespace 的原图、轨迹和大图标注，一次生成小图与 GT。各类采样数分别为 7、7、14，过滤无有效 GT 的窗口，按来源大图划分 train/val，详见 [使用说明](sd_converter.md)。

```text
原图 + 轨迹 + {type}_data_6k.json
                ↓ 图像/GT 联合裁剪、类别与属性转换
prepared/{type}/images/ + sd_map_infos_train.pkl + sd_map_infos_val.pkl
```

映射为：`image_name → sample_id/img_path`、`class → label`、`data → points`、`left_top → crop_origin`；属性规范化为字典，保留 `source_id` 和大图内的 `source_instance_id`。line 的 train/val 为 35/7 张，box 为 22/5 张，freespace 为 18/3 张，已与原版实际输出对照。图像及原始 GT 可读取；训练目标的固定点数采样和属性编码留在后续 transform 中。

HRDX converter 随后产生同样协议，可以分别准备点云 BEV 和航拍图的数据版本。类别、属性和图像表达允许不同，但转换后都必须满足相同结构与像素坐标约定；不同图像输入的 GT 必须分别正确对齐。训练/验证区域划分需在切块前确定，重叠路段及其不同输入表达保持相同划分。

## 6. 实现顺序与验收

1. **数据接口**：实现统一 infos 校验、`BevMapDataset`、SD 已裁剪数据转换；验证样本数量、路径、尺寸、标签、属性和 split，检查错误类别、越界点、空 GT 与无 GT 的区别。
2. **图像通路**：实现单图加载和打包，比较同一 JPEG 的 RGB 解码、归一化及最终输入 tensor。严格对齐时采用与 legacy Dataset 相同的解码后端；不能仅凭肉眼相似判断一致。
3. **训练目标**：迁移原版采样、点序和属性编码，比较相同实例的逐项输出；测试开放 lane、非 lane、闭合线和不足候选的填充。旧代码对所有开放 lane 只保留原始点序，不能擅自改成正反序等价。
4. **模型联调**：先对齐 encoder forward，再接 vector head、loss 和单图过拟合。尚未完成模型迁移前，Dataset 能加载不能视为训练链路已经验证。

原版闭合候选超过上限时会随机采样。将目标生成从 loss 移到数据 worker 会改变随机数消费位置；对照需固定候选或控制随机源，不能把随机差异误判成实现错误。

设计依据：现有 [BaseMapDataset](../../mmhdmap/datasets/base_map_dataset.py)、[MapDataSample](../../mmhdmap/structures/map_data_sample.py)、[PackMapInputs](../../mmhdmap/datasets/transforms/formatting.py)，以及原版 [SD line Dataset](../../../VMA/projects/mmdet3d_plugin/datasets/sd_driving_line_dataset.py)、[配置](../../../VMA/projects/configs/vma_res152_e80_line.py) 和 [切块脚本](../../../VMA/tools/custom/generate_line_final_from_origin.py)。
