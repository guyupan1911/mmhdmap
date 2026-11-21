# VMA 调研与 RDX 数据适配讨论

整理日期：2026-11-21。

本文记录对 VMA 论文、公开代码、本地数据和 checkpoint 的检查，以及在 RDX 上训练同类模型的可行性分析。已验证事实、工程推断和待验证事项分别说明；本文不代表已经完成 RDX 数据转换、VMA 训练或精度评估。

## 1. 资料与本地路径

| 资料 | 位置 |
| --- | --- |
| VMA 论文 | <https://arxiv.org/abs/2304.09807> |
| VMA 官方仓库 | <https://github.com/hustvl/VMA> |
| VMA 本地代码 | `/home/guyu/projects/VMA` |
| 数据准备说明 | [prepare_dataset.md](../../../VMA/docs/prepare_dataset.md) |
| 训练与评估说明 | [train_eval.md](../../../VMA/docs/train_eval.md) |
| SD 下载数据 | `/home/guyu/data/vma/sd_data` |
| SD checkpoints | `/home/guyu/data/vma/sd_ckpts` |
| NYC 预训练权重 | `/home/guyu/data/vma/nyc_ckpts/seg_pretrain.pth` |
| HRDX 本地代码 | `/home/guyu/projects/HRDX` |
| RDX 本地数据 | `/data/guyu/RDX` |

本地路径反映本次讨论环境；后续代码或数据版本变化时，应重新核对实现。

## 2. VMA 的输入与网络结构

### 2.1 网络读取的是已经生成的俯视图

VMA 的单个图块模型从一张俯视图中预测地图要素的类别、矢量点序列，以及配置启用的属性。输入可以是航空影像，也可以是点云地图投影后的 BEV 栅格图。

应区分“单个模型的输入”和“整套系统的数据来源”：模型接收一张图块，不代表生成底图时只使用一帧传感器数据。SD 底图对应离线场景地图，可以由多帧、多趟采集数据拼接得到。

模型不负责把前视相机图像转换成 BEV，也不直接接收原始点云完成场景重建。

### 2.2 可以理解为 image backbone + MapTR 风格矢量解码器

更完整的结构为：

```text
BEV / 航空影像图块
        ↓
Image backbone
        ↓
ChannelMapper，多尺度特征
        ↓
Deformable Transformer encoder
        ↓
VMA vector decoder，实例与点查询
        ↓
类别、点坐标、可选属性
```

因此，“image backbone + MapTR-like vector decoder”是合理概括，但省略了 neck 和 Transformer encoder。

SD line 配置使用 ResNet-152、256 维特征、6 层 encoder 和 6 层 decoder；box 与 freespace 配置使用 7 层 decoder。不同任务的点数、实例数量和属性分支不同。

主要代码：

- [line 配置](../../../VMA/projects/configs/vma_res152_e80_line.py)
- [box 配置](../../../VMA/projects/configs/vma_res152_e80_box.py)
- [freespace 配置](../../../VMA/projects/configs/vma_icurb_backbone_e80_freespace.py)
- [VMAHead](../../../VMA/projects/mmdet3d_plugin/vma/dense_heads/vmahead.py)
- [decoder](../../../VMA/projects/mmdet3d_plugin/vma/modules/decoder.py)

## 3. 场景重建、切块与合并

### 3.1 论文系统与公开代码的边界

论文讨论的完整流程包含点云场景重建、BEV 底图生成、分块矢量提取、结果合并与人工校验。场景重建涉及动态物体处理、运动补偿、定位与建图、多趟点云对齐等。

本次检查的公开推理代码从已有 BEV 大图和图像像素坐标下的轨迹开始，没有提供从原始 LiDAR、IMU、GPS 到完整拼接地图的重建流水线。不能把“论文介绍了重建方法”理解成“仓库可以直接输入原始点云完成全部步骤”。

### 3.2 SD 切块

公开 demo 使用 6000×6000 大图，裁出 1000×1000 图块：

- 沿轨迹按弧长采样，推理入口默认每条轨迹取 15 个中心。
- 裁剪框与图像坐标轴平行，不随车辆朝向旋转。
- 中心位置限制在有效范围内，并保存图块左上角坐标。
- line 分支还启用 `TrajPadSampler` 补充覆盖；检查到的实现用步长 500 的网格候选，结合图像内容比例与已有覆盖情况筛选。
- 训练预处理也会对底图、GT 同步裁切；具体采样数与推理入口不同，不能混用。

这里的 6000、1000、阈值和采样数属于公开实现的设置，不是网络架构必须满足的固定物理范围。

代码：[inference_and_combine_function.py](../../../VMA/demo/inference_and_combine_function.py)、[utils.py](../../../VMA/demo/utils.py)、[tools/custom](../../../VMA/tools/custom)。

### 3.3 合并预测

共同步骤是把图块内坐标加上左上角偏移，恢复到大图像素坐标，再根据几何关系去重与合并。

| 类型 | 主要处理 |
| --- | --- |
| line | 距离去重，判断重叠段、端点关系和方向，拼接折线，并结合属性结果、进行简化 |
| box | 跨图块去重和重叠合并；部分类型合并后重新求最小外接旋转矩形 |
| freespace | 合并重叠区域并简化轮廓 |

line 实现中可见重采样、距离阈值、匹配点数阈值及 Douglas–Peucker 简化。这些是图像坐标处理，其中“匹配至少 30 个点”不能解释成“重叠至少 30 米”。box/freespace 部分去重代码使用边界缓冲后的几何计算重叠，不能一概理解为标准填充多边形 IoU。

这些操作主要是几何启发式后处理，不是全局地图拓扑优化，也不保证自动获得完备的道路连接关系。

代码：[line 合并](../../../VMA/demo/combine_and_save_driving_line.py)、[box 合并](../../../VMA/demo/combine_and_save_driving_box.py)、[freespace 合并](../../../VMA/demo/combine_and_save_driving_freespace.py)。

## 4. NYC 与 SD 分别提供什么

### 4.1 NYC：航空影像道路边界提取

`prepare_dataset.md` 第一部分提供 NYC 数据的下载和准备途径：

| 内容 | 用途 |
| --- | --- |
| 原始航空影像 `.jp2` | 下载后生成 TIFF 图块 |
| `labels/dense_seq` | 道路边界实例的有序点序列，作为矢量训练 GT |
| `labels/binary_map` | 二值边界图，用于评估 |
| `data_split.json` | `pretrain/train/valid/test` 划分 |
| iCurb 预训练 checkpoint | 初始化模型的图像主干 |

切块脚本将原始影像按 5×5 网格裁成 1000×1000、互不重叠的图块，只保留划分文件指定的样本。任务配置为 `map_classes = ['curb']`，即道路边界矢量提取，不涵盖 SD 中所有地图类别。

数据准备文档链接的是预训练权重；`train_eval.md` 另有训练完成的 NYC 模型下载入口。二者不能混为一谈。

使用脚本前还需修正或核对公开代码中的路径差异：文档的 `sh .tools/icurb/get_label.bash` 缺少 `/`；下载脚本的输出目录 `cropped_tiff_1` 与配置要求的 `cropped_tiff` 不一致。

### 4.2 SD：不同地图要素的训练样例与推理模型

本次讨论未找到足以确认 SD 缩写全称的依据，因此本文将其作为仓库数据集名称使用，不推断它是“标准精度地图”等特定术语。

需要区分两类样例：

- 仓库 `demo`：两张 6000×6000 图像及轨迹，主要用于推理演示，没有对应训练 GT。
- 下载的 `/home/guyu/data/vma/sd_data`：line、box、freespace 三套数据，各有 5 张 6000×6000 图像、轨迹和对应类别的 JSON 标注，可用于理解和验证训练数据流程；这不等于完整训练集。

三套目录中同名图像的文件哈希不完全一致，不应未经检查就认为它们是逐字节相同的底图。

### 4.3 为什么有三个 SD 模型

| 模型 | 类别 / 任务 | 几何表示 | 主干 |
| --- | --- | --- | --- |
| `sd_line.pth` | lane、curb、stopline；另有线型、方向等属性 | 50 个实例槽位，每实例 50 点 | ResNet-152，3 通道输入 |
| `sd_box.pth` | Crosswalk、SpeedBump、NoParkingLine、Arrow、Diamond_marking、ExclusiveLaneSign、InvertedTriangleMarking | 20 个实例槽位，每实例 5 点，表达闭合四边形 | ResNet-152，3 通道输入 |
| `sd_freespace.pth` | Diversion / diversion，导流区轮廓 | 50 个实例槽位，每实例 50 点 | iCurb backbone，4 通道输入 |

特别注意：此处 freespace checkpoint 的类别只有导流区，不能直接称为通用可行驶区域分割模型。

三个文件分别保存完整网络，并非共用一个 backbone 的三个小头。独立训练便于针对不同几何、属性和后处理设置任务配置，但不是架构上必须拆分。适配 RDX 时只需训练目标任务对应的模型。

原始数据与训练类别还有命名差异，例如 line 原始标注中的 `stop_line` 与模型类别 `stopline`，需要在转换时显式映射。

## 5. Checkpoint 实际检查结果

### 5.1 NYC 的 seg_pretrain.pth

检查文件：`/home/guyu/data/vma/nyc_ckpts/seg_pretrain.pth`。

- 共 632 个参数或缓冲区条目，全部以 `img_backbone.` 开头。
- 首层 `conv1.weight` 形状为 `(64, 4, 7, 7)`，是四通道输入。
- 除主干 `layer1`～`layer4`，还保存 `toplayer`、`latlayer*`、`smooth*`、`semantic_branch` 和单通道 `output_layer` 等参数。
- 当前 VMA 的 `iCurb_BackBone` 只定义主干部分，额外分割层不会在该 backbone 中使用。
- 没有 VMA Transformer、vector decoder 和矢量预测头参数，不能单独完成 VMA 道路边界折线推理。

因此，“它是 backbone 预训练权重”描述的是在 VMA 中的用途；文件本身还保留了原分割网络的部分参数。其用途符合道路边界分割预训练，但文件没有类别和完整训练历史，不能只凭参数证明原始训练过程。

实现：[icurb_backbone.py](../../../VMA/projects/mmdet3d_plugin/vma/backbones/icurb_backbone.py)。

### 5.2 SD 文件为何显著更大

以下单位均为 MiB；模型列统计权重与缓冲区的张量存储，文件还有少量序列化开销和元数据。

| 文件 | 文件总大小 | 模型张量 | 优化器状态 |
| --- | ---: | ---: | ---: |
| NYC `seg_pretrain.pth` | 176.4 | 176.2 | 无 |
| SD `sd_line.pth` | 906.8 | 303.2 | 602.4 |
| SD `sd_box.pth` | 884.5 | 295.9 | 587.7 |
| SD `sd_freespace.pth` | 685.8 | 228.6 | 456.4 |

SD 文件顶层均包含 `meta`、`state_dict` 和 `optimizer`，元数据记录训练到 epoch 80。其体积差异主要来自：

1. SD 保存优化器状态，约占文件三分之二；NYC 预训练文件没有。
2. SD 保存完整 VMA，包含主干、neck、Transformer 和预测分支。
3. line/box 的 ResNet-152 主干约 222.4 MiB，freespace 的 iCurb 主干约 162.3 MiB；其余网络与属性分支也不同。

只比较模型张量，SD line、box、freespace 分别约为 NYC 预训练文件的 1.72、1.68、1.30 倍。这里比较的是完整 SD 模型与 NYC 预训练网络，不是完整 SD 与完整 NYC 模型。

Checkpoint 内嵌配置还保留了与当前公开代码不同的旧类名和配置项。参数检查验证了文件结构，但尚未实际运行完整加载、推理来验证所有兼容性。

### 5.3 优化器状态是什么

模型权重决定前向预测结果；优化器状态记录训练过程中用于下一次参数更新的历史信息。

SD 使用 AdamW，其状态包含：

- `exp_avg`：梯度的指数移动平均。
- `exp_avg_sq`：梯度平方的指数移动平均。
- `step`：更新步数。

前两项通常各自与对应的可训练参数同样大，所以模型权重约 300 MiB 时，优化器状态可能再占约 600 MiB。它们用于恢复训练；只加载模型权重仍可继续训练，但更新历史会重新初始化。

推理不更新参数，因此不需要优化器状态。保留相同权重、移除优化器状态不会改变模型预测，SD 推理文件可以缩小到接近表中的模型张量大小。本次没有修改或精简原 checkpoint。

## 6. SD 底图、强度与 Jet 伪彩色

### 6.1 已观察到的证据与结论边界

VMA 的 `demo/image_data/sample_1.jpg` 呈现深蓝背景、青色道路和较亮的黄色标记，颜色分布与 Jet 类色表接近。平坦区域的像素更贴近该色表，JPEG 压缩和色度抽样会引入偏差；代码中也存在深蓝填充值和强度图相关命名。

工程上可以采用“强度底图经某种颜色映射产生三通道图像”的工作假设。但公开代码没有给出原始底图生成器，尚不能确认：

- 是否严格使用 OpenCV JET 或 Matplotlib jet，以及其具体版本与离散化方式。
- 原始标量、强度归一化和截断规则。
- 每格使用最大值、均值、分位数还是某个代表点。
- 地面提取、高度过滤、动态物体过滤和多帧融合规则。

视觉相似性可以支持假设，不能证明完整生成链路相同。

### 6.2 本次生成的对照图

| 实验 | 处理方式 | 输出目录 |
| --- | --- | --- |
| 单帧 PCD，时间戳 `1789535788800123692` | 使用原始 intensity；范围 x∈[0,100] m、y∈[-40,40] m，分辨率 0.05 m/px；每格最大强度；0～255 直接映射 OpenCV JET；未做高度过滤 | `/home/guyu/data/bev_previews/1789535788800123692` |
| 拼接后的 `origin_ibev.jpg` | 原图为 12800×12800、单通道 uint8；直接逐像素应用 JET，无额外强度拉伸 | `/home/guyu/data/bev_previews/stitched_1789535788400201321` |

单帧结果包括 `bev_intensity_jet.png`、带坐标说明图、有效掩码、栅格数据、脚本与 metadata。单帧底图稀疏，不能将它与多帧拼接底图的稠密程度直接比较。

拼接结果包括全尺寸 `origin_ibev_jet.png`、裁剪图、预览图、脚本与 metadata。图像偏蓝是因为大部分原始强度较低，而不是颜色映射失败。完整输入路径保存在各目录的 `metadata.json` 中。

### 6.3 单通道和伪彩色都能用于训练

在 VMA 架构下，二者都可以作为输入：

- 单通道强度图：修改 backbone 首层为单通道，或复制成三通道。
- 三通道伪彩色图：直接适配三通道 backbone。

固定颜色映射没有增加新的观测信息，量化或截断还可能损失信息。使用三通道便于衔接常见预训练图像主干，但这只是合理动机，不能据此断言作者的唯一目的。

“都能通过训练学习”不代表收敛速度或最终精度完全相同，也不代表一个训练好的模型可以随意更换输入表达。训练与推理必须采用一致的生成和归一化流程。

## 7. RDX 点云 RGB 字段的检查

检查样本：

```text
/data/guyu/RDX/pointclouds/2025_05_01_12_54_30/300_1746129303_620.pcd
```

### 7.1 文件实际保存了什么

- 111246 个点，PCD `DATA binary`。
- 字段为 `x y z rgb`，没有 `intensity`。
- `rgb` 以 float32 字段承载打包颜色位，不是一个可直接使用的浮点强度。

解包应先按位重解释为 uint32，再提取颜色：

```python
packed = rgb_float.view(np.uint32)
r = (packed >> 16) & 255
g = (packed >> 8) & 255
b = packed & 255
```

该样本观察到 178 种颜色；与离散 Matplotlib jet 色表取整结果高度接近，约 84.8% 的点颜色完全匹配，所有点颜色与最近色表项的 RGB 欧氏距离不超过 1。它强烈支持“颜色来自 Jet 类伪彩色映射”，但不能证明被映射的原始量必然是 intensity，也不能外推为整个 RDX 数据集已经验证。

### 7.2 当前代码没有给出 RGB 的生成来源

检查到的 HRDX 数据转换、Dataset 和点云加载代码读取已有 PCD；未找到生成该 PCD RGB 字段的上游导出代码。可视化中另有 TURBO 着色，但它用于渲染输出，不能作为 PCD 颜色来源的证据。

因此，仍需上游导出器或数据说明才能确认原始标量、色表和归一化方式。

### 7.3 现有加载代码的语义问题

[loading_lidar.py](../../../HRDX/plugin/datasets/pipelines/loading_lidar.py) 中将打包 RGB 直接转换为数值：

```python
col = col.view(np.uint32).astype(np.float32)
```

注释提到低字节代理，但实际实现没有进行低字节掩码，也没有反查色表。后续若将第四列作为 intensity，则得到的是打包颜色整数，不是真实强度。

本次仅指出此问题，没有修改 HRDX 实现。新的 BEV 转换器应显式选择“解包 RGB”或“近似反查色表索引”，不能把打包整数直接作为强度聚合。

## 8. 在 RDX 上训练 VMA 的可行性

### 8.1 结论与前提

如果 RDX 提供点云、准确且同步的定位轨迹，以及与其一致的人工地图 GT，就具备训练同类 VMA 的基本条件。核心工作是构建对齐的 BEV 图像与矢量标注，不是要求输入一定使用 Jet。

准确轨迹可用于直接拼接，不一定需要重新实现整套 SLAM；是否需要配准优化，应根据实际标线重影检查决定。位姿用于离线预处理，通常不作为 VMA 网络输入。

```text
按地理区域划分 train / val / test
              ↓
点云、位姿、GT 坐标和时间核对
              ↓
拼接、过滤、固定规则栅格化
              ↓
统一米到像素映射，生成底图与 GT
              ↓
同步裁块、裁 GT、重采样、类别和属性转换
              ↓
VMA 训练与图块评估
              ↓
大图合并、米制评估与可视化检查
```

### 8.2 RGB 点云的两种底图生成方式

| 方式 | 做法 | 限制 |
| --- | --- | --- |
| 直接保留颜色 | 解包 RGB，为每格选择代表点颜色 | 直接平均 RGB 可能产生偏离原色表的颜色；需固定点选择规则 |
| 反查近似标量 | 找到最近 Jet 色表索引，先聚合标量，再统一着色 | 恢复的是色表索引，不保证恢复物理强度或原始动态范围 |

即使无法恢复原始 intensity，只要图像表达稳定、包含足够的道路几何信息，也可以训练。需要抽查跨帧、跨场景是否使用同一色表和强度尺度。

### 8.3 数据适配的关键事项

1. **坐标系与时间同步**：核对 PCD 是传感器、车体还是其他坐标系；避免重复施加外参。HRDX Dataset 的部分 `ego2global` 字段实际写入 `lidar2g`，不能只根据名称判断。
2. **底图质量**：检查扫描运动畸变、动态车辆、路面与高架层混叠、地面过滤和多帧拼接重影。准确轨迹不自动解决所有问题。
3. **尺度统一**：明确米/像素、原点、轴方向和裁剪偏移；点云、GT、轨迹共用同一转换。像素阈值必须随分辨率换算。
4. **标注语义**：类别、折线/多边形表示、实例分段、闭合规则和属性需显式映射。缺失标注区域不能简单作为负样本。
5. **训练任务**：如果只需要车道线、道路边界，先训练 line 模型，不必同时建立三个模型。没有标注的属性分支应移除或正确忽略。
6. **权重适配**：可尝试 SD checkpoint 初始化，但需核对类别头、属性头、输入通道和代码兼容性；它不能保证跨数据域直接推理有效。
7. **数据划分**：先按地理区域或路线划分，再拼接切块；避免同一路段、重复采集或重叠图块跨训练与验证集泄漏。
8. **评估目标**：这是离线 BEV 矢量提取。若底图融合了未来帧，不能把其结果当作严格在线单帧感知结果比较。

### 8.4 建议的第一阶段验证

先选择一个标注完整的路段，用固定规则生成拼接 BEV，并叠加 GT，确认标线清楚、无明显重影且坐标对齐。随后用少量图块验证训练是否能拟合，再扩展到按地理区域隔离的训练与验证集。

需要比较单通道与 Jet 时，应固定数据划分、栅格化、模型主体和评估规则，单独改变输入表达及必要的首层适配。现阶段尚无实验依据断言其中一种一定更好。

## 9. 当前结论与未完成事项

已完成的工作是论文与代码调研、样例与 checkpoint 结构检查、点云颜色分析，以及两组 Jet BEV 对照图生成。

尚未完成 RDX 多帧重建与 GT 叠加验证、全数据颜色一致性检查、VMA 数据转换器、checkpoint 完整加载验证或 RDX 训练评估。SD 原始着色流程和 RDX RGB 的上游来源也仍待确认。

目前可以确认的工程方向是：沿用 VMA 的图像到矢量结构，在 RDX 上构建统一且准确对齐的 BEV 图像—GT 数据。强度单通道与伪彩色三通道都是可选输入表达；底图质量、GT 对齐与数据一致性比是否使用某种色表更基础。
