# mmhdmap 框架设计

本文记录 mmhdmap 作为独立 HD Map 感知任务库的框架设计建议。目标是基于 OpenMMLab 2.0 的接口和运行机制，为地图分割、矢量地图及后续时序任务提供统一的数据、模型和评估接口。

本文是设计方案，不代表相关模块已经实现。首个落地目标是迁移 HRDX 的 nuScenes stage 1 BEV 分割流程，并验证数据与计算结果的一致性。

## 1. 当前基础与设计原则

当前仓库已有以下基础：

- `mmhdmap/registry.py` 定义了 `MODELS`、`DATASETS`、`TRANSFORMS`、`METRICS`、`TASK_UTILS`、`VISUALIZERS` 等注册表。
- `mmhdmap/structures/det_data_sample.py` 中的 `DetDataSample` 目前是一个空的 `BaseDataElement` 派生类。
- 模型和数据处理实现主要集中在 `projects/` 中，公共库接口尚需整理。

框架建设遵循以下原则：

1. 定义地图任务的数据语义，再实现对应的数据结构和模块。
2. 复用 MMEngine 的 Runner、模型基类、评估与数据容器能力。
3. 使用 mmhdmap 自己的注册表和默认命名空间组织组件。
4. 先跑通 stage 1 的完整训练和验证流程，再提取经过验证的公共能力。
5. 按实际复用需求增加抽象，避免提前设计所有任务的复杂继承体系。

## 2. 统一地图任务的数据约定

数据结构只能约束部分类型和形状，不能自动保证坐标系、单位和类别语义一致。Dataset、transform、模型、评估器和可视化器必须共同遵循明确的数据约定。

| 约定 | 需要明确的内容 |
| --- | --- |
| 坐标系 | 地图几何使用 ego、LiDAR 还是 global 坐标，各轴方向是什么 |
| 单位 | 米、像素、归一化坐标分别在哪些阶段使用 |
| BEV 范围 | `roi_size` 的顺序、原点位置、栅格行列与物理坐标的对应关系 |
| 类别 | 类别名称、编号、背景和忽略区域的表达方式 |
| 栅格标注 | 使用 `[C, H, W]` 二值通道还是 `[1, H, W]` 类别编号 |
| 矢量标注 | 点序列、闭合线、变长实例、等价排列的表达方式 |
| 时序 | 历史帧顺序、场景边界、位姿变换方向 |

HRDX 当前存在模型内翻转分割 GT、部分投影字段名称与实际参考系不完全对应的情况。迁移时应先记录并保持原有语义，验证一致后再统一命名和处理位置。

具体坐标约定和字段名称需要在实现前确定，不能仅依赖变量名称推断。

## 3. MapDataSample 与内部容器

建议定义直接继承 `BaseDataElement` 的 `MapDataSample`，组织单个地图样本的元信息、GT 和预测结果。

```text
MapDataSample
├── metainfo
│   ├── sample_idx、token、scene_id、timestamp
│   ├── 标定和位姿信息
│   └── 地图坐标系、BEV 范围和栅格尺寸
├── gt_sem_seg：PixelData
├── pred_sem_seg：PixelData
├── gt_instances：InstanceData
└── pred_instances：InstanceData
```

上述结构是建议接口，字段按任务需要赋值，不要求所有字段同时存在。

- `PixelData` 保存 BEV 栅格 GT、预测 mask 或 logits。
- `InstanceData` 保存地图实例的 `labels`、`vectors`、`scores` 等关联字段。
- 图像、点云等模型输入通常放在 `inputs` 中。
- 中间特征通常属于模型内部计算，不需要放入 DataSample。

对于 stage 1，先实现分割相关字段即可。后续矢量地图或跟踪任务再增加对应字段及约束。

不应因为地图任务使用三维坐标，就将 BEV mask 存入 `PointData`。BEV mask 是规则二维网格，适合使用 `PixelData`；`PointData` 面向按点索引的数据。

仅增加 `vectors` 等字段通常不需要新的底层容器。只有出现稳定的特殊校验、索引或拼接需求时，才扩展已有容器或定义新容器。

## 4. 注册体系与模块发现

通过 mmhdmap 的注册表构建需要配置化的组件：

```python
from mmhdmap.registry import DATASETS, MODELS, TRANSFORMS
```

配置统一使用：

```python
default_scope = 'mmhdmap'
```

需要提供明确的模块发现方式，例如 `register_all_modules()`，或通过注册表的 `locations` 自动导入。注册表中声明的模块路径必须与实际目录和导入结构一致。

注册对象主要包括模型、Dataset、transform、Metric 和可视化器等需要通过配置创建的组件。普通几何函数和数学工具直接导入即可，不需要全部注册。

`MapDataSample` 通常由打包 transform 直接实例化，不必专门注册。已有的外部组件可以复用，但其所属 scope 和导入方式需要明确。

## 5. 数据加载与批处理职责

| 模块 | 主要职责 |
| --- | --- |
| Dataset | 读取索引和标注，组织场景与样本信息 |
| 地图读取适配器 | 对接 nuScenes、AV2 等地图格式，提取局部几何 |
| Transforms | 加载数据、增强、矢量化、栅格化 |
| `PackMapInputs` | 输出 `inputs + MapDataSample` |
| `MapDataPreprocessor` | 搬移设备、组织 batch，以及按约定完成归一化和 padding |

建议 Dataset 基于 MMEngine `BaseDataset` 复用通用能力；有特殊取样需求时再扩展对应方法。Pipeline 中的 transform 建议继承 MMCV `BaseTransform`，实现 `transform()`。该接口不限定处理对象必须是图像，地图几何和标签处理同样适用。

单样本打包后的建议结构：

```python
{
    'inputs': {
        'imgs': image_tensor,
    },
    'data_samples': map_data_sample,
}
```

图像归一化、padding 等操作应明确由 transform 或 data preprocessor 中的一方负责，避免重复处理。采用上述结构后，不再依赖旧 `DataContainer` 的批处理语义。

不同 Dataset 应尽量输出一致的内部表达，让通用模型无需根据数据集名称切换逻辑。

## 6. 模型接口与组件划分

建议模型基于 MMEngine `BaseModel`，遵循统一入口：

```python
forward(inputs, data_samples=None, mode='tensor')
```

| 模式 | 返回内容 |
| --- | --- |
| `loss` | 训练损失字典 |
| `predict` | 包含预测结果的 `MapDataSample` 列表 |
| `tensor` | 原始 Tensor 或约定的中间计算结果 |

训练复用标准调用流程：

```text
Runner / TrainLoop
    → DataLoader
    → model.train_step()
    → DataPreprocessor
    → forward(mode='loss')
    → loss 汇总与 OptimWrapper 参数更新
```

使用默认损失解析逻辑时，应遵守其损失键名规则，让需要参与反向传播的损失被正确汇总。

模型内部可以按实际需要拆分：

```text
图像 backbone / neck
    ↓
BEV encoder
    ↓
可选的时序融合或多模态融合
    ↓
地图预测 head
```

BEVFormer 是 BEV 特征提取的一种实现。公共接口应允许替换其他 BEV encoder，避免将 BEVFormer 特有参数变成所有地图模型的必选参数。

## 7. 时序数据与状态管理

时序能力影响 Dataset、Sampler、模型和评估，应明确以下问题：

- 一次取样是单帧，还是包含历史帧的片段？
- 历史帧按什么顺序排列，历史不足时如何处理？
- 训练时如何采样历史，测试时是否要求严格时间顺序？
- 流式模型在何时清空状态，如何识别场景切换？
- 分布式测试如何保持场景内的连续性？

当前迁移先保留显式传入当前帧和历史帧的方式，验证 HRDX 原有行为。流式推理状态管理后续作为独立能力设计，不必在 stage 1 同时引入。

## 8. 评估、可视化与结果导出

评估和可视化应从 Dataset 中独立出来：

- 基于 `BaseMetric` 实现地图指标，消费模型输出的 `MapDataSample`。
- 提供地图可视化器，展示 BEV GT、预测、矢量实例以及相机投影。
- 按需要提供结果导出工具，适配不同数据集要求的提交格式。

指标需明确阈值、有效区域、坐标单位和类别映射。将旧 `evaluate()` 移入 Metric 文件只是代码迁移的一部分，还需要核对评估语义。

Stage 1 先支持现有 BEV 分割评估，验证与 HRDX 一致后，再引入矢量地图和跟踪指标。

## 9. 公共库与实验项目边界

建议目录组织如下，具体模块随实现逐步增加：

```text
mmhdmap/
├── registry.py
├── structures/
├── datasets/
│   ├── transforms/
│   └── samplers/
├── models/
│   ├── data_preprocessors/
│   ├── backbones/
│   ├── necks/
│   ├── bev_encoders/
│   ├── heads/
│   ├── losses/
│   └── task_modules/
├── evaluation/
├── visualization/
└── utils/

projects/
├── maptracker/
└── mask2map/

configs/
tools/
tests/
docs/
```

公共库保存稳定、可跨模型复用的能力；`projects/` 保留实验模型特有的结构和研究代码。注册表的 `locations` 应随最终目录调整，例如 Metric 实现放入 `evaluation/` 后，应更新对应导入位置。

训练和测试入口尽量薄，使用统一配置、模块注册和标准 Runner，不将某个模型的特殊训练逻辑写进公共入口。

## 10. 首阶段落地与验收

建议按以下顺序推进：

1. 明确 stage 1 的坐标、类别、GT 和时序数据约定。
2. 实现 mmhdmap scope、必要模块注册、`MapDataSample` 和 `PackMapInputs`。
3. 迁移 Dataset、地图处理 transforms 和 `MapDataPreprocessor`，对齐样本输入与 GT。
4. 将 stage 1 模型接入 `BaseModel` 接口，对齐权重加载、预测和损失。
5. 实现分割 Metric 与必要可视化，使用标准 Runner 跑通训练和验证。
6. 根据验证结果提取公共组件，再扩展矢量地图或其他模型。

验收应覆盖：

- 样本索引、历史帧选择和场景信息一致。
- 图像预处理、相机投影、地图几何和分割 GT 一致。
- 相同输入及权重下，模型输出和损失在约定容差内一致。
- 反向传播、参数更新和小样本过拟合符合预期。
- 评估指标与原实现采用相同语义，结果可以对应比较。

首阶段的目标是：通过 mmhdmap 自己的注册表、DataSample、打包 transform、data preprocessor 和 Metric，使用标准 Runner 完成 stage 1 训练与验证，建立经过实际验证的公共接口。
