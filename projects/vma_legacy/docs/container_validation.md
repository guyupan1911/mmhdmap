# VMA 镜像与 SD line 验证记录

验证日期：2026-11-21。

结论：已创建 `vma-legacy:blackwell-cu128` 镜像，验证了 SD line checkpoint 的严格加载、42 个样例图块的推理，以及从该 checkpoint 微调的单图块过拟合。这里验证的是旧版 VMA 的图像任务计算链路，尚未进行 OpenMMLab 2.0 模型迁移。

## 1. 本机与环境选择

宿主机为 NVIDIA RTX PRO 5000 72GB Blackwell，计算能力 12.0，驱动 580.126.09；`nvcc` 为 CUDA 13.0。VMA 安装文档中的 PyTorch 1.9.1 / CUDA 11.1 不能直接作为这代 GPU 的运行环境。

容器选择 PyTorch 2.7.0 / CUDA 12.8，保持 MMCV 1.4.0、MMDetection 2.14.0、MMSegmentation 0.14.1 的旧接口。Python 为 3.11。容器 CUDA 不要求与宿主机安装的 toolkit 完全相同；这里已经实际运行 GPU 前向和反向计算。

参考：[PyTorch 2.7 的 Blackwell 支持说明](https://pytorch.org/blog/pytorch-2-7/)、[NVIDIA GPU 计算能力表](https://developer.nvidia.com/cuda/gpus)。

兼容调整包括：

- 复用本地 `mmhdmap-dev:latest` 中已经为 sm_120 编译的 MMCV 2.1 CUDA 扩展，配合 MMCV 1.4 Python 接口。
- 将导入范围限定在 VMA 图像路径，避免构建与本任务无关的旧版点云算子。
- 复制原 MVX 基类到镜像内，保留图像 backbone/neck/head 构建逻辑，去掉无关的 3D 导入。
- 使用共享的 MMDetection bbox coder builder 和 Compose；补充被新 PyTorch 移除的 `torch._six` 常量。
- 处理未使用的旧 3D IoU 导出与 corner-pool 导入兼容问题。

这些调整只发生在镜像内的 VMA 副本，原 `/home/guyu/projects/VMA` 没有修改。不是完整 MMDetection3D 兼容层，也不是对全部 MMCV 算子的兼容承诺。

## 2. 实现与复现入口

- [Dockerfile](../Dockerfile)
- [完整环境说明和命令](../README.md)
- [构建脚本](../build.sh)
- [容器运行脚本](../run.sh)
- [兼容补丁](../runtime/patch_runtime.py)
- [推理与过拟合验证脚本](../runtime/validate.py)
- [CUDA 算子对照](../runtime/check_ops.py)

VMA 源码快照 commit：`a6e98e8f704e72499b33da84715ba39e8705e02e`。

最终镜像 ID：`sha256:76e5d4e80e4a3189f144b93b15888aa442633c4abe734de6f4b4887ab32df83c`。环境清单和构建日志保存在输出目录的 `pip_freeze.txt`、`base_image_id.txt`、`image_id.txt`、`build.log`。

当前运行入口见 [projects/vma_legacy/README.md](../README.md)，默认输出目录为 `/home/guyu/projects/mmhdmap/work_dirs/vma_legacy`；数据目录仅保留原始数据与 checkpoint。数据和 checkpoint 只读挂载；输出目录单独可写。运行容器禁用网络，使用 GPU 0 或通过 `VMA_GPU` 指定的单卡。

结果入口：[本地可视化汇总页](/home/guyu/projects/mmhdmap/work_dirs/vma_legacy/index.html)。

## 3. 权重和 CUDA 算子检查

`sd_line.pth` 的 1631 个参数/缓冲区条目与公开 SD line 配置构建的模型全部匹配：

| 检查 | 结果 |
| --- | --- |
| 缺失参数 | 0 |
| 多余参数 | 0 |
| 参数形状不匹配 | 0 |
| 加载方式 | `strict=True` |

Deformable attention 的 CUDA 实现与 PyTorch 参考实现对照：前向最大绝对差约 `1.19e-7`；三个输入梯度的最大绝对差分别约 `4.77e-7`、`1.91e-6`、`4.77e-7`。此外，完整网络训练实际执行了前向、损失和反向计算。

记录：[权重审计](/home/guyu/projects/mmhdmap/work_dirs/vma_legacy/checkpoint_audit.json)、[算子检查日志](/home/guyu/projects/mmhdmap/work_dirs/vma_legacy/ops_check.log)。

## 4. 样例推理

使用原仓库的 `generate_line_final_from_origin.py` 处理下载的 5 张 6000×6000 SD line 样例。由于部分图像含多条轨迹，得到 42 个 1000×1000 图块；原脚本划分为 35 个 train 和 7 个 valid。本次对全部 42 个图块推理。

总计 462 个 GT 实例，置信度不低于 0.3 的预测为 469 个。

| 匹配距离阈值 | Precision | Recall |
| --- | ---: | ---: |
| 5 px | 75.69% | 76.84% |
| 10 px | 78.89% | 80.09% |
| 20 px | 82.09% | 83.33% |

这是诊断指标，不是官方 mAP：预测和 GT 均按弧长重采样 100 点，计算对称平均最近邻距离，再进行同类别 Hungarian 匹配。需要结合未匹配实例计数理解几何误差。

这些样例可能与 checkpoint 原训练数据重合，不用于证明泛化性能。42 个图块均保存了 GT/预测对照图和矢量预测 JSON，可作为后续迁移的参考输出。

记录：[指标汇总](/home/guyu/projects/mmhdmap/work_dirs/vma_legacy/inference/summary.json)、[逐图块指标](/home/guyu/projects/mmhdmap/work_dirs/vma_legacy/inference/inference_metrics.json)、[示例对照图](/home/guyu/projects/mmhdmap/work_dirs/vma_legacy/inference/infer_000.jpg)。

## 5. 从 SD checkpoint 微调的单图块过拟合

图块为 `sample_4_5.jpg`，Dataset 索引 26，包含 7 个 GT 实例。使用真实 GT、原始类别/属性/几何损失，更新所有配置允许训练的参数。

设置：FP32，AdamW，初始学习率 `2e-5`，backbone 学习率倍率 0.1，weight decay 0.01，梯度裁剪 35；关闭 GridMask 与 dropout，学习率按 cosine 衰减，训练 300 步。保留原 backbone 冻结与 BN 行为。

| 指标 | 训练前 | 训练后 |
| --- | ---: | ---: |
| 预测实例数，GT 为 7 | 8 | 7 |
| 匹配实例平均 Chamfer 距离 | 2.074 px | 0.066 px |
| 5 px Precision | 75.00% | 100% |
| 5 px Recall | 85.71% | 100% |
| 总训练 loss，含中间 decoder 层 | 首步 4.588 | 末步 0.131 |

300 步训练阶段约 66 秒，峰值 PyTorch allocated 显存约 7.19 GiB。这是本机特定单图块、batch size 1 的测量，不代表正式训练吞吐或总显存需求。

记录：[过拟合结果](/home/guyu/projects/mmhdmap/work_dirs/vma_legacy/overfit_pretrained/overfit_summary.json)、[损失曲线](/home/guyu/projects/mmhdmap/work_dirs/vma_legacy/overfit_pretrained/loss_curve.png)、[训练后对照图](/home/guyu/projects/mmhdmap/work_dirs/vma_legacy/overfit_pretrained/overfit_0300.jpg)、[保存的权重](/home/guyu/projects/mmhdmap/work_dirs/vma_legacy/overfit_pretrained/overfit_line.pth)。

已在独立容器中重新构建模型、严格加载保存后的 checkpoint，再次推理该图块；预测数量和几何指标与保存前一致，平均距离仍为 `0.06621947687781969 px`。见 [重载结果](/home/guyu/projects/mmhdmap/work_dirs/vma_legacy/reload_check/inference_metrics.json)。

## 6. 重置整个矢量预测头的额外实验

为区分“已有模型本身预测良好”和“训练链路能够学习”，另做了更困难的实验：保留 SD backbone/neck，重新初始化整个 `pts_bbox_head`，包括 Transformer、查询和预测分支。

依次进行 300 步、降低学习率后的 700 步、关闭 dropout 并衰减学习率的 1000 步。阶段之间加载上一阶段权重、重新创建优化器，不是精确断点续训。

该实验从没有置信度达标的预测，学出了全部 7 个实例；2000 步后平均几何距离为 5.314 px，10 px 阈值 precision/recall 均为 100%，但 5 px 阈值均仅为 14.29%。几何精度仍不如从完整 SD checkpoint 微调的结果，不能把“实例数量恢复、loss 下降”当作从头训练已经达到精确过拟合。最终数值记录于 [额外实验结果](/home/guyu/projects/mmhdmap/work_dirs/vma_legacy/overfit_deterministic/overfit_summary.json)，各阶段均保留日志和可视化。

这个结果提示后续迁移时应分别检查类别、几何与属性损失，尤其是重新初始化查询与矢量头后的收敛，而不能只检查总 loss。

### 6.1 看起来像多个实例的原因：单个实例内部折返

针对可视化中的平行重线，重新加载 `overfit_deterministic/overfit_line.pth`，导出索引 26 的逐实例坐标，与原 SD 权重及完整权重微调结果比较。三者使用同一个 `predict()`，均为 `model.eval()`，绘图阈值均为 0.3；不存在训练图使用另一套去重逻辑的差别。

重置头实验最终输出 7 个实例，按同类别最近 GT 分配恰好分别对应 7 个不同 GT。问题出在实例内部的 50 个点：4 条 lane 分别有 21、21、23、23 段沿首尾方向反向移动超过 1 px，折线长度为首尾距离的 10.59–10.98 倍。单条折线反复往返且有横向偏差，因此叠画成数条平行线。原 SD 权重和完整权重微调后的 4 条 lane 均无此反向段，长度比约为 1。

代码中的 `Pts2LinesLoss` 对开放折线的内部点计算到相邻 GT 线段所在直线的距离；在近似直线的车道上，沿线移动对该项约束很弱。原配置点到线、点到点、方向损失权重分别为 7.5、0.05、0.005，后两项才直接帮助恢复点的位置顺序和方向。结合重新初始化整个 Transformer/查询/预测分支的设置，结果与“学会贴近目标线，但点序尚未收敛”一致。权重数字不能直接代表梯度比例；尚未通过损失消融证明单一因果。

Chamfer 距离主要反映几何覆盖，对沿相同位置往返不敏感。因此之前的 10 px precision/recall 为 100% 不代表点序或矢量拓扑正确，该重置头实验仍不能视为成功过拟合。后续评估还需检查点序、方向和折线长度。

诊断文件：[逐实例坐标](/home/guyu/projects/mmhdmap/work_dirs/vma_legacy/instance_diagnostics/reset_head/infer_026.json)、[各实例统计](/home/guyu/projects/mmhdmap/work_dirs/vma_legacy/instance_diagnostics/point_order_report.json)、[实例分色图](/home/guyu/projects/mmhdmap/work_dirs/vma_legacy/instance_diagnostics/reset_head_instances.jpg)、[仅绘制一个实例及前 8 段方向](/home/guyu/projects/mmhdmap/work_dirs/vma_legacy/instance_diagnostics/single_instance_backtracking.jpg)。可通过同目录的 `analyze.py` 重新生成诊断图和统计。

## 7. 本次验证的边界

已经验证的是：SD line 原网络、严格 checkpoint 加载、图块推理、原 Dataset 目标生成、匹配与损失、单卡优化、权重保存。

尚未验证：box/freespace、6000×6000 大图合并、原始分布式 `tools/train.py` 入口、FP16 训练、多图块或完整数据集训练，以及 HRDX 数据适配。

当前训练使用显式单卡 optimizer 循环调用原 VMA `forward_train`，不是宣称旧版完整 Runner/所有 mmdet3d 工具均已兼容。环境也未与可运行 CUDA 11.1 的旧 GPU 做逐位数值对照。

下一步可以把这里保存的输入图块、严格加载的权重和预测 JSON 作为 OpenMMLab 2.0 迁移的对照，再迁移训练接口。
