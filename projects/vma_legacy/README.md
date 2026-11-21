# VMA Legacy：原版 SD line 实验

这里维护原 VMA 的兼容镜像、Compose 和验证脚本，尚未迁移到 OpenMMLab 2.0。背景调研见 [VMA 调研文档](../../docs/vma/README.md)。

## 文件与数据

- `Dockerfile`、`docker-compose.yaml`：构建和运行 `vma_legacy:blackwell-cu128`。
- `build.sh`、`prepare.sh`、`inference.sh`、`train_overfit.sh`：构建、切块、推理和单图过拟合。
- `runtime/`：兼容补丁、原模型训练/推理验证程序。
- 数据与原始权重保留在 **`/home/guyu/data/vma`**：

```text
/home/guyu/data/vma/
├── sd_data/       # 原始 SD 样例
├── sd_ckpts/      # sd_line.pth 等
└── nyc_ckpts/
```

实验结果、切块和构建日志统一保存在 **`mmhdmap/work_dirs/vma_legacy`**（已被 Git 忽略），可通过 `VMA_WORK_DIR` 调整。

原 VMA 仓库默认 `/home/guyu/projects/VMA`，只读挂载到 `/workspace/VMA`。启动时复制到容器临时目录，应用兼容补丁后运行；宿主机原仓库保持不变。`runtime/` 只读挂载，数据/权重只读挂载到 `/data`，结果写入 `/work_dirs`。

## 运行

需要 Docker Compose（支持 additional_contexts）、NVIDIA Container Toolkit，以及已有的 **`mmhdmap-dev:latest`** 基础镜像：PyTorch 2.7.0 / CUDA 12.8、为 Blackwell sm_120 编译的 MMCV 2.1.0 扩展。构建时需下载依赖；运行容器禁用网络。

```bash
cd /home/guyu/projects/mmhdmap/projects/vma_legacy
# 可选：cp .env.example .env，然后调整 VMA_REPO、VMA_DATA_DIR、VMA_WORK_DIR、VMA_GPU
bash build.sh
bash inference.sh                   # 自动准备数据，推理全部 42 个图块
bash train_overfit.sh                # sample_4_5.jpg，完整 SD 权重微调 300 步
```

每次运行创建 `work_dirs/inference_时间_随机后缀/` 或 `overfit_时间_随机后缀/`，保留 `run.log`、指标 JSON、可视化；训练另保存最终预测和 `overfit_line.pth`（仅模型权重）。脚本自动使用当前用户 UID/GID。过拟合默认 lr=2e-5、cosine LR，关闭 GridMask/dropout，保留原模型、匹配和损失。

```bash
VMA_GPU=1 bash inference.sh --index 26
bash train_overfit.sh --steps 500 --lr 1e-5
# 替换为实际输出目录，独立加载过拟合权重：
bash inference.sh --index 26 \
  --checkpoint /work_dirs/overfit_时间_随机后缀/overfit_line.pth
bash run.sh python /opt/vma-support/check_ops.py
```

镜像保留 MMCV 1.4 / MMDetection 2.14 的旧 Python 接口，复用基础镜像的 CUDA 扩展。构建直接读取 `VMA_REPO` 作为独立构建上下文，不向 mmhdmap 复制 VMA 源码、样例数据或 checkpoint。

## 已验证结果

- 原 SD checkpoint：1631 项严格匹配，42 个图块完成推理。
- 完整权重微调：300 步后 7 GT / 7 预测，平均 Chamfer 约 0.066 px；保存重载成功。
- 重置预测头的额外实验仍有点序折返，不能当作精确过拟合。

[完整实验记录](docs/container_validation.md) · [已有结果汇总](/home/guyu/projects/mmhdmap/work_dirs/vma_legacy/index.html)

指标为样例诊断，非官方 mAP；尚未验证 box/freespace、大图合并及原分布式训练入口。`/home/guyu/data/vma` 为实际数据目录，没有软链接；容器 `/output` 挂载用于兼容历史记录。数据目录仅保留数据与原始 checkpoint。后续迁移版的数据目录建议使用 `vma_mmhdmap`。
