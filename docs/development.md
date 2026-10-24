# 开发工具

## 安装开发包

在实际运行代码的 Python 环境中，从项目根目录执行可编辑安装：

```bash
python -m pip install -e .
```

例如在开发容器中，项目根目录是 `/workspace/mmhdmap`。安装后可从其他目录
导入 `mmhdmap`，不需要在脚本中修改 `sys.path`。修改已有 Python 源码后，
重新运行程序即可；修改包元信息或依赖后，应重新执行安装命令。

`setup.py` 声明公共库直接使用的 MMEngine、NumPy、PyQuaternion 和 PyTorch
依赖。请先安装与设备匹配的 PyTorch；现有开发镜像已经提供这些依赖。
MMCV 的 CUDA 算子、MMDetection、MMDetection3D 和数据集 SDK 等实验依赖
继续由 Docker 配置管理，安装公共包不会自动配置完整实验环境。

发布包仅包含 `mmhdmap` 及其子包。`projects` 中的实验代码、配置文件及
`tools` 脚本从源码仓库使用，不包含在 wheel 中。

需要开发工具时，可以使用：

```bash
python -m pip install -e '.[dev]'
```

这会额外安装构建工具和 pre-commit；Git hook 仍需按下文单独启用。
`pyproject.toml` 声明 setuptools 构建后端，包信息和依赖统一维护在
`setup.py` 中。

## 提交前自动格式化

项目通过 pre-commit 调用固定版本的 YAPF，仅执行 Python 代码格式化。
规则由根目录的 `.style.yapf` 定义。

在实际执行 `git commit` 的环境中，从项目根目录安装并启用：

```bash
python -m pip install pre-commit
pre-commit install --install-hooks
```

每个开发者 clone 仓库后都需要执行安装命令。首次安装 hook 环境需要
访问 GitHub 和 Python 包源，后续复用缓存环境。宿主机和容器共享仓库时，
应在执行提交的环境中安装 hook，确保生成的 hook 可以访问 Python 解释器。

提交时，YAPF 会格式化本次暂存的 Python 文件；处理范围是整个文件，
不局限于修改的行。格式无需改变时提交继续；格式被修改时提交暂停，
改动保留在工作区，不会自动暂存。

查看格式修改后，重新暂存对应文件并提交：

```bash
git diff
git add path/to/changed_file.py
git commit -m "Describe the change"
```

手动处理已暂存的文件：

```bash
pre-commit run yapf
```

手动处理指定文件：

```bash
pre-commit run yapf --files mmhdmap/datasets/base_map_dataset.py
```

当前 hook 不运行 import 排序、语法升级、lint 自动修复或测试。

## 查看地图在环视图像中的投影

在开发容器的项目根目录运行：

```bash
python tools/stage1/test_nusc_map_dataset.py \
    projects/map_tracker/configs/map_tracker_stage1.py \
    --out-dir work_dirs/map_projection_vis
```

相机地图投影默认开启；右侧 BEV 仍显示原生分辨率底图
与标注。默认提取 `road_divider`、`lane_divider`、`ped_crossing`，可以通过
`--map-layers` 选择类别。使用 `--basemap-dir ''` 可以只提取
标注并投影到相机，不读取大型底图 PNG。`--sample-idx` 用于选择样本。

检查脚本使用高度偏移 0：路面平面为 `global_z = ego2global[2, 3]`。
这是假设路面经过参考 LiDAR 时刻的 ego 原点，并非已知路面高程。
这种水平面近似不能恢复坡度、道路起伏或高架道路。

`NuScenesMapExtractor` 提供局部几何及其 `local2global` 平面变换；
`visualization/map_projection.py` 负责三维坐标转换、近裁剪和图像边界裁剪；
`MapLocalVisualizer.add_datasample` 默认 `project_map=True`、`ground_height=0`，
调用方提供地图几何及 `map_local2global`，在原始 RGB 图像副本上绘制，随后
按现有布局缩放拼接。折线分段和多边形孔洞分别绘制，多边形仅显示轮廓。
仅展示相机或 BEV 时，可在 Python API 中显式传入 `project_map=False`。

传入图像必须与相机内参对应；图像经过 resize、crop 或 flip 后，调用方
需要同步更新投影参数。当前使用新版 PKL 中每路相机拍摄时刻的 ego pose
和静态外参计算 global2cam，考虑相机与参考 LiDAR 的采集时间差。
六路相机共享同一个地图平面，高度仍以参考 LiDAR 时刻的 ego 原点为基准，
不会随各相机位姿变化。投影尚未处理前景遮挡，也没有真实道路高程，
不能直接当作精确的图像分割 GT。

Dataset 配置读取 `mmhdmap_nuscenes_mini_infos_{train,val}.pkl`，当前容器
数据根目录为 `/home/guyu/data/nuscenes/mini`。底图目录默认取 Dataset
的 `data_root/basemap`，也可通过 `--basemap-dir` 覆盖。
