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
