# 开发工具

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
