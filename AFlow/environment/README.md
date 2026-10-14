# BraTS / SAM-Med3D Python 环境

正式入口按以下优先级寻找 Python：

1. 环境变量 `AFLOW_PYTHON`；
2. `AFlow/.venv/bin/python`；
3. 临时兼容的 `AFlow/../sammed3d-venv/bin/python`；
4. 当前 `PATH` 中的 `python`。

推荐从仓库根目录建立本地环境：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu126 \
  -r environment/requirements-medical.txt
bash scripts/doctor_brats_single_repo.sh
```

`requirements-medical.txt` 是科研主链的最小依赖；
`sammed3d-venv.freeze.txt` 是 2026-08-13 整理时实际可运行环境的完整
`pip freeze`，用于审计和精确对照。当前旧 sibling venv 不删除，但正式脚本
不再依赖它的绝对路径。

如需显式覆盖源码或基础权重：

```bash
export AFLOW_PYTHON=/path/to/python
export AFLOW_SAMMED3D_ROOT=/path/to/SAM-Med3D
export AFLOW_SAMMED3D_CHECKPOINT=/path/to/sam_med3d_turbo.pth
```
