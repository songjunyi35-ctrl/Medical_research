# SAM-Med3D vendored snapshot

- 上游项目：`uni-medical/SAM-Med3D`（上游 README 声明为官方仓库）
- 项目地址：<https://github.com/uni-medical/SAM-Med3D>
- 本地来源：`/mlx_devbox/users/songjunyi.666/playground/SAM-Med3D-main`
- 复制日期：2026-08-13 UTC
- 许可证：Apache License 2.0，完整文本见 [`LICENSE`](LICENSE)
- 精确上游 commit：未知。来源目录不是 Git 仓库，不能可靠恢复 commit ID；因此
  本快照以逐文件 SHA-256 作为可复现身份。

只复制当前 AFlow 科研实际依赖的小型源码：`segment_anything/`、`utils/`、
`train.py`、`inference.py`、`validation.py`，以及历史 Phase-0 入口需要的
`phase0_reproduce.py`、`prompt_runner.py`。同时保留上游 README 和 LICENSE。

没有复制论文 PDF、assets、演示数据、旧 results、日志、`__pycache__` 或上游
checkpoint。基础权重单独保存在 AFlow 的
`artifacts/checkpoints/sam_med3d_turbo.pth`，不进入 Git。

完整文件哈希见 [`SHA256SUMS`](SHA256SUMS)。关键文件：

| 文件 | SHA-256 |
|---|---|
| `train.py` | `474d9765e29c0196e9cf1da2242869c2cbd6970632d9f113357d38a8094e1e3f` |
| `inference.py` | `5a31aff32805f297eb5632b4362d1aa60bde4491d168868bc750a43e332c3960` |
| `validation.py` | `e64661b8f06d3d1a531a051b96c85cb2d257ce5da2aad3a1ef2e0544cc5b8cc3` |
| `utils/click_method.py` | `0838972fb2f33de579a2652d3225f862d84dc6df113b152ea6018730c9b792d7` |
| `LICENSE` | `43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1` |

原 `SAM-Med3D-main` 目录是只读备份，不在本次整理中删除或修改。
