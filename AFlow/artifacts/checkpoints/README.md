# 本地 checkpoint

该目录中的大权重只保存在工作磁盘，不进入 Git。默认基础权重为：

`sam_med3d_turbo.pth`

期望 SHA-256：

`899a46d04d3b70f723282ceb489149373558bf0aaba389a346f5ab57da5cdd3c`

可用 `AFLOW_SAMMED3D_CHECKPOINT` 覆盖默认路径。不要把训练 checkpoint 从
`workspace/BraTS/` 移到这里；那些路径已绑定历史结果和哈希。
