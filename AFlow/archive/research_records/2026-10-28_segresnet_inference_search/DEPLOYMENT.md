# 固定w32第171轮已选推理配方

唯一权重及配方见DEPLOYMENT_RECIPE.json。新增训练0。五种镜像视图编号2、3、4、6、7，等权概率平均，温度2的确定性概率体积解码；不读取标签。该方案在原108验证选择，未新增270测试。

沿用原单T1ce的N4、原生裁剪/归一化和160×208×160零填充。工具接收预处理后的.npy图像，形状为[D,H,W]、[B,D,H,W]或[B,1,D,H,W]；不会将裸NIfTI误当作已预处理输入。geometry.json为单例几何或等长列表，例如：

```json
{"canvas_offset": [12, 20, 8], "extent": [135, 167, 144]}
```

在仓库根目录执行：

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 ../sammed3d-venv/bin/python -m scripts.segresnet_inference_search_20261028.deploy --images preprocessed_images.npy --geometry geometry.json --output predictions
```

输出每例原生裁剪格的mask与automatic_point（.npz），坐标相对裁剪格；返回原始NIfTI格时沿用原图像几何和裁剪origin。已有输出不会覆盖。增强视图batch8，原始视图（若依赖）batch2；批量及运行环境改变不保证浮点逐位一致。

8例图像到掩膜独立重算已与归档逐体素完全一致，见DEPLOYMENT_VERIFICATION.json。至少保留一个体素的体积解码只在本批WT非空病例验证，不能把本验证结果当无肿瘤筛查性能。
