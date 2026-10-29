# 最终选定推理配方的部署接口

当前审计最高`0036_local_brain_joint`，Scan0.784382113/Point260，8次前向。当前最高独立API重新前向270例通过，mask差0体素，所有point相同，Dice差0。

```python
from scripts.segresnet_auto_inference12h_20261028.deploy_group_v1 import FrozenGroupPredictor as FrozenCorePredictor
predictor = FrozenCorePredictor("archive/research_records/2026-10-28_segresnet_auto_inference12h/PORTABLE_RECIPE_CURRENT.json")
# images: float32 [B,1,X,Y,Z], original native1mm T1ce normalization; dimensions divisible by8
# geometry: image-derived canvas_offset/extent, removes padding
predictions = predictor.predict(images, geometries)
# each result: boolean mask and automatic_point in original native crop
```

API不接GT、病例ID或逐例阈值表；从图像/自身多视图预测计算支撑、权重与切点，严格核验唯一171轮checkpoint SHA，eval/bf16、无梯度。脑区支撑需输入中至少存在非零脑体素，合成显存门使用ones。

当前Group接口兼容核心配方，并新增确定性强度单组/多组融合；core计算仍复用冻结v3。ROI/phase专用视图不在该接口范围，必须额外对应预测器，不能冒称支持。输入必须沿用工程原预处理，不宣称任意外部MRI可直接代入。

旧0008独立fresh270核验：[2160次前向/全部mask0差](live_verification/0008_consensus_cc/RESULT.json)。当前候选是否fresh完成以其live_verification/DONE为准。独立复现只是工程一致性，所有规则仍test-tuned，不是独立泛化。模型与输入不上传科研库。
