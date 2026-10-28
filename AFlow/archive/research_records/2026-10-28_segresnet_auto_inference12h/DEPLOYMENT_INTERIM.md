# 当前无训练核心视图部署接口

当前审计最高`0023_brain_holes`，Scan0.784311696/Point260，8次前向。当前最高的独立API重新前向270尚未完成；旧0008的fresh270通过仅支持旧配方，不能替代当前最高核验。

```python
from scripts.segresnet_auto_inference12h_20261028.deploy_core_v3 import FrozenCorePredictor
predictor = FrozenCorePredictor("archive/research_records/2026-10-28_segresnet_auto_inference12h/PORTABLE_RECIPE_CURRENT.json")
# images: float32 [B,1,X,Y,Z], original native1mm T1ce normalization; dimensions divisible by8
# geometry: image-derived canvas_offset/extent, removes padding
predictions = predictor.predict(images, geometries)
# each result: boolean mask and automatic_point in original native crop
```

API不接GT、病例ID或逐例阈值表；从图像/自身多视图预测计算支撑、权重与切点，严格核验唯一171轮checkpoint SHA，eval/bf16、无梯度。脑区支撑需输入中至少存在非零脑体素，合成显存门使用ones。

v3支持核心镜像、共识/case/非线性/joint、图像脑区支撑、局部MRF及组件/形态组合；group/ROI/phase若成为最高须额外对应预测器，不得强行套核心视图接口。输入必须沿用工程原预处理，不宣称任意外部MRI可直接代入。

旧0008独立fresh270核验：[2160次前向/全部mask0差](live_verification/0008_consensus_cc/RESULT.json)。当前候选是否fresh完成以其live_verification/DONE为准。独立复现只是工程一致性，所有规则仍test-tuned，不是独立泛化。模型与输入不上传科研库。
