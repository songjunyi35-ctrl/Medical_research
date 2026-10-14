# 占比target结果：定位改善，分割提分未支持

两seed固定168轮完成，原图像、抽样、原增强RNG与初值/全部记录的第一batch图像byte配对通过，export均0；开发统一按原hard GT评分，独立复核完成。

seed11：Scan0.746658626、Subject0.732690014、Point175/180；相对nativecontrol差−.000103690/−.000522819、Point+5。两Dice95%CI跨0。

seed12：Scan0.739852166、Subject0.726270323、Point167/180；差−.004325251/−.003047812、Point+4。Scan CI[-.009006786,-.000025938]刚低于0，Subject CI跨0；不把极近0的端点夸大为普遍/稳健显著恶化。两seed平均Scan差−.002214471，主要增益门直接失败，不由CI数值精度决定。

直觉分解：几何占比构造和总量守恒工程得到支持；它并没有在这套原DiceCE损失/固定阈值/硬GT评分下提高分割。两个seed定位命中都有增加，但不等于完整边界覆盖更好，也不证明概率校准或具体边界机制。质量守恒误差缩小不能直接换成模型Dice收益。

成本：两新训练各约7673秒，合计4.2627 GPU小时、峰值20.664GiB；开发360forwards/10.99秒；soft label cache2.672GiB，CPU构建39.27分钟。部署仍4.70M/1×且不需要标签cache。只有同一dev180/78 reused evidence，没有test/oldconfirmation/fixed validation/outer/external访问。

当前配方及其单模型seed11条件性测试分支关闭，不扫核/阈值/seed，不把两个定位正结果当分割成功，不宣称所有软标签方法无效。SoftSeg整套方法同时改激活/loss，此结果也不是完整SoftSeg复现结论。

10:31后续高层安排显式修订：另立不同成本的独立2模型等权融合研究，检验旧患者采样模型的覆盖与占比模型的定位是否能从同一平均logit输出中互补。它是开发揭示后的新提案，两个成员原来的失败状态不变，不改本实验门，不称1×提分；单权重失败就关闭。没有依据test选择成员。

证据：[结果](RESULT.md)、[raw](RAW_DEVELOPMENT.json)、[验证](VERIFICATION_RECEIPT.json)、[训练重放](TRAINING_REPLAY_FINAL.json)、[新独立融合提案](../2026-09-10_data_recipe_ensemble/IDEA.md)。
