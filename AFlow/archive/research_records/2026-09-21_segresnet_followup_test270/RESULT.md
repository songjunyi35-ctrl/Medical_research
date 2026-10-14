# 两个固定候选的270测试汇总

本次两个固定候选各一次270扫描/119患者测试均已完成。组合166在native grid的Scan 0.766272、Subject 0.750958、Point 236/270；对原native173差-0.002306、95%CI[-0.009078,+0.005791]。患者等权172在128³的Scan 0.759967、Subject 0.745647、Point 250/270；对预训练179差-0.001682、95%CI[-0.008389,+0.005860]。不同栅格不直接相减。验证门未过的事实保留；本次为用户授权、复用内部队列探索，无本批test调优或新outer/confirmation。[组合测试](archive/research_records/2026-09-21_segresnet_pretrain_native_e166_test270/RESULT.md) · [患者等权测试](archive/research_records/2026-09-21_segresnet_patient_equal_e172_test270/RESULT.md)。

| 候选 | 评分栅格 | Scan Dice | Subject Dice | Point |
|---|---|---:|---:|---:|
| 公开预训练＋原分辨率166 | native | 0.766271575 | 0.750958477 | 236/270 |
| 预训练＋患者等权172 | 128³ | 0.759967225 | 0.745646555 | 250/270 |

每个候选与同格参考的全部配对区间在各自报告；预训练179的native参考只是同模型重评分，不是新训练方法或额外候选。未因测试分数调整任何模型、阈值或参数，没有新outer/confirmation。

[组合报告](../2026-09-21_segresnet_pretrain_native_e166_test270/RESULT.md) · [患者等权报告](../2026-09-21_segresnet_patient_equal_e172_test270/RESULT.md) · [共同预注册](IDEA.md)。
