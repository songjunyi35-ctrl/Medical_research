# 本轮两个机制结论的一手文献边界（只读检索，2026-09-06）

目的：为本session两条已冻结的描述性机制结论提供文献上下文；不改变任何已保存数字，不授权调参。

## 1. Flip8的"保守去假阳性"机制有文献先例，但幅度不可移植

- 项目观察：固定Flip8−identity的precision点估计在有保存计数的五个外部cohort（Africa95/CPTAC33/WAW-pre152/WAW-post31/UCSD-BraTS2024test56）中5/5为正，其中四处事后CI全正；预测体素通常减少；recall近零。
- 一手先例：细胞分割TTA研究（Scientific Reports 2020, 10.1038/s41598-020-61808-3）明确写TTA"can help to fit whole objects (remove false positives or add true positives)"，并把收益描述为"better segmentation borders and a reduced rate of false positive or/and false negative detections"；其U-Net聚合方式与本项目相同（对逆变换概率图求平均再阈值）。TTA不确定性框架的经典来源为Wang et al., Neurocomputing 2019（aleatoric uncertainty via TTA）。
- 边界：文献任务（细胞/2D、Mask R-CNN实例分割）与本项目3D T1ce WT不同，数值幅度与显著性不可搬运；本项目Africa上Flip8曾表现为precision升+体素减，而CPTAC上为recall升，说明"共识去偶然多画"只是主导倾向而非普适定律。8×成本结论不变。

## 2. Sharpness-aware家族的OOD收益在文献中本就是混合/条件性的

- 项目观察：GCSAM184 checkpoint相对Full972 KD168的Scan差在七个已揭示cohort中方向可逆：仅Africa95（远域、大WT）CI全正，仅UCSD-BraTS2024test56（近训练域、患者隔离）CI全负，其余五处跨0；事后分解显示其"扩张/coverage"倾向在近域变成纯precision代价。
- 一手上下文：SAM家族对OOD泛化的文献证据并不一致——有工作报告SAM/变体在zero-shot OOD上显著优于Adam基线（ACM ISICT 2023, 10.1145/3628797.3629009），也有分析明确"SAM does not improve the generalization in OOD"且"minimizing the total loss sharpness does not guarantee sharpness across individual domains"（arXiv 2412.05169）。
- 边界：本项目的GCSAM/Full差异不是optimizer单因（fit864 vs 972、epoch、seed同时不同；见2026-09-06_gcsam_external_interpretation），文献引用只说明"平坦度→跨域收益"在一般文献中同样是条件性的，与本项目"方向依cohort可逆"的观察相容；不构成任何重训、rho搜索或router授权。

## 使用规则

两节均为解释性上下文：不得据此宣称显著、外推到未见域，或反向选择checkpoint/视图/阈值。引用以原文为准，未下载付费全文的只引用检索可见摘要/开放版本。
