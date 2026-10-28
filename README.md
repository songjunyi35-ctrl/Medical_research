# Medical_research

单T1ce MRI脑肿瘤分割研究代码、实验档案和论文材料。2026-10-14工作树快照，沿用原科研迁移包的范围，保留历史正负结果。

## 最新Dice已超过0.8

固定972训练扫描／444患者、108验证扫描／50患者、270内部测试扫描／119患者，WT标签1/2/3，排除切除腔4。

| 内部270扫描，原生1mm栅格 | Scan Dice | Subject Dice | 点命中 |
|---|---:|---:|---:|
| 先前最好融合 | 0.775891 | 0.761694 | 261/270 |
| nnU-Net残差编码器L单模型 | **0.811283** | 0.795536 | 265/270 |
| 新模型与两个既有SegResNet的固定融合 | **0.812965** | **0.797512** | **266/270** |

最佳融合较先前最好融合增加0.037074，患者整簇配对95%区间为[0.030140, 0.044262]。融合相对新单模型增加0.001683。

**这些是test-tuned探索性结果，不是独立泛化或官方隐藏榜成绩。** 原内部270队列被重复用于诊断与有限融合选择，未取得新的outer/confirmation评测；270标签未进入训练梯度。保留原始角色清单、验证选优规则及全部六项固定融合结果。108验证选第525轮；训练完成预算内550轮，不能称为默认1000轮完整复现。完整管线有多项变化，不能把收益全部归因于架构。不同栅格和推理成本分开比较。

## 2026-10-28论文算法调研存档

新增24篇相关研究的比较、可靠解剖参照与变化证据解耦分割候选方案，以及六页导师汇报稿。当前为研究设计，尚未实现或训练，没有新增精度结果。

- [算法调研与选题比较](AFlow/archive/reviews/2026-10-28_thesis_algorithm_survey/REPORT.md)
- [算法草案与消融设计](AFlow/archive/reviews/2026-10-28_thesis_algorithm_survey/IDEA.md)
- [导师汇报稿](AFlow/archive/reviews/2026-10-28_thesis_algorithm_survey/BRIEFING.md)

本次仅补充上述调研材料，其他代码和实验档案仍沿用原发布快照；不表示已同步截至10月28日的全部工作树。

## RACE-Seg算法实现存档

已实现单T1ce整脑分割的局部解剖参照检索、变化类型对比和参照失效门控，并通过7项CPU合成检查及双A10040GB完整原生尺寸合成检查。本次提交存档实现与首次匹配实验协议；患者效果由后续实验结果判断。

- [实现说明与结构图](AFlow/archive/research_records/2026-10-28_race_seg/IMPLEMENTATION.md)
- [冻结实验方案](AFlow/archive/research_records/2026-10-28_race_seg/IDEA.md)
- [核心模型](AFlow/sammed3d/race_seg.py)、[训练与协调器](AFlow/scripts/race_seg_20261028/)、[合成检查](AFlow/tests/test_race_seg.py)

## 阅读入口

- [最终结果报告与比较区间](AFlow/archive/research_records/2026-10-13_nnunet_resenc_reference/RESULT.md)
- [导师汇报](AFlow/AFLOW_BRATS_WEEKLY_REPORT.md)
- [排行榜](AFlow/AFLOW_BRATS_LEADERBOARD.md)与[机器结果](AFlow/AFLOW_BRATS_LEADERBOARD.json)
- [方法目录](AFlow/AFLOW_BRATS_METHOD_CATALOG.md)、[完整研究过程](AFlow/AFLOW_BRATS_RESULTS.md)、[当前交接](AFlow/AFLOW_BRATS_HANDOFF_CURRENT.md)
- [完成审计](AFlow/archive/research_records/2026-10-13_nnunet_resenc_reference/COMPLETION_AUDIT.json)与[独立结果复核](AFlow/archive/research_records/2026-10-13_nnunet_resenc_reference/INDEPENDENT_RESULT_AUDIT.json)
- [新参考管线代码](AFlow/scripts/nnunet_resenc_20261013/)、[历史公共实现](AFlow/sammed3d/)
- [迁移与环境指南](README_MIGRATION.md)、[论文材料](article/)

![108验证曲线](AFlow/archive/research_records/2026-10-13_nnunet_resenc_reference/VALIDATION_CURVE.png)

## 仓库范围与复现

包含代码、配置模板、角色清单、实验协议、JSON/CSV结果、日志和文档。沿用原迁移包排除规则，不上传MRI影像、患者标签、模型权重、预测缓存、虚拟环境和本机凭据。**克隆代码后不能直接生成原模型预测**；需要另行取得合规数据和对应权重、映射历史绝对路径并核对哈希。固定纯SegResNet e193仍是比较锚点，128³测试Dice 0.759179不能直接与原生栅格相减。

```bash
git clone https://github.com/songjunyi35-ctrl/Medical_research.git
cd Medical_research
python3 verify_migration_bundle.py
```

`MANIFEST.sha256.tsv`记录本次快照每个文件的SHA-256及大小；校验不依赖Git历史。上游许可证保留在各自目录，本仓库不重新授权第三方代码或论文。历史脚本和冻结收据按原样保留，启动前须阅读对应协议；旧实验不会因上传而自动恢复。
