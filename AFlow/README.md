# AFlow：单T1ce脑肿瘤分割研究

输入仅为T1ce磁共振图像；目标为整体肿瘤，包含标签1/2/3，不包含切除腔标签4。

## 最新结果（2026-10-14）

固定原972训练扫描／444患者、108验证扫描／50患者、270内部测试扫描／119患者，输入始终为单T1ce。最新完整nnU-Net残差编码器L参考与有限融合已完成：

| 原生1mm栅格，内部270扫描 | Scan Dice | Subject Dice | 点命中 |
|---|---:|---:|---:|
| 先前最好融合 | 0.775891 | 0.761694 | 261/270 |
| 新残差编码器L单模型 | **0.811283** | 0.795536 | 265/270 |
| 新模型与两个既有SegResNet的固定融合 | **0.812965** | **0.797512** | **266/270** |

最佳融合较先前最好融合增加0.037074，患者整簇配对95%区间为[0.030140, 0.044262]；相对新单模型增加0.001683。融合为新模型前景减背景logit权重0.75、旧native173与旧组合166各0.125，概率阈值0.4。

**这些是复用内部测试队列的test-tuned探索性结果。** 已使用270结果进行诊断及在六项事先固定融合中选择；不代表独立泛化、官方隐藏测试或外部确认。不同栅格与推理成本分开列示，不能直接横比。

训练完成550/550轮，137500次更新尝试、137449次实际优化器更新，双A10040GB约15.47小时。由108验证选择第525轮，原生Scan Dice为0.784672。550轮为预算约束，不是默认1000轮的完整复现；整个参考管线改变多项因素，不能把收益全部归因于架构。

- [最终报告、区间与访问范围](archive/research_records/2026-10-13_nnunet_resenc_reference/RESULT.md)
- [完成审计](archive/research_records/2026-10-13_nnunet_resenc_reference/COMPLETION_AUDIT.json)
- [完整结果独立复核](archive/research_records/2026-10-13_nnunet_resenc_reference/INDEPENDENT_RESULT_AUDIT.json)
- [训练验证曲线](archive/research_records/2026-10-13_nnunet_resenc_reference/VALIDATION_CURVE.png)

## 固定比较基线与历史主线

研究从单T1ce → 图像处理 → SegResNet → WT分割展开，原N4纯193仍是比较锚点。历史无N4基础对照已按用户规则停在50/200轮，未见明确同轮差异；当前已完成最新残差编码器参考及条件融合，没有运行中的训练。

| 材料 | 入口 |
|---|---|
| 历史无额外N4对照 | [50轮停止结果](archive/research_records/2026-09-18_segresnet_plain972_no_n4/RESULT.md) |
| 原N4纯193基线 | [结果与模型](archive/research_records/2026-09-17_segresnet_plain972_e193_test270/RESULT.md) |
| 清理记录 | [已删除SAM权重与旧派生缓存](archive/reviews/2026-09-18_segresnet_only_cleanup/RESULT.md) |

`sammed3d/`目录包含沿用的SegResNet/数据公共实现，名称保留以维持历史代码哈希。旧入口供历史审计；实验归档保留正负结果，已删除SAM权重不能直接重放。

## 我要向导师汇报，先看哪里？

1. **[导师汇报](AFLOW_BRATS_WEEKLY_REPORT.md)**：起点是什么、做了什么优化、Dice达到多少、哪些结论成立。
2. **[从8月13日起的日期流水账](AFLOW_BRATS_RESULTS.md)**：按日期连续读完整研究过程。
3. **[SegResNet方法目录](AFLOW_BRATS_METHOD_CATALOG.md)**：从固定纯模型基线出发，查看主要方法、同栅格结果和负结果索引。

## 当前成绩

固定原972纯SegResNet第193轮：128³栅格验证Scan Dice为0.740032、内部270测试为0.759179，点命中242/270；映射到相同原生栅格的参考Scan Dice为0.761134。128³与原生栅格分开比较。最新0.811283／0.812965见上方和排行榜，原193仍是固定比较锚点。

师兄路线基础模型复现为 **0.594226**，但采用真实标签辅助定位和交互点击；与全自动成绩不是同条件对照。

## 我要核对证据或继续实验

- [当前接续状态](AFLOW_BRATS_HANDOFF_CURRENT.md)
- [分队列排行榜与置信区间](AFLOW_BRATS_LEADERBOARD.md)
- [原始研究证据索引](archive/reviews/2026-09-14_mentor_facing_cleanup/EVIDENCE.md)
- [执行与文档规范](AGENTS.md)
- [新方法通俗预注册模板](EXPERIMENT_IDEA_TEMPLATE.md)

技术路径、完整哈希、运行日志和历史阶段状态放在资料层；导师汇报正文不再复制这些内容。

最新参考代码在`scripts/nnunet_resenc_20261013/`；训练实际采用修订入口v2，恢复评测采用`finalize_v3.py`，原冻结源码与工程修订收据均保留。历史脚本的绝对路径是运行证据，迁移时须明确映射新路径并核验哈希。GitHub科研迁移版只含代码、文档和记录，不含MRI影像、患者标签、模型权重、预测缓存或本机凭据。


## 根目录其他文件是什么？

平时只需看上面的导师汇报、日期流水账和方法记录。`AGENTS.md`及当前交接供Agent接续；排行榜MD用于精确核对，唯一保留的JSON是它的机器数据；方法模板只在新实验前使用。旧交接现为短跳转页，不再是一份重复的当前交接。

三份旧JSON与旧哈希清单已移入[历史技术材料](archive/legacy_root_artifacts_20260914/README.md)。这是历史文档迁移记录。2026-09-18按用户授权另行删除SAM权重和不用的派生缓存，详见当前清理收据。
