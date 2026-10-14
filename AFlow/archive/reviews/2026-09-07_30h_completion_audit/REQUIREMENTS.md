# 原30小时科研目标：完成条件与证据清单

状态：**全部完成条件已满足**。最终Goal计时检查timeUsedSeconds=108,030，超过108,000秒；两项最新实验完成、监督器均exit0，结果/收据/代码测试/文档校验已核对。机器见证见[FINAL_VERIFICATION](FINAL_VERIFICATION.json)。工作完成不等于所有模型晋级，也不抹去来源/数据复用限制。

## 原目标与授权变化

原目标：从当前交接接手，持续30小时，先完整链路审计，优先合法新外部来源；必要时做新train-only单因素研究，遵守七问、预注册、冻结评测、归档、文档校验，不因一项负结果提前停止。

用户中途暂停并交由Fable，后要求继续，并明确批准排除旧confirmation后的864扫描复用开发。故30小时按Goal累计活动时间核对，不能用9月5日至7日的日历时差，也不能把Fable的8.5小时当作本Goal额外已工作时长。新内部实验按用户批准的reused-development角色执行，不冒称原不可用的fresh inner容量恢复。

## 逐项验收

| 要求 | 当前证据与覆盖范围 | 完成判断 |
|---|---|---|
| 持续30小时，负结果后继续 | 最终Goal累计108,030秒；负结果后继续完成其他研究、综合分析和核验，没有按单次失败提前停止 | 已满足≥108,000秒；暂停/Fable时段不计入 |
| 阅读并审计完整链路 | [原session七条链路审计](../2026-09-05_30h_chain_audit/SESSION.md)、[教师患者暴露清单](../2026-09-05_30h_chain_audit/TEACHER_INNER_EXPOSURE.json)、[再次接手审计](../2026-09-07_gpt6_reentry/RESULT.md)、[暂停前N4家族36文件收尾](N4_FAMILY_CLOSURE.md) | 已核对；审计识别的来源/历史暴露限制仍保留，不称全部已修复 |
| 优先合法新外部来源 | 本Goal阶段已完成Africa95、CPTAC33、WAW-pre152、WAW-post31及UCSD75来源纠正；[5面板/14行对应核查](EXTERNAL_RESULT_LINKS.md)核对raw/receipt/cache-index和榜单一致 | 来源恢复与已完成结果证据已核对；受权限/标签/输入门阻塞者保持其关闭边界 |
| 区分Fable贡献 | [Fable 8.5h记录](../2026-09-06_fable_8p5h_session/SESSION.md)中的UCSD56由Fable完成；本次只审计其数值精度，不冒称本次新增队列 | 归属边界清楚 |
| 数据角色明确且患者隔离 | [授权](../../research_records/2026-09-07_reused_development_roles/AUTHORIZATION.md)、[SPLIT](../../research_records/2026-09-07_reused_development_roles/SPLIT.json)：fit684/313、dev180/78；旧confirmation53人排除，fixed validation/test患者零交叉 | 当前check_roles通过；每次formal admission/evaluation重新核对 |
| 七问和预注册先于训练/评测 | 两方法IDEA/PREREG由各IMPLEMENTATION_FREEZE和TRAIN_PREACCESS绑定；[收据链核查](RECEIPT_LINKS.md)校验六STARTED与训练前收据、开发前收据及原报告的绑定和时间顺序 | 已通过；时间为本地记录，不冒称第三方公证 |
| 单因素与固定控制 | 辅助只改loss系数0→0.1；采样只改每患者总概率，复用预指定control；六训练168轮/121步、配对初值/增强/导出均过，672个抽样epoch-policy SHA重建一致 | 已完成；同一干预可改变多个机制，未单独证明覆盖是唯一收益原因 |
| 冻结数据/checkpoint/grid/阈值/输出/成本/统计门 | 两PREREG固定128³、WT阈值0、Point argmax、无测试时增强/后处理、168轮、两seed、患者聚类bootstrap与成功/停止门；两DEV_PREACCESS、六checkpoint和RAW均已绑定 | 已完成，未改门、未选seed或追加模型访问 |
| 不用旧test选择、不冒称盲证据 | 两项新方法只评批准的reused dev180；最终访问字段无新test/outer/confirmation/external；两个训练seed共用78人 | 两项字段与调用链已核验，未把两seed当156名独立患者 |
| 实现失败透明、保留负结果 | 辅助初版导出差在formal前修复，最终error0；辅助全部效果门失败；采样Dice四CI正但seed11 Point−3，联合门失败保留；描述性求和测试失败及修正已记录 | 两项已收口；不抹去Dice正证据，也不平均Point或挑seed放行 |
| 原始报告、收据、checkpoint/cache-index重SHA | 辅助[19条归档](../../research_records/2026-09-07_structured_background_aux/ARCHIVE_SEAL.json)、采样[25条归档](../../research_records/2026-09-07_subject_balanced_sampling/ARCHIVE_SEAL.json)，共44文件条目重SHA通过（含复用条目） | 已完成；原raw/RESULT JSON不因解释补充而改写 |
| 数值核验覆盖真实要求 | 两正式验证器重算计数Dice、患者平均、区间、门；两份点审计各720行，含复用control，唯一模型—病例对1080全部exact | 已完成；未保存logit，不声称独立argmax或完整预测mask重放 |
| 榜单角色/排名/CI与MD/JSON同步 | leaderboard validator通过；复用开发不加入test/旧validation108榜；UCSD56 MD/JSON状态措辞同步纠正，非status字段投影SHA不变 | 已通过；最后封存时再运行，不把validator当来源真实性证明 |
| 根文档布局与方法解释 | 9个根Markdown合法；过时训练阶段已归档，catalog两终态与综合报告齐全；新增归档35文档/100链接初检无失效 | 已更新；新增机制/总审计文档后最后再核全部链接 |
| 代码与资源安全 | 最新全仓814测试通过、27既有warnings、40.95秒；不修改运行中冻结代码、不删材料。两个监督器exit0，GPU只读查询均0MiB/0%，workspace约6.6GB余量 | 代码和任务终态通过；最终再确认资源状态，不干预其他用户进程 |
| 最终交付 | 中文综合报告说明单因素、当前可比整体/1×范围、CI、成本、访问和机制；handoff/catalog/leaderboard/report/archive/checkpoint入口齐全 | 已完成；保留正证据与失败门，不把工作完成当整体性能成功 |

## 不能抹掉的限制

官方BraTS Synapse与本地Kaggle逐文件字节parity未完成；新864池是历史复用，不是独立盲开发。EGD374和CoRe-BT250仍需用户/机构完成数据使用协议，未代签、填表或发邮件。其他关闭队列的标签域/输入几何门不因额度充足而放宽。旧外部观察属于已知研究背景，不是新的候选确认；当前结果不证明某外部错误已解决。

## 下一步

不要再轮询或恢复已结束的65466/12286。原30小时轮次完成；未来研究需遵守当前数据授权和新的前瞻协议。EGD/CoRe-BT协议、Synapse parity等仍是公开记录的限制，不因本Goal完成而视作已解决。
