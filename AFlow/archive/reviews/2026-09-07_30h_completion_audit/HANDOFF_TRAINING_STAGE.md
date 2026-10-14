# 2026-09-07训练阶段交接摘录（历史）

以下保留06:39–10:42阶段记录；两项实验后来均已完成，不要据此恢复训练或评测。原root相对链接已调整为本归档位置。

### 06:39–10:42训练期间记录（以下未开dev/运行中为当时快照）

首个正式候选为训练期切除腔辅助监督：共享双通道SegResNet，两臂从零、无教师；唯一变化aux loss权重0→0.1，最终裁掉aux仍1×。固定168轮×2seeds×2arms、四checkpoint完成后才一次性dev180；所有结果为reused development，非新外部确认。[七问](../../../archive/research_records/2026-09-07_structured_background_aux/IDEA.md)、[预注册](../../../archive/research_records/2026-09-07_structured_background_aux/PREREG.md)。fit RC缓存684例WT byte exact、561例有RC；GPU合成配对、梯度与裁除bit-exact全过，峰值约20.66GiB。实现和TRAIN_PREACCESS已冻结；2026-09-07 06:39 UTC第一seed2026090711的control在GPU0、candidate在GPU1正式启动，初始state SHA完全一致。工作目录`workspace/BraTS/sammed3d_t1ce_wt/structured_background_aux_20260907`；待第一对完成后按原协议跑第二seed2026090712，尚未访问dev性能。

自动接续已由监督器负责（06:49 UTC PID775932/工具会话65466），首pair主PID772686/773051，06:50两臂均15轮。它将按固定顺序完成第二pair→dev评测→RESULT与验证收据；状态见[CHAIN_PROGRESS](../../../archive/research_records/2026-09-07_structured_background_aux/CHAIN_PROGRESS.json)和[实验SESSION](../../../archive/research_records/2026-09-07_structured_background_aux/SESSION.md)。不要重复启动train/evaluate。CHAIN_DONE后还需由接手agent更新catalog/handoff并跑validator；失败先查CHAIN_FAILURE，不默认重启。

08:42 UTC更新：第一seed两臂已完成168轮，control/candidate用时7222.6/7294.2秒、export error均0，全部168轮配对检查和checkpoint重SHA通过。末10轮训练Dice0.785106/0.785573几乎tie；这不是dev结果。第二seed已由监督器启动（control PID804755、candidate PID804756），初值一致；当前继续等待第二pair完成，再统一dev180。见[首pair训练复核](../../../archive/research_records/2026-09-07_structured_background_aux/PAIR1_TRAINING_REVIEW.json)。

额外训练模态可行性审计：fit684目录实际只有T1c与seg，t1n/t2w/t2f全0；历史四模态manifest指向的`/mnt/bn/liveaim/songjunyi/brats2024`在本机不存在，不能据旧6750文件记录假定当前可用。未来多模态教师需先恢复同患者合法源与parity；当前辅助监督不受影响。[文件可用性审计](../../../archive/reviews/2026-09-07_fit_modality_availability/RESULT.md)。

下一独立因素已在当前dev揭示前预注册：患者等概率、患者内保留原生体积相对权重。fit-only概率审计显示患者总概率相差22倍、32人占27%，每轮预计覆盖271人；患者均匀参考299人。新实验只改采样，预先指定复用当前两个native control（aux=0），两候选从相同初值训练168轮；须等当前完整流程结束后才运行，不能按其dev结果改方案。尚无新方法效果。见[审计](../../../archive/reviews/2026-09-07_fit_subject_sampling/RESULT.md)、[七问](../../../archive/research_records/2026-09-07_subject_balanced_sampling/IDEA.md)、[预注册](../../../archive/research_records/2026-09-07_subject_balanced_sampling/PREREG.md)。

下一实验CPU实现已就绪，采样7+归档3测试通过；新GPU smoke/训练尚未运行。当前辅助监督CHAIN_DONE之后，先更新其RESULT解释、catalog/handoff并校验，然后按[新实验SESSION](../../../archive/research_records/2026-09-07_subject_balanced_sampling/SESSION.md)写SOURCE_CANONICAL_CLOSED.json并调用`python -m scripts.run_subject_balanced_workflow`。该新链按smoke→freeze→双候选→dev→归档执行，失败停止；不要修改原正在运行的训练/评测文件。

09:42 UTC续接：监督器会话65466仍live，第二pair已超过82/168轮，开发尚未访问。补充fit-only取舍核查：患者均匀使最小肿瘤四分位抽样质量44.89%→27.44%，预计患者覆盖271→299、扫描覆盖仅476→480；既定方案/门未改。核查3测试通过，catalog/layout通过；workspace剩6.8GB。见[取舍报告](../../../archive/reviews/2026-09-07_fit_subject_sampling/SAMPLER_TRADEOFF.md)。

第二pair跨100轮后已存[训练曲线快照02](../../../archive/research_records/2026-09-07_structured_background_aux/TRAINING_SNAPSHOT_02.png)，control103/candidate102轮，仅fit日志。两项方法dev揭示前另冻结[漏画/多画描述计划](../../../archive/reviews/2026-09-07_development_error_description/PLAN.md)，10项合成测试通过。各方法正式RESULT/VERIFICATION完成后，可分别运行`python -m scripts.describe_reused_development_errors --study auxiliary`或`--study subject_sampling`；只读已有dev计数，0新forward，全队列门不变，按fit固定体积切点输出全部组，不用子组挽救失败门。描述结果仍需文字解释归档，不修改原训练/评测脚本。

用户要求GPT-6继续科研、希望重启goal并强调额度充足。接手时goal paused（累计83,770秒），随后产品已恢复，续接实查active。本轮已从Fable raw独立重算三组均值/Point/绝对及配对CI，并核收据SHA，全部一致；重新计算旧inner排除集合后仍仅2人/3 scans。尚未新增模型或测试访问。

角色调整已获用户批准并执行：原972池排除旧confirmation全部53人/108扫描后为864 scans/391人，固定313/78患者fit/development，结果为684/180 scans。明确称reused development；本实验两臂无教师、matched control、预注册和跨seed复核，其他角色继续保护。批准前提案与审计保留在[接手提案](../../../archive/reviews/2026-09-07_gpt6_reentry/RESULT.md)、[角色修正](../../../archive/reviews/2026-09-07_gpt6_reentry/ROLE_SCOPE_CORRECTION.md)和[机器审计](../../../archive/reviews/2026-09-07_gpt6_reentry/AUDIT.json)。
