# 患者均衡采样执行记录

- 当前辅助监督实验的dev尚未揭示时，完成fit概率审计：患者总概率max/min21.996、最高32人质量27.012%、每轮预计覆盖271.023人；患者均匀算术参考298.866人。只读fit metadata，不改运行中的训练。
- 七问/PREREG/DESIGN_FREEZE先于源dev访问：固定每患者权重和1、患者内原生体积比值不变，2个同seed候选168轮、复用预先指定native controls、aux=0；不取决于源实验性能。
- CPU采样7测试和新归档3测试通过；复用统计归档原3测试也通过。新wrapper只在新Python进程替换旧训练入口的采样函数与协议目录，旧脚本文件不变；subject_balanced arm让旧训练入口aux系数保持0。
- GPU smoke和正式训练尚未运行。必须等源辅助监督CHAIN_DONE，完成其catalog/handoff更新，再写SOURCE_CANONICAL_CLOSED.json（status=source_results_archived_and_canonical_docs_updated，source_result_sha256为源RESULT.json实际SHA），才能调用python -m scripts.run_subject_balanced_workflow。
- 新runner将依次smoke→admission freeze→双卡两候选→一次性dev360新forwards→独立重算归档。需要根文档收口时看CHAIN_DONE的canonical_document_update_pending；不要重复运行已开始的链。
- 新增模块整合后全仓回归：779 passed、27条既有CUDA沙箱/SciPy弃用warnings、0失败（41.70秒）。当前源辅助监督第二pair继续训练；本新workflow仍未启动GPU或开发访问。
- 09:38 UTC新增fit-only抽样取舍核查：最小体积四分位质量44.89%→27.44%，预期不同扫描476.36→479.78；证据见../../reviews/2026-09-07_fit_subject_sampling/SAMPLER_TRADEOFF.md。该描述性核查3测试通过；初版顺序不变性测试因约1e−16浮点求和差失败，修正测试数值比较后通过，未改训练公式/门或冻结代码。源实验此时无DEV_PREACCESS。
- 源辅助监督10:43完成，双seed效果门失败但所有工程/归档门通过。RESULT、catalog/handoff更新与三validator通过后写SOURCE_CANONICAL_CLOSED，源RESULT SHA=a59c919f358890cfe1f7552796d8259b0088cff99522494489dadfa1ecf62c1e。没有按该负结果修改本方案。
- 10:47:03 UTC正式启动workflow，会话12286、监督器PID840708；合成GPU smoke全部通过，TRAIN_PREACCESS冻结，两seed候选随后在双A100同时训练。原监督器65466已exit0，不再轮询。当前新workflow自动接续训练→dev360新forward→归档；不要重复启动。
- 约前21轮完成后，独立numpy公式重建native和patient-equal权重，再用独立CPU生成器连续972 draws、保留前968索引；两seed/两策略21轮SHA均exact。患者覆盖约271→299–300，小体积抽样约45%→27–28%；只是训练机制验证，不横比不同采样分布的fit Dice。工具3合成测试通过，报告archive/reviews/2026-09-07_fit_subject_sampling/SAMPLER_REPLAY_SNAPSHOT.md。
- 最终除原workflow外，还需人工运行`python -m scripts.audit_subject_sampler_replay --final`（全168轮）、`python -m scripts.describe_reused_development_errors --study subject_sampling`（0新forward计数描述）、`python -m scripts.audit_saved_development_points --study subject_sampling`（重复读取已授权dev标签核点，不跑模型），再做结果解释、canonical更新与校验。
- 训练约116轮后完成所有补充核查工具的全仓回归：806 passed、27既有CUDA沙箱/SciPy弃用warnings、41.81秒、0失败（工具会话3960 exit0）。正式训练/评测代码未改，新候选dev仍未访问；无需仅因继续等待而重复跑全仓测试。
- 12:49:32 UTC workflow会话12286正常exit0，两候选168轮全部完成，dev360新forward/12.12秒，control旧行exact复用。两seed Scan差+0.006713/+0.011729、Subject差+0.007420/+0.013406，四个配对CI都高于0；Point−3/+5，唯一失败项seed11 Point安全门。原联合gate保持false，关闭当前精确策略的晋级，不挑seed12、不平均Point放行，不调mix/temperature/epoch/seed。
- 完整168轮独立重建通过：两个seed/两策略全部有效索引SHA与权重SHA exact；患者覆盖约271→299、小体积抽样45%→27%。预定分组八个Scan差均正，seed11最小组Point−3、seed12同组+3；不是所有小病灶一致受损。count-only描述与720条点/真值标签核查均完成，0新forward。
- 两张A100经只读查询均0MiB/0%，无计算进程。原30hGoal仍active（12:50累计104,543秒）；当前进入最终解释/归档核验，不能因训练结束提前宣称30h完成。
