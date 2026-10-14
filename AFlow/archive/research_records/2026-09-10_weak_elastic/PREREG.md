# 弱形变双抽样层预注册

2026-09-10，IDEA.md的七问、全部数字/流程/门构成本次预注册，不按新开发结果修订。

四新增臂：seed2026090711/12 × native_elastic/subject_elastic。固定无形变对照分别为structured_background_aux_20260907的control和subject_balanced_sampling_20260907的subject_balanced。对照checkpoint/报告/历史raw SHA由IMPLEMENTATION_FREEZE与TRAIN_PREACCESS绑定。冻结后旧源脚本不修改。

数据使用继承2026-09-07_reused_development_roles/SPLIT.json与AUTHORIZATION.md。fit684训练，dev180仅一次四候选统一性能读出；数据缓存128³沿用nearest-exact标签和旧影像，阈值0、Point为logit argmax，identity1×，无后处理、无交互SAM。每病例保存Dice、Point坐标及命中、交并/体积计数。bootstrap复用evaluate_structured_background_aux.py固定患者簇算法，10,000次，两seed统计随机种子分别2026091011/12；每个对比重置同seed，双侧百分位95%CI。同一78人，不能说两seed是两个独立cohort。

新的评估/冻结/归档入口在首次新dev访问前另绑定SHA；训练逻辑/形变参数必须在首次正式训练前绑定。工程smoke可使用合成及预定fit manifest前8例，不访问dev像素/保护集模型性能。训练重现核验必须检查168轮采样SHA、增强RNG、初态、fixed168、输出state。

失败记录必须区分工程/预算/科学门，说明失败前访问范围。成功只能触发IDEA中的条件性full864 refit新预注册，不能直接挑fit seed上榜。无新test时不新增榜单排名；所有test评测（含失败/未超榜）均更新MD/JSON并验证。
