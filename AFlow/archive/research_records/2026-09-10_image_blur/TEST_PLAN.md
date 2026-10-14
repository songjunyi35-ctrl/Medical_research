# 条件固定seed11测试面板

必须两seed的模糊开发联合门与独立验证全部通过；固定seed2026090711，不择较好seed。对照是同seed的原患者均衡模型，不是native模型；两者fit684/313，无教师，唯一追加训练图像模糊差异。

同test270/119、原128硬GT/阈值0/最高logit定位点；对照/候选各identity与八镜像，8视图固定逆变换后顺序float32均值，identity复用第一个视图，共4320前向。额外180开发保持选择角色，不作训练，不称full864。

使用resumed_test270_helpers：八镜像候选Scan/Subject均须超过旧overall和同模式对照，Point>=265且不低于对照，相对旧overall两Dice配对患者簇95%CI下界>0才满足严格科学门。1×Scan是否超过旧0.7520031755617648单独报告；所有结果正/负均更新两榜，不按test选择最终变体。bootstrap10000次患者簇、seed2026091032，未校正多比较。

该窗口若另一分支已有TEST_PREACCESS就禁止此面板；当前没有任何test准入。剩余不足10分钟执行/核验余量或计算截止21:24:55临近时不启动。旧confirmation/fixed validation/outer/external不评测；复用内部test非盲、历史test-informed与镜像源限制保留。
