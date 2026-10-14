# 预模型实施修订：标签内存布局及工程访问措辞

日期2026-09-05 UTC。新教师/学生尚未构造、未训练、未产生E/F预测或指标。

1. 首次cache构建在原始NIfTI canonical轴翻转产生负stride标签数组时，由torch.as_tensor拒绝而停止。修复只在送入既有label转换前做np.ascontiguousarray，数值、标签定义、几何、插值、候选N4开关和所有门均不变。增加负stride与正stride副本结果完全一致的合同测试。原不完整/tmp/aflow-no-n4-ohu9tw/train保留，不被训练使用；新完整cache写train_v2。
2. 原PREREG中“Fold labels ... remain unopened”字面上与其后要求构建完整972标签cache和独立重建冲突。精确边界为：允许固定数据工程读取train972原始标签以执行不依赖模型的空间转换/字节一致性检查；禁止任何E/F模型预测、准确率、epoch/参数/病例选择或人工查看标签来修改方案，直到所有模型冻结。训练loader及教师仍只读取本fold930 fit标签。
3. 首次失败的多进程构建可能已触读队列中的train角色原始标签，未保存逐文件读取轨迹，因此不虚称E/F原始标签从未在工程中打开；没有E/F模型或性能指标、fixed validation/outer/confirmation/test270/UCSD75访问。数据本来就是历史train池，并非新的盲原始标签，当前证据始终是post-test development。
4. 此修订只澄清工程访问和修复数组布局，不改变方法、split、seed、损失、epoch、阈值、bootstrap或成功/停止门。原IDEA/PREREG保留哈希，后续正式冻结收据同时绑定此修订。
