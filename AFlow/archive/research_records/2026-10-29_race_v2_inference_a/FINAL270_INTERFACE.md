# 最终270入口与恢复约定

安静执行入口为`scripts/race_v2_inference_a_20261029/final270_watch.py`，数值核心为保留不改的`final270.py`，独立冻结见FINAL270_ENTRY_FREEZE.json。V1与V2冻结均单独保留，V2只适配root出版接口。CPU门仅使用合成小数组，证明schema文档不能授权执行、等权logit配方支持config=null；没有读取临床270数组或评分。

只等待总控真正的`archive/research_records/2026-10-29_race_v2_stage/FINAL_TEST_SELECTION.json`。必须采用总控schema `aflow.race_v2.final_test270.v1`；先核对训练完整终态DONE、预算、RESULT的SHA和轮数/更新数，108审计证据、最多两个固定候选、源代码和配方。`FINAL_TEST_SELECTION_SCHEMA.json`永远不能触发。选择文件一旦访问就记录SHA，执行各批次和解码时再次核对，不根据270结果改参。

候选models内嵌模型元数据，primary在首。RACE164直接使用root显式登记的旧2160原子manifest路径与SHA，所有实际atom逐SHA/形状/有限值核验。新最终best只在root已登记完成108的固定CPU严格重载handoff后，以正确RACE-v2构造载入；不套w32或固定epoch164加载器。模型manifest路径固定为`final270/models/race_v2_final/ATOMS_READY.json`并与B只读共享。

新模型患者前须通过动态native index画布的batch8/尾6合成门，reserved≤38GiB，然后独占MODEL_PREACCESS一次执行2160病例前向。参照矩阵T R T共轭、有效支撑同步翻转。多个最终配置共享同一模型的8视图，不增加第9次identity。GT只在完整raw READY之后打开，解码API只接logits和影像。

GPU0队列始终使用`/tmp/aflow_race_v2_20261029/locks/worker1154121_gpu0.lock`。w32由B在GPU1负责；A不跨GPU、不重新运行B模型。所有新数组与保存掩膜只放worker1154121/tmp，文件0444、完成的atoms目录0555，不转移临床数组。

失败后不自动重做患者网络前向。PREACCESS存在但raw journal不完整时停止并保留证据；journal完整2160且所有SHA有效时只补原子READY并恢复统计，不运行网络。已有READY可用`--statistics-only`复算固定解码与统计。

先保存canonical候选RAW `final270/candidates/single/RAW_TEST270.json`，在同worker重新打开全部270保存掩膜和整个原始裁剪GT，独立复算整数计数、Dice、Point及聚合指标，生成MASK_AUDIT_SINGLE.json。出版回执MASKS_AUDIT_RECEIPT_SINGLE.json含status=passed、scans=270、all_masks_counts_points_exact=true，并绑定canonical RAW+selection+审计源码SHA。之后才原子发布`final270/RAW_SINGLE.json`，含完整逐例TP/FP/FN、Dice、Point、保存掩膜路径/形状/SHA、candidate_id、selection_sha256及masks_audit_receipt_path/SHA。回执绑定原canonical RAW而非包装RAW，避免循环SHA。完整比较与全部变差病例在候选RESULT.json中，患者整簇10000次bootstrap seed20261029，选择后仅描述性，继承test-tuned历史。
