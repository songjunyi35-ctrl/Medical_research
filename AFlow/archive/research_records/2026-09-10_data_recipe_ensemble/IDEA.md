# Experiment idea: 不同数据训练方式的固定等权输出融合

Status: prospective_new_inference_study_after_constituent_development_disclosure

## 1. 现在的问题是什么？

患者均衡模型两seed dev Scan0.753475/0.755906，Point167/168；占比标签模型0.746659/0.739852，Point175/167。两个单模型配方都已按原联合门关闭。不同指标有不同优点，但预测是否真互补未知；当前尚无任何新test。

## 2. 一句话直觉

让两种不同数据训练方式的模型各投半票，看看能否同时保留较好的区域覆盖和定位，不直接拼接两个指标或改最高点定义。

## 3. 为什么现在选择？

这是看到复用开发结果后产生的新提案，不称事前原计划或独立确认。[nnU-Net官方配置选择](https://raw.githubusercontent.com/MIC-DKFZ/nnUNet/master/nnunetv2/evaluation/find_best_configuration.py)包含验证集模型组合评估，为多模型互补提供范式，不保证本组合收益。旧同配方多seed融合负结果不直接检验两种数据训练表示；但不能因此宣布旧失败模型成功。

## 4. 具体怎么做？

同一128³图像→相同seed的两个冻结模型（subject_balanced、fractional_occupancy）各一次→两个float32 raw logits等权算术平均→原阈值0/平均logit最高点→原hard128 GT评分。无标签进入融合函数，不拼接“一个模型的Dice/另一个模型的Point”，不加后处理。

## 5. 单因素/固定项

唯一新操作=0.5×sampling logits+0.5×occupancy logits。两个模型都只fit684/313、无教师，dev同180/78。固定两seed2026090711/12、epoch168、原输入/硬GT/batch8/BF16、阈值0；只一个成员组合和一个权重，不扫网格，不调温度/seed/阈值。不重新训练、不修改成员。旧negative recipes保持关闭。

## 6. 成功/停止

每seed：融合Scan与Subject严格超过两成员各自最大值；Point至少为两成员最高命中数−1；相对旧nativecontrol Subject≥+.003、Scan>0且两seed平均Scan差≥+.003；相对两个成员的两Dice10k患者簇95%CI下界均≥−.003。bootstrap seeds2026091111/12，未多比较校正。baseline逐例原始输出须与已保存各自raw exact，工程先合成/fit8 smoke。任一失败立即关闭，不换权重/成员/seed；不能只借患者均衡原收益来放行。

如双seed全门过且剩余时间足够，事前固定seed11允许另冻结一个test270面板：两个成员及融合各identity/固定八镜像，六输出全报；成员1×/8×，融合2×/16×。实际只需两成员各8×共4320forwards，融合不再调用模型。整个8h仍最多一个final test panel，此前没有test。test前必须另绑定全部checkpoint/报告/脚本/data/output/metrics/CI，时间不足则不test、不临时压缩门。

## 7. 成本/风险

新训练0；dev两seed×两成员×180=720模型forwards，融合0额外forward；GPU预计数分钟，部署2个4.70M模型/2×或16×，不是1×单模型。主要风险是输出相关、logit尺度不同或平均削弱较强成员；旧单模型失败结论不改变。没有新大cache，预注册后一次readout。

## 8. 当前结果

未实施/评测，无融合收益证据。截止仍11:28:19 UTC。
