# Experiment idea: RHUH发布态影像的Otsu-N4输入桥接

Status: preregistered input-only feasibility; no labels or model inference

## 1. 现在的问题是什么？

RHUH后两个时点80 scans/40 subjects的官方T1ce已通过MD5/大小与no-N4输入桥接，但当前E/F还有两个独立候选：no-N4和只把N4拟合mask改为Otsu200。若最后只有Otsu候选过门，不能看到性能结果后才临时决定RHUH如何预处理。历史正值mask N4对编码敏感，最坏完整链误差0.057679；RHUH标签未下载且模型仍0 forward。

## 2. 一句话直觉

先把黑背景恢复，再让“自动补光”只参考可靠脑组织，就像调灯时看屋内主体，不让最暗的墙角主导曝光。

## 3. 为什么选择这个idea？

- 已冻结的train-only机制实验中，OtsuThreshold(full image,0,1,200)拟合mask把最敏感样例的编码往返完整链误差从0.057679降到0.000021，3/3机制门通过；说明它与失败的正值mask N4不同，旧负结果没有否定它。
- SimpleITK官方N4示例使用Otsu前景mask。RHUH发布者官方commit `46cf3825278243036c08131b62dcfcfb1d3bba39`说明发布图已SynthStrip+CaPTK z-score且未做N4；若Otsu模型胜出，需在不看标签/模型的前提下确认其训练时输入配方可以确定地用于此发布态。
- 这不是以RHUH选择Otsu。现在只做无标签工程QC；未来模型资格完全由未揭示E/F及条件Full972决定。
- 一手来源：[SimpleITK N4示例](https://simpleitk.readthedocs.io/en/master/link_N4BiasFieldCorrection_docs.html)、RHUH[官方预处理仓库](https://github.com/smcch/RHUH-GBM-dataset-MRI-preprocessing/tree/46cf3825278243036c08131b62dcfcfb1d3bba39)。

## 4. 具体怎么做？

每份官方T1ce先重算SHA并canonical RAS，从六面边界众数恢复黑背景为0。随后固定用Otsu200从完整图估计N4拟合mask，shrink2、四级各50迭代；mask只用于估计平滑bias field，校正场应用到完整正支持，非正值保持0。再用image-only正支持bbox裁脑、源正值population z-score、三线性缩到128³并最近邻脑mask清零。只保存哈希/QC，不保存图像cache、不加载标签或模型。

若以后获准，Otsu Full972模型接收同一输入，固定Flip8 raw-logit平均、阈值0、argmax Point；RHUH Whole Tumor定义为所有positive published tumor labels的union。

## 5. 唯一变化与固定项

- 唯一变化：相对已通过的RHUH no-N4桥接，在背景解码后加入冻结Otsu200-fit N4。
- 固定不变：80 scans/40 subjects/时点1和2、官方身份、背景算法、canonical orientation、完整正支持、image-only crop、源z-score、128³、未来模型/阈值/Flip8/无SAM/无后处理。标签不参与输入选择。

## 6. 成功、失败和停止规则

- Primary工程指标：80/80官方SHA；背景门；Otsu mask非空且只含正输入；N4后支持exact、finite；bbox/标准化/128³输出全部通过；固定首3例image/mask/Otsu-mask SHA独立重建exact。
- Promotion gate：80例所有门和3例重建全过，才保留为Otsu候选的RHUH输入桥接。
- Failure/stop rule：任一失败即模型前关闭，不调Otsu bins、N4层数/迭代、背景门、病例或插值。
- 是否允许进入test：当前不允许任何标签/模型/指标。未来仅当Otsu E/F全部门通过并完成条件Full972，另写preaccess后才可下载全部80标签并一次性评固定Flip8；不授权其他旧cohort。

## 7. 成本与风险

- 训练时间：0。
- 峰值显存：0 GPU；CPU多进程≤20分钟。
- 磁盘：不建cache，只写KB级报告。
- 推理forwards：当前0；未来若授权为640。
- 最大技术风险：在RHUH已z-score图上再做N4可能产生与BraTS raw→Otsu-N4不同的域效应；工程门不证明准确。Otsu/N4 CPU也比no-N4复杂。

## 8. 最终结果（完成后填写）

- Validation：等待input-only QC。
- Test（如授权）：未授权。
- 置信区间/病例分布：无模型指标。
- 是否支持原直觉：待QC。
- 学到了什么：待填写。
- 下一步：通过也只等待Otsu Stage A与Full972资格。
- 报告/归档/哈希：待填写。
