# Experiment idea: N4只用自动前景拟合偏置场，保留完整图像

Status: preregistered before implementation, mechanism check, candidate cache/student training, and first E/F model readout. Shared canonical teacher preparation is already running; no paired student has yet trained or accessed E/F metrics.

## 链路审视与启发

[本轮链路审视](../../reviews/2026-09-05_30h_chain_audit/SESSION.md)与[N4原因隔离](../../reviews/2026-09-05_n4_scale_diagnostics/PLAN_02.md)是起点。Brain crop旧协议+0.054250、完整972轨迹对匹配control验证Scan+0.010943提示输入/数据作用，但160³和width24失败不能支持盲目加算力。镜像一致loss下降35–36%而准确率未提高，要求机制与准确率分别过门。现在近零信号的编码精度损失在第三份train图造成22.1%局部相对误差，N4最后一级11→10次迭代、最终128³差0.057679；相同缩放但无float32偏移往返时仅0.0000038。不能把它误称纯比例缩放失败或ITK旧边界bug。

## 1. 现在的问题是什么？

当前N4拟合掩膜是所有正值，包含极弱的脑边缘信号。三份train-only诊断中两份有354/405个小于1的正值；第二份没有这类值且编码误差很小。去掉N4的匹配实验正在准备教师，但仍可能失去有益的亮度场校正。是否只需改变拟合参考区域，尚未验证。

## 2. 一句话直觉

估计照片怎样补光时，只参考更可靠的区域；补光后仍保留整张照片，不裁掉暗组织。

## 3. 为什么选择这个idea？

[SimpleITK官方N4示例](https://simpleitk.readthedocs.io/en/latest/link_N4BiasFieldCorrection_docs.html)在无指定mask时使用OtsuThreshold(image,0,1,200)。Otsu（大津法）通过亮度直方图把背景与较可靠前景分开；它是官方现成配置，不搜索阈值、percentile或侵蚀大小。[ITK说明](https://docs.itk.org/projects/doxygen/en/stable/classitk_1_1N4BiasFieldCorrectionImageFilter.html)提醒负值及小于1的强度可造成差结果。

此前百分位裁剪实际改掉模型输入的强度尾部；这里仅改变N4用于估计偏置场的mask，最终保留全正值支持，不直接裁强度或删标签。原背景解码失败的是不变N4管线的数值等价主张；当前明确改变N4拟合掩膜，绝不冒充原管线等价。这是另一个预注册单因素检验。

## 4. 具体怎么做？

canonical原图→只以原图生成Otsu200 mask→N4依原shrink2/4×50迭代估计场→把场应用到原图所有正值体素→维持原脑包围盒、源非零z-score与128³缩放→同一学生/固定隔离教师的原监督与知识蒸馏目标→阈值0分割与argmax点。

不把Otsu mask当肿瘤标签或最终脑裁剪mask，不把其外的正常组织清零。原图≤0仍按原管线清零；标签WT1/2/3、排除4不变。N4的其他参数和原版SimpleITK2.5.5/ITK5.4.6不改。

## 5. 唯一变化与固定项

唯一变化是N4拟合mask：原正值支持→官方Otsu200自动前景。共享[no-N4预注册](../2026-09-05_no_n4_clean_teacher_replication/PREREG.md)的fold-clean教师、canonical对照、E/F42-case划分、初始学生seed20261313/20261323、原sampling/增强/loss/168轮/阈值与固定Flip8。共享对照是在任何候选学生训练或E/F模型指标前登记，不是看到对照分数后新增比较。

本轮验证家族包括两个候选：移除N4、替换N4拟合mask，二者都与同一canonical对照做单因素比较。E/F是共同保留的开发集合，不能把两个候选比较描述成两份独立患者证据。全部候选完成或按预设工程门停止前不打开E/F；区间为每比较的95%区间，不作未注册的家族显著性声明。

## 6. 成功、失败和停止规则

先机制门：同一固定首3份train图，各自Otsu-N4(original)与Otsu-N4(float32全图z-score再恢复背景)比较；完整标准化128³tensor max误差≤0.001、mean≤0.00001，原正值支持/最终脑bbox完全相同，Otsu拟合mask非空。记录mask一致性但不以少量mask位差代替完整输出门。任一失败关闭，不调200 bins/公式/侵蚀/阈值，不做候选cache/训练或RHUH访问。

机制全过才做972 train cache；全标签文件与canonical SHA完全一致、全部crop geometry一致，首3份独立重建exact，代码合同和双卡2步smoke全过才训练2位候选。两位教师来源对每fold固定为当前no-N4实验的新history/EMA fit930，自己fold42患者从未用于其训练。

性能门与原no-N4相同：每fold、每style(identity/Flip8)，Scan/Subject delta>0，Point delta≥−1/42，患者簇bootstrap95%区间下界均≥−0.003；两fold平均identity/Flip8 Scan delta均>0.001，Subject均>0。10,000患者簇重采样，使用原fold种子20261331/20261332，共享对照及同一重采样索引。任一失败不调参/加seed/重测旧cohort。

通过才允许固定Full972重训并冻结，仍不自动授权外部test。若两个候选都通过，分别保留结论；Full972可共享同一新full教师与同seed canonical对照，按原no-N4优先再Otsu的固定顺序，进入任何尚未开始的完整训练阶段都必须有≥8h窗口。没有时间完成则不启动该阶段，不把StageA误称最终模型。

## 7. 成本与风险

机制检查预计数分钟CPU、0GPU/0模型。若通过，新增9.49GiB缓存放/tmp；双A100新增2学生约3–4h墙钟、6–8 GPU-hours，每卡24GiB硬门，不额外重训已共享的教师/对照。部署仍4.70M、identity1×或固定Flip8 8×；多一次Otsu预处理扫描，无额外模型forward。若N4真正需要全脑暗组织参与，自动mask也可能降低准确率；不能只凭数值更稳就放行。

## 8. 最终结果

待机制检查；没有候选模型、E/F指标或RHUH标签/模型访问。原文保持冻结，结果另存。
