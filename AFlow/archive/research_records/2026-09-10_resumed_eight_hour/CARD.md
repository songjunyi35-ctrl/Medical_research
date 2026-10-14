## 第二个八小时研究结束：未超越旧榜

新窗口2026-09-10 13:39:55—21:39:55 UTC；报告时间2026-09-10T21:40:46.271631+00:00。六个新模型全部固定168轮完成，投入约12.976 GPU小时。旋转与模糊都未过联合开发门，未新增test/outer/confirmation/fixed validation/external性能，不把开发分数冒充超榜。

**一句话/动机：** 分别检验整体小角度旋转和标签不变的轻度图像模糊，针对覆盖与错误最高点不同步；结果没有支持固定配方的稳定总体提升。
**改变/固定：** 各比较只改变登记的增强因素；fit684/313、reused dev180/78、采样层、初始化、基础随机流、168轮、主损失、阈值0、raw-logit argmax和1×部署保持控制。历史对照复用，没有宣称完整旧训练轨迹从头重现。

| 本轮新候选（dev180/78） | Scan Dice | Subject Dice | Point /180 |
|---|---:|---:|---:|
| 2026090711_native_rotation | 0.751434115 | 0.736974074 | 167 |
| 2026090711_subject_rotation | 0.747542506 | 0.733798810 | 171 |
| 2026090712_native_rotation | 0.742949898 | 0.728538202 | 169 |
| 2026090712_subject_rotation | 0.749582832 | 0.736779971 | 159 |
| 2026090711_subject_blur | 0.747191029 | 0.734375709 | 170 |
| 2026090712_subject_blur | 0.752820543 | 0.741177014 | 165 |

**结果与直觉：** 旋转在患者均衡层两seed自身对照Scan差−0.005932/−0.006323，四Dice区间均为负；原采样层符号不稳且点门失败。模糊自身对照Scan差−0.006284/−0.003085、Point+3/−3，未形成双seed收益。失败配方关闭，不调强度/阈值/seed救门；不证明所有数据处理到上限。

**同队列旧榜：** reused internal test270/119 overall Scan仍0.757269662、Subject0.741253585、Point265/270（Full972知识蒸馏模型固定八镜像，8×）；单次前向Scan仍0.752003176、Subject0.737266420、Point252/270（GCSAM184）。这里是项目内部复用队列，不是官方隐藏榜或全球SOTA。
**区间范围：** 新候选只在同一78人的复用开发集评估，两seed不是独立队列；10,000次患者簇配对95%CI未作多重探索校正。完整区间见各RESULT.json，点估计与统计支持分开。

**成本与访问：** 新训练6条/12.976 GPU小时；正式开发共1800前向，其中候选1080、旧模型诊断720，工程smoke另计。候选mask270MiB与旧诊断mask180MiB均已核对；旧诊断类别cache360MiB、新fit类别cache1.336GiB保留在/tmp，六模型及恢复点保留。推理仍4.70M级/1×，当前没有获准新部署模型；全864分支0训练/0测试，原dev未转为训练。

**下一优先级：** 先设计保留不同组织区域标签的匹配对照，原始fit0—4类cache已684/684独立核验，尚未训练区域模型；其次才是更高成本的原生体素块/盲滑窗参照。旧官方训练源字节parity限制保留，不重开保护队列调参。

[本窗口总报告](archive/research_records/2026-09-10_resumed_eight_hour/RESULT.md)、[机器汇总](archive/research_records/2026-09-10_resumed_eight_hour/RESULT.json)、[证据核验](archive/research_records/2026-09-10_resumed_eight_hour/VERIFICATION_RECEIPT.json)、[后续研究判断](archive/research_records/2026-09-10_resumed_eight_hour/RESEARCH_SYNTHESIS.md)、[区域标签工程](archive/research_records/2026-09-10_fit_subregion_labels/RESULT.md)、[旧点记录更正](archive/research_records/2026-09-10_resumed_eight_hour/POINT_RECORD_CORRECTION.md)。
以下带旧时间的运行记录保留历史；本轮所有训练/候选评测均已结束，不重启已关闭入口。
