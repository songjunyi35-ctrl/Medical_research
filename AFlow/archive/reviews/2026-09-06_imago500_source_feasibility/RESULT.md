# IMAGO500来源可行性审计

状态：当前外部WT基准不合格且需要用户/机构DUA；0账号、0患者清单、0影像、0标签、0checkpoint、0模型、0性能。

## 为什么看它

2026-06-11发布的Amsterdam IMAging in GliOma（IMAGO）首版有500名成人弥漫性胶质瘤患者：全部有术前T1平扫、T1对比增强、T2和FLAIR，381人有术后14天内四模态，368/500为IDH-wildtype GBM。阿姆斯特丹机构不在BraTS2024七站，规模和治疗阶段都很有吸引力。

## 标签门

一手论文明确说明：

- 所有mask均由PICTURE术前/术后模型自动生成，使用四个结构序列。
- 术前模型输出0背景、1水肿/非强化、2坏死、3强化；术后模型只输出强化残留。
- 一位有6年经验的神经放射科医师只做“视觉检查并决定接受”；技术验证再次写明，自动mask若不正确就不纳入。论文没有说对错误边界进行人工修订，也没有提供de novo人工ground truth。
- 因此发布的249份术前和190份术后segmentation是“自动预测经人工接受筛选”，不是本项目要求的专家体素真值。用它给另一个自动模型算Dice会把PICTURE的误差当成真相，还引入“只保留自动模型看起来正确病例”的选择偏差。
- 术后mask又只含强化残留，不能构成项目的BraTS治疗后WT=`{NETC,SNFH,ET}`。

一手证据：[Scientific Data论文](https://www.nature.com/articles/s41597-026-07424-8_reference.pdf)第5–8页与[正式页面](https://www.nature.com/articles/s41597-026-07424-8)。

## 访问门

Release 1.0托管于EUCAIM/Health-RI XNAT。论文要求用户注册EUCAIM、请求数据、交给negotiator、签署禁止重识别/未授权转发的Data Usage Agreement（数据使用协议，DUA），再申请Health-RI账号；也可邮件提供姓名、机构、国家、邮箱、项目题目和描述。Codex不能代表用户或机构签署，也未发送邮件。

## 决定

当前路线在metadata层关闭：不下载249个自动术前mask或把“人工接受”改写成“人工修订”。即使用户以后完成DUA，它也只能作为无标签输入域、自动伪标签研究或需要另有人工标签的开发资源；除非发布者提供逐体素人工修订版及明确标记，否则不进入当前外部Leaderboard。

这只否定IMAGO release1.0作为人工WT基准，不否定其影像、临床/分子信息、无监督域研究或未来人工标注版本。

