# CFB-GBM v3外部来源可行性：GTV不等于WT

状态：❌ metadata-only target语义门关闭；0 manifest/NIfTI/label/model/GPU。

## 1. 现在的问题是什么？

WAW152已补一份新人工外部结果，但仍需寻找真正治疗阶段接近、未见的全肿瘤（WT）队列。CFB-GBM v3于2026-06更新，264名按Stupp方案治疗的胶质母细胞瘤患者、术前/术后MRI与347个肿瘤轮廓，看起来可能补足这一缺口。

## 2. 一句话直觉

放疗医生画的“要照射的核心靶心”不是分割任务要求的“连同周围FLAIR异常在内的整片范围”；答案纸很专业，但题目不同。

## 3. 为什么审计它？

它来自法国Centre François Baclesse，机构不同；有T1Gd、多时间点、临床常规放疗医生轮廓，且2026 v3新增347 GTV，现有MOTUM/UTSW等失败未覆盖它。官方TCIA页面和发布者GitHub能直接说明标签生成，不需要先下载208GB主体。

## 4. 具体查了什么？

只读TCIA v3页面、发布者segmentation README及v2.0论文摘要。官方写明GTV由Raystation/Precision/Eclipse治疗计划软件中的radiotherapist按放疗指南勾画；v3新增347个binary GTV。增强补全还使用BraTS2021预训练并在CFB真值上微调的nnU-Net，结果由五位放疗肿瘤科医师验证。标签目标是contrast-enhancing tumor与necrotic core合并的GTV，不包含本项目WT要求的non-enhancing FLAIR abnormality。

## 5. 只改变什么，哪些不变？

没有模型方法；只新增一个来源语义审计。当前WT=`{necrotic/nonenhancing core, FLAIR abnormality, enhancing tumor}`、T1ce-only、阈值和榜单口径不变，不能为取得大队列把GTV改名为WT。

## 6. 怎样算成功/失败？

只有逐例提供与BraTS WT同义的专家标签、可在像素前冻结病例，并有可审计选择性下载时才允许另立协议。官方明确仅GTV，首要语义门失败；208GB Aspera、部分自动补全和无患者桥是次要限制。停止且不下载。

## 7. 代价和下一决定

0数据主体、0标签、0GPU/forward；仅公开网页。当前WT任务永久关闭CFB-GTV路线。若未来目标明确改成radiotherapy GTV，应建完全独立任务、标签定义和榜单，不复用本项目WT数字。

## 一手证据

- [TCIA CFB-GBM v3](https://www.cancerimagingarchive.net/collection/cfb-gbm/)
- [发布者segmentation README](https://github.com/AurelienCD/CFB-GBM/blob/main/segmentation/README.md)
- [CFB-GBM v2.0论文](https://arxiv.org/abs/2608.17884)
