## 答案标签不变的轻度图像模糊：❌ 固定开发联合门未通过

**一句话：** 练习略失焦图像，答案不动，检验是否减少对细碎亮点的依赖。
**出发点：** native开发Scan0.746762/0.744177；患者均衡0.753475/0.755906，但点命中未稳定改善。具体负结果不等于达到上限。
**具体做法：** 仅20%训练图像追加sigma0.5—1高斯模糊，原硬标签字节不变，患者均衡抽样固定。fit684/313，reused dev180/78；原4.70M网络、初始化、主损失、168轮、基础增强随机流、阈值0/argmax/1×固定，无教师。

**结果：**

| 候选 | Scan Dice | Subject Dice | Point /180 |
|---|---:|---:|---:|
| 2026090711_subject_blur | 0.747191029 | 0.734375709 | 170 |
| 2026090712_subject_blur | 0.752820543 | 0.741177014 | 165 |

**是否支持直觉：** 分别看同采样对照的面积变化和点命中；下列结论不由native复合收益替代自身对照。

- 2026090711_subject_blur：自身对照Scan差-0.006284、Subject差-0.006258、Point差+3；至少一项自身对照区间整体为负。失败项：胜过自身患者抽样对照的扫描Dice；胜过自身患者抽样对照的患者Dice；扫描Dice患者簇区间安全；患者Dice区间安全；相对native患者Dice至少增加0.003。

- 2026090712_subject_blur：自身对照Scan差-0.003085、Subject差-0.001548、Point差-3；自身对照的分割变化仍有区间不确定性。失败项：不降低自身患者抽样对照点命中；胜过自身患者抽样对照的扫描Dice；胜过自身患者抽样对照的患者Dice。

同一78人反复复用，两个训练seed不是独立队列；10,000次患者簇配对95%区间不作多重选择校正。工程导出/随机流通过不等于泛化成功。完整各对比区间和原门见RESULT.json；不把正点估计都称显著。

**代价/下一步：** 新训练4.330 GPU小时（单卡占用墙钟之和），最高allocated显存20.664GiB；本次新开发360前向/11.33秒，无新影像cache，部署1×。精确配方关闭，不改参数/门或择seed救结果；不准入其测试分支。

本记录尚无新test/outer/confirmation/fixed validation/external性能。旧同test270/119 overall Scan仍0.757269662（Full972知识蒸馏模型固定八镜像），1次前向Scan仍0.752003176；开发180分数不能与它们直接相减。历史训练源官方字节parity未证明限制保留。

[七问](archive/research_records/2026-09-10_image_blur/IDEA.md)、[结果](archive/research_records/2026-09-10_image_blur/RESULT.md)、[机器结果与CI](archive/research_records/2026-09-10_image_blur/RESULT.json)、[原始开发](archive/research_records/2026-09-10_image_blur/RAW_DEVELOPMENT.json)、[独立验证](archive/research_records/2026-09-10_image_blur/VERIFICATION_RECEIPT.json)。

- [2026090711_subject_blur checkpoint](workspace/BraTS/sammed3d_t1ce_wt/image_blur_20260910/seed2026090711_subject_blur/fixed_epoch168.pth)：SHA `bc9fb2cc7cd574032e7091316ceaa06f1e665a014ab4b30296fbe46733593ba6`。

- [2026090712_subject_blur checkpoint](workspace/BraTS/sammed3d_t1ce_wt/image_blur_20260910/seed2026090712_subject_blur/fixed_epoch168.pth)：SHA `f611c3ba152ae9086bf813266d90167830c7b34cffc3f2a548a96e7dad8915a9`。
