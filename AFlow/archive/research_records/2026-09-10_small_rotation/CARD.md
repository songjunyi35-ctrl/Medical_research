## 小角度输入网格旋转：❌ 固定开发联合门未通过

**一句话：** 训练时轻微转动整个输入，检验朝向鲁棒性。
**出发点：** native开发Scan0.746762/0.744177；患者均衡0.753475/0.755906，但点命中未稳定改善。具体负结果不等于达到上限。
**具体做法：** 图像/标签共同追加p0.2、每轴±15°输入网格旋转；原采样层内其他因素固定。fit684/313，reused dev180/78；原4.70M网络、初始化、主损失、168轮、基础增强随机流、阈值0/argmax/1×固定，无教师。

**结果：**

| 候选 | Scan Dice | Subject Dice | Point /180 |
|---|---:|---:|---:|
| 2026090711_native_rotation | 0.751434115 | 0.736974074 | 167 |
| 2026090711_subject_rotation | 0.747542506 | 0.733798810 | 171 |
| 2026090712_native_rotation | 0.742949898 | 0.728538202 | 169 |
| 2026090712_subject_rotation | 0.749582832 | 0.736779971 | 159 |

**是否支持直觉：** 分别看同采样对照的面积变化和点命中；下列结论不由native复合收益替代自身对照。

- 2026090711_native_rotation：自身对照Scan差+0.004672、Subject差+0.003761、Point差-3；自身对照的分割变化仍有区间不确定性。失败项：相对native点命中减少不超过1例。

- 2026090711_subject_rotation：自身对照Scan差-0.005932、Subject差-0.006835、Point差+4；至少一项自身对照区间整体为负。失败项：胜过自身患者抽样对照的扫描Dice；胜过自身患者抽样对照的患者Dice；扫描Dice患者簇区间安全；患者Dice区间安全；相对native患者Dice至少增加0.003。

- 2026090712_native_rotation：自身对照Scan差-0.001228、Subject差-0.000780、Point差+6；自身对照的分割变化仍有区间不确定性。失败项：扫描Dice患者簇区间安全；相对native扫描Dice增加；患者Dice区间安全；相对native患者Dice至少增加0.003。

- 2026090712_subject_rotation：自身对照Scan差-0.006323、Subject差-0.005945、Point差-9；至少一项自身对照区间整体为负。失败项：不降低自身患者抽样对照点命中；胜过自身患者抽样对照的扫描Dice；胜过自身患者抽样对照的患者Dice；相对native点命中减少不超过1例。

同一78人反复复用，两个训练seed不是独立队列；10,000次患者簇配对95%区间不作多重选择校正。工程导出/随机流通过不等于泛化成功。完整各对比区间和原门见RESULT.json；不把正点估计都称显著。

**代价/下一步：** 新训练8.646 GPU小时（单卡占用墙钟之和），最高allocated显存20.695GiB；本次新开发720前向/22.62秒，无新影像cache，部署1×。精确配方关闭，不改参数/门或择seed救结果；不准入其测试分支。

本记录尚无新test/outer/confirmation/fixed validation/external性能。旧同test270/119 overall Scan仍0.757269662（Full972知识蒸馏模型固定八镜像），1次前向Scan仍0.752003176；开发180分数不能与它们直接相减。历史训练源官方字节parity未证明限制保留。

[七问](archive/research_records/2026-09-10_small_rotation/IDEA.md)、[结果](archive/research_records/2026-09-10_small_rotation/RESULT.md)、[机器结果与CI](archive/research_records/2026-09-10_small_rotation/RESULT.json)、[原始开发](archive/research_records/2026-09-10_small_rotation/RAW_DEVELOPMENT.json)、[独立验证](archive/research_records/2026-09-10_small_rotation/VERIFICATION_RECEIPT.json)。

- [2026090711_native_rotation checkpoint](workspace/BraTS/sammed3d_t1ce_wt/small_rotation_20260910/seed2026090711_native_rotation/fixed_epoch168.pth)：SHA `3dd1460fec01dc6c919f9666c0e8f8546bf645050eab7c90f8186c3defcafab9`。

- [2026090711_subject_rotation checkpoint](workspace/BraTS/sammed3d_t1ce_wt/small_rotation_20260910/seed2026090711_subject_rotation/fixed_epoch168.pth)：SHA `760b165160ad1bbe586024ae246916a4560d51652efd4835e3c114cf0242166a`。

- [2026090712_native_rotation checkpoint](workspace/BraTS/sammed3d_t1ce_wt/small_rotation_20260910/seed2026090712_native_rotation/fixed_epoch168.pth)：SHA `00fed7d67f5353d2fa8d7a81ecd2be06ea277ca6bb3949bc1954bda3d97898c3`。

- [2026090712_subject_rotation checkpoint](workspace/BraTS/sammed3d_t1ce_wt/small_rotation_20260910/seed2026090712_subject_rotation/fixed_epoch168.pth)：SHA `1c7d451141e889f8ff0684a57ac12b4a5ef391482cbbe2933dae9002eaf3de45`。
