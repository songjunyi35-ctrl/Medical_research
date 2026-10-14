# 结果解释：安全实现的弱形变没有提高本配方的分割表现

四条168轮全部完成后统一dev180/78读出，随后独立重算与GT点核验通过。状态：两固定配方均未过门，不扫概率/位移/epoch/seed；条件性full864形变路线不准入。

| seed/抽样 | 新Scan | 新Subject | Point | 对自身无形变control的Scan差 |
|---|---:|---:|---:|---:|
| 11/native | 0.745530193 | 0.731085915 | 172/180 | −0.001232123 |
| 12/native | 0.744131842 | 0.730979527 | 167/180 | −0.000045576 |
| 11/subject | 0.742229367 | 0.732751825 | 172/180 | −0.011245456 |
| 12/subject | 0.751125554 | 0.739670683 | 165/180 | −0.004780402 |

native两seed扫描CI均跨0，不能说稳定显著变差；但没有达到扫描/患者增益门。Point分别+2/+4，说明定位改善和整块覆盖改善不是一回事。

subject层相对自己的无形变control，两seed扫描CI分别[-.017368,-.005362]、[-.009261,-.000387]；第一seed患者Dice差−.007882、CI[-.012823,-.003119]，第二seed患者差−.003054、CI跨0。Point分别+5/−3。因此这次组合没有保住旧患者抽样的Dice收益，更没有稳定同时改善定位。

关键归因边界：subject_elastic seed12相对native control的Scan/Subject确实为+.006948/+.010353、区间全正，但旧subject-only该seed原本更高。只看它对native就会把原患者抽样收益误记到形变上。本次2×2固定层比较保留了这个负结果，没有挑该seed或换control放行。

机制支持到哪里：场/标签/边界安全、随机序列和精确导出都通过，故不是实现未跑通；原始直觉“弱形状变化提升完整分割”没有得到本配方支持。native定位数有改善，但具体是否因高置信核心、边缘覆盖或插值效应造成，缺少单独因果消融，不能断言。工程安全不等于泛化收益，也不证明所有弹性增强或数据处理到上限。

成本：四新训练合计约8.6265 GPU小时，双卡约4.3小时；峰值20.664GiB，开发720forwards/21.84秒，部署仍4.70M/1×。全部672轮抽样/RNG及形变流独立重放exact，四导出error0。历史80GB control本轮复用，未重新从头训练；本机40GB两旧native模型开发逐例计数/Point parity已复现，但不能扩张为完整旧训练轨迹bit-exact重放。

证据仅同一180扫描/78患者reused development，两seed不是独立外部cohort，CI未多比较校正。没有test270、oldconfirmation、fixed validation、outer或external新模型访问。旧榜overall/1×未变，不能拿这里的开发值直接超榜。

下一步按揭示前方案：已准备的精确体积占比target，保持图像/原损失/原采样和评分硬GT，只在GPU smoke/时间门通过后双seed完整训练；不加入本次elastic、不用本次失败例改参数。原dev仍为dev，未被本轮任何full864训练消费。

证据：[RESULT](RESULT.md)、[机器结果](RESULT.json)、[raw](RAW_DEVELOPMENT.json)、[开发前收据](DEV_PREACCESS.json)、[独立验证](VERIFICATION_RECEIPT.json)、[全训练重放](TRAINING_REPLAY_FINAL.json)。
