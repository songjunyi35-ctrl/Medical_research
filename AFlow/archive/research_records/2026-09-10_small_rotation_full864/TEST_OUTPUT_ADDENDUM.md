# 事前固定测试输出与精确门

未访问新test像素/前向。只有新旋转主开发、全864训练与独立工程门通过，才另写TEST_PREACCESS。

沿用旧test270/119的原硬128标签、float32 Dice计数、患者簇bootstrap10000次/seed2026091032。两模型各identity/8镜像，按原固定顺序float32累加，阈值0，Point为各输出raw logit argmax。存四组二值预测mask270MiB，独立重算交并/体积/Dice、保存点命中及CI；不保存全logits，不声称独立重算argmax。所有四行（含负结果）入两榜。

本次使用resumed_test270_helpers，不沿用旧weak_elastic helper的264点宽容门。精确新科学门：candidate Flip8 Scan>旧overall且>matched control；Subject>旧overall且>matched control；Point>=265且>=matched control；相对旧overall的Scan及Subject配对95%CI下界均>0。1×Scan是否>旧0.7520031755617648单独描述，不作为选择视图或科学门替代项。代码单测必须验证264失败、CI恰好0失败、Subject/Point不得被Scan补偿。

比较背景仍是复用非盲内部test、历史test-informed流程，不称新的独立外部证据。失败不重新测试/救门；没有准入就没有test结果。
