# Amendment 03：固定3例child证据与训练cache身份加固

冻结时间：2026-09-05，FAMILY_READOUT前，任何UTSW全量preaccess/新病例/模型访问前。

实现审计发现：全量builder用canonical桥接的`COPY_A_QC.json`取得固定3例期望image SHA，但protocol只绑定汇总`INPUT_QC.json`。汇总虽然记录child SHA，builder仍应直接重验child文件。另模型加载已绑定checkpoint/role/seed/epoch/fit，却可再显式验证payload的训练cache index属于对应canonical/no-N4 arm。

修订仅加两项身份门：

1. protocol固定`COPY_A_QC.json` SHA=`a1776d09d7e84a786e3fc27f114ec4119a6f02bf1cdb35102906a3e8876a3665`；builder从该绑定常量路径读取。
2. evaluator加载checkpoint时，`input_cache_index_sha256`必须等于`full.verify_cache(role)`返回的对应arm冻结SHA。

cohort、输入数值、N4、模型权重、评测、门和访问顺序均不变。旧implementation收据保留，新收据为`EXTERNAL_IMPLEMENTATION_AMENDMENT_03.json`。

