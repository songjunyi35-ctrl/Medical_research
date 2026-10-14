# Amendment 04：冻结UTSW三分区标签域

冻结时间：2026-09-05，FAMILY_READOUT前，任何UTSW全量preaccess/新病例/模型访问前。

UTSW论文和FeTS标签约定三个肿瘤区使用1/2/4，固定3例也只出现`{0,1,2,4}`子集。全量builder原本只要求整数/含positive；未知正整数若直接并入WT会掩盖来源异常。

新增来源门：每例raw label值集合必须是`{0,1,2,4}`的子集，仍须含0和至少一个positive。通过后WT定义仍是所有positive的union；没有改标签目标、病例或性能门。任一未知值立即在cache构建中停止。

其他协议/实现不变。新收据为`EXTERNAL_IMPLEMENTATION_AMENDMENT_04.json`。

