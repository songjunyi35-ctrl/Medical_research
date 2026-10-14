# Amendment 01：外部病例ID适配

冻结时间：2026-09-05T19:33:23Z，第二次转换前。

首次执行在BT0001源图/标签读取和冻结的finite/nonnegative/geometry检查之后，被共享`_case_id`的BraTS专用正则拒绝；该函数源码只验证字符串并原样返回，不参与标准化、crop、resize、标签映射或任何随机过程。没有`INPUT_QC.json`、模型、GPU或旧cohort访问。

本修订唯一改变：允许固定`BT0001/BT0002/BT0003`三个外部ID绕过BraTS命名正则并保持原字符串；其他ID仍调用原validator。通过独立wrapper在运行时注入，不修改或覆盖原冻结脚本。

输入图、全网格`r`标签、双副本、all-positive→WT、no-N4、bbox/z-score/128³、所有成功/停止门完全不变。该修订不得扩展到其他UTSW病例。

