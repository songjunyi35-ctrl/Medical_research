# 固定患者模型八镜像预注册

IDEA七问、唯一视图组、两seed门和时间边界构成完整计划。使用既有evaluate_t1ce_wt_flip8_tta_replication_inner.flip8_logits，REFLECTION_AXES顺序原样；原logit>0/最高值点/batch8/BF16，源成员只fit684/313且无teacher。

先合成已有反射算子测试与fit8×2模型×8=128forward smoke，之后新DEV_PREACCESS绑定IDEA/PREREG、源码、source raw/checkpoint/cache/角色hash、输出和2880forward预算，两个模型完整后判门。仅1个固定策略，不研究各种镜像子集。固定test seed11不是挑好seed，test仍需单独前收据，最多2160forward两输出；无模型/训练参数修改。

这是基于当前开发结果和历史固定推理证据的新增研究，不称原8小时最初盲计划，也不撤销已失败的单模型/融合门。高层剩余时间安排明确调整，实验内部停止门冻结。保护队列未获准前不访问，八小时截止不延长。
