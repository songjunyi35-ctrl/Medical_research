# 多视图点共识

Status: preregistered_before_new_dev_forward

## 七问
1. 问题：固定Flip8在dev180/78的区域Dice提高，但seed12 Point从168降到160。
2. 直觉：八个镜像各自指出肿瘤位置，选择被最多视图支持的位置，减少单视图尖峰。
3. 依据：上一轮同模型Flip8的Scan/Subject区间全正；失败集中在点定位而非区域掩膜。新规则不改mask、不用GT，属于独立点聚合。
4. 做法：八个还原raw-logit各得到一个候选点；对每个候选点统计其在八个logit中是否为正，先最大支持票数，再以八视图平均logit打破平局；区域仍用原八视图平均logit>0。
5. 唯一变化：Point聚合规则；模型、八视图、mask、阈值、数据和seed固定。
6. 成功：两seed Point不低于identity且达到169/162底线，同时Scan/Subject保持原Flip8值；否则关闭，不搜索邻域半径或权重。
7. 代价：不增加模型前向，需保存/计算八个点和票数；仅reused dev，不进入test除非两seed全门。

当前无结果；不把开发点提升称超榜。