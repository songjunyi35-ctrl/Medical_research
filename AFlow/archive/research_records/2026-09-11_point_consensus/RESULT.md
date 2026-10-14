# 多视图点共识结果

固定八镜像区域mask完全不变，仅将八个视图的argmax点按正logit支持票数、平均logit打破平局。

seed2026090711：相对identity Point +2，与原Flip8相同；seed2026090712：相对identity −6，但比原Flip8的160恢复到162。区域Scan/Subject Dice与原Flip8逐例完全相同。

第二seed仍未恢复到identity 168，因此Point安全门失败。该规则关闭，不搜索邻域半径、票权、阈值或seed；无test/confirmation/outer/external访问，新增2880次开发前向。

[七问](IDEA.md)、[预注册](PREREG.md)、[raw](RAW_DEVELOPMENT.json)、[开发前收据](DEV_PREACCESS.json)。
