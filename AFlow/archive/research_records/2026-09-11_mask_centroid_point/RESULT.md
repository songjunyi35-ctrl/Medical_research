# 预测掩膜几何中心点结果

固定Flip8区域mask不变，只把Point改为预测前景几何中心投影。Dice逐例exact，但Point相对Flip8 seed11下降40、seed12下降30，两个seed均低于identity和预设底线。

因此该无训练点规则关闭，不搜索半径/连通域/tie-break，不访问test。它说明简单几何中心不适合作为本项目自动点生成器。

[七问](IDEA.md)、[预注册](PREREG.md)、[raw](RAW_DEVELOPMENT.json)。
