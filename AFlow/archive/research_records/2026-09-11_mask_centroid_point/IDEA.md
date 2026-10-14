# 预测掩膜几何中心点

Status: preregistered_before_new_dev_forward

1. 问题：固定Flip8在两seed提高区域Dice，但seed12 Point从168降到160；区域与最高logit点排序冲突。
2. 直觉：对模型自己预测的前景区域求几何中心，再投影到最近的预测前景体素，避免一个孤立高logit尖峰决定点。
3. 依据：仓库已有CAPG-3D独立中心头概念；当前plain checkpoint没有中心头，不能伪造其输出。本规则是无训练、无GT的预测后处理，针对已观察Point/区域分离，不能借历史中心头收益。
4. 做法：固定Flip8平均logit>0二值mask→所有预测前景体素几何质心→选择欧氏距离最近的前景体素，平分时取线性索引最小；mask、Dice和阈值不变。若mask为空则保留原argmax。
5. 唯一变化：Point生成规则；两个subject模型、两seed、全部八镜像、区域输出和原hard GT固定。
6. 成功：两seed Point不低于identity且达到169/162底线；Scan/Subject逐例与原Flip8 exact；任何门失败立即关闭，不搜连通域/半径/平分规则。
7. 成本：新增0训练；dev2880forwards（复用Flip8 logits计算），额外CPU坐标计算和很小显存；不进入test除非两seed全门。

当前无结果；不把开发Point改善称超榜。