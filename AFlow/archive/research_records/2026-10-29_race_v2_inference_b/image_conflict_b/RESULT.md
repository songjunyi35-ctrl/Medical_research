# 对侧图像证据冲突裁决：108完成

固定8配置，2模型/16前向；复用缓存新增神经前向0。单T1ce正向强度差不等同实证强化。

|配置|Scan|Subject|Point|
|---|---:|---:|---:|
|contra_absolute_k11_b0.2|0.769754121|0.759556318|98/108|
|contra_absolute_k11_b0.1|0.769679136|0.759401420|98/108|
|contra_absolute_k5_b0.1|0.769604095|0.759339650|98/108|
|contra_absolute_k5_b0.2|0.769602545|0.759396919|98/108|
|contra_positive_difference_k11_b0.1|0.769283602|0.758927641|98/108|
|contra_positive_difference_k5_b0.1|0.769273137|0.758994812|98/108|
|contra_positive_difference_k11_b0.2|0.768846745|0.758455481|98/108|
|contra_positive_difference_k5_b0.2|0.768777485|0.758547501|98/108|

8新+2冻结参考的五折条件稳定性Scan 0.769323766。完整配对Scan/Subject区间、全部变差病例、冲突/有效脑/病灶/边界内trust与权重改变量在RESULT/RAW。GT仅最终评分。无270访问，无独立泛化确认。
