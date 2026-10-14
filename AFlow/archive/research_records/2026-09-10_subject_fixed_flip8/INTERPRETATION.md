# 固定八镜像结果解释

同一患者采样模型的八镜像平均在reused dev180/78提升Scan/Subject Dice：seed11为0.005769/0.005255，seed12为0.003084/0.003244，两个配对Dice区间均为正。

但Point分别为+2和−8；seed12从168降至160，违反预注册安全门。因此不能挑seed11进入test，也不能把开发0.759与旧test0.757比较。点核查720/720 exact；seed12 lost11/gained3，6个lost病例Dice反而提升，说明区域覆盖与最高点不总同步。

0训练、2880次开发前向、峰值约5.175GiB。无test/confirmation/outer/external访问；固定八镜像应用关闭，正向Dice和定位负结果同时保留。