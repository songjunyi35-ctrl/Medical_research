# 同图像外观unary：108完成

4固定项，新增神经前向0；原RACE8+w32base8，部署2网络16forward+每类5EM。全crop GT只评分。

|候选|Scan|Subject|Point|
|---|---:|---:|---:|
|appearance_k1_lambda0.5|0.769895754|0.759519932|99/108|
|appearance_k1_lambda0.25|0.769857194|0.759488069|99/108|
|appearance_k3_lambda0.5|0.769848917|0.759478565|99/108|
|appearance_k3_lambda0.25|0.769847816|0.759480029|99/108|

完整区间/所有变差/患者条件5折/种子计数、回退与trust见RESULT/RAW。强置信错误可污染种子，不能称独立Bayesian后验。本family后不再新开网格，进入原80终态和最终至多2候选闭环。
