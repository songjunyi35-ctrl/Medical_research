# 完整轨迹公共inner阶段结果

Candidate在epoch101–200中按Scan选择epoch168。其common inner108 Scan/Subject/Point=`0.732367/0.711816/103/108`；control同epoch=`0.722341/0.702356/103/108`。

Candidate-control Scan=`+0.010026`，95% CI `[+0.001367,+0.017933]`；Subject=`+0.009460`，95% CI `[-0.001959,+0.019206]`。77/108 scans、35/49 subjects改善，Point持平。所有inner promotion与工程门通过，授权full756/full972从随机初始化重训到同一epoch168。

两臂各200轮约4.42小时，peak20.29GiB，初始state SHA exact。Fixed validation108、outer、confirmation、test均未访问。
