# 异质数据配方融合结果解释

0.5患者均衡模型与0.5体积占比模型的raw-logit融合，seed11/12 Scan为0.754116/0.750449。相对native对照的Dice区间为正，但seed12低于更强的患者均衡成员，seed11 Subject也未超过两成员最大值。

因此固定组合未通过“超过两成员”的门，不调权重、不挑seed、不进入test。融合0训练、720开发前向、峰值约4.849GiB；无test/confirmation/outer/external访问。