# SegResNet单主线项目清理

用户明确授权删除SAM-Med3D权重及不用的项目材料。按模型血统逐项删除，未按sammed3d目录名称整体删除。

删除1114个权重/派生缓存文件，逻辑大小37.956GiB；磁盘可用空间从1.701增至38.318GiB。

保留全部SegResNet权重（包括当前193和初始参数）、当前N4脑区train972/validation108/test270缓存、原始数据/清单、代码及所有实验指标/报告。旧SAM权重不再可用，历史冻结文件只记录当时的状态；不得自动恢复旧SAM。没有重新训练/测试。

[删除前全路径/大小/SHA](HASHED_INVENTORY.json) · [删除流水](DELETED.jsonl) · [机器收据](RESULT.json)。
