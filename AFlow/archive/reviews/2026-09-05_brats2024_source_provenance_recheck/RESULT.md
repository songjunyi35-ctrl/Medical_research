# BraTS2024训练源官方archive parity复查

## 结论

官方corrected-v2 archive与本地Kaggle version-2三包的字节parity仍未解决。当前开发训练的执行依据是2026-07-22用户明确覆盖development fit/tune阻塞，不是官方archive已经核验通过。因此所有当前内部结果必须保留“本地Kaggle镜像来源、未证明与官方corrected-v2逐字节一致”的provenance限制。

## 当前官方状态

- Synapse实体仍为`syn60086071` version2，文件名`BraTS2024-BraTS-GLI-TrainingData.zip`，modified `2024-06-17T21:24:56.161Z`，file handle `143397620`；2026-09-05匿名实体响应SHA仍是历史冻结的`b79d94ef…`。
- 匿名文件请求返回HTTP403：存在未满足的access requirements。
- 合法分页查询得到唯一requirement9606685 version3，名称“BraTS 2024 Post-Challenge Data Access”，类型`SelfSignAccessRequirement`、控制DOWNLOAD；不要求certified user或validated profile，但必须由登录用户自行同意。
- 本机没有Synapse CLI或`synapseclient`；本会话不能代用户登录/自签，也没有尝试创建approval。

## 本地状态与影响

历史状态文件记录本地三份Kaggle version2镜像共1350 scans/613 subjects/6750文件，只能证明镜像下载完成，不能证明官方zip/member parity。官方zip历史冻结MD5为`1d910b17d6cd32e38aa6296b8dfb7c77`。

2026-07-22用户明确允许在development fit/tune边界覆盖旧`authorizes_formal_training=false`，同时保留来源限制为论文provenance说明；confirmatory/test并未因此自动授权。当前E/F、条件Full972均应描述为镜像来源的development，不得称官方corrected-v2字节确认。

RHUH、UTSW等外部TCIA来源有各自独立官方清单/传输证据，不由这个BraTS训练源缺口自动否定；但它们的模型checkpoint仍继承训练源provenance caveat。

## 怎样真正解决

需要用户在Synapse登录后自行同意requirement9606685，并提供可用的授权下载环境；随后取得官方version2 archive、核对MD5，再运行现有`verify_brats2024_source_provenance.py`做archive-to-local逐文件SHA-256比较。未完成前不能改写状态为passed。

## 成本与访问

本复查只读数KB官方JSON：0 archive主体、0图像/标签体素、0GPU、0模型、0指标。

## 证据

- [机器结果](RESULT.json)
- [实体响应](source_metadata/ENTITY.json)
- [requirement列表](source_metadata/ACCESS_REQUIREMENTS.json)
- [requirement9606685](source_metadata/ACCESS_REQUIREMENT_9606685.json)
- [匿名下载错误](source_metadata/DOWNLOAD_ERROR.json)
- 历史状态：`workspace/BraTS/training_protocols/brats2024_source_provenance_status_v1.json`
- 现有核验器：`scripts/verify_brats2024_source_provenance.py`

