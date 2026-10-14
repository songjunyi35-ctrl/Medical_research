# 条件性test输出审计补记（本候选训练/开发/测试前）

本分支若两seed通过且整个session尚无test panel，只对事前固定seed11的control/candidate各identity/八镜像评一次。新增二值mask打包存档与另一条件分支相同：C序/little bitorder，四文件合270MiB，index和各文件SHA绑定，0新增forward，仍4320。

只增强结果核验，不改变模型、阈值、镜像、选择或停止门。预分配未写区不是预测，失败必须记录已写case数。原始logits不保存，因此从mask独立核算Dice，不声称独立重算argmax。

本项目历史测试结果及基线流水线已经已知；新候选只在reused development选择，不用本次新test panel调参数。必须明确历史test-informed、reused/nonblind/nonexternal身份，不把“本次未调参”扩写成整个历史从未受test影响。
