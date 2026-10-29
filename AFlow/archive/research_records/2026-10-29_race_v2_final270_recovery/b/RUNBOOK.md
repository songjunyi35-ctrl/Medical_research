# B固定最终融合恢复

原1154121已不在平台list，原/tmp不可用；已冻结FINAL5b74不改。原80训练已完成并保留，root新授权把1154259闲置GPU1给B。CPU严格重载w32e171/SHA通过，CUDA尚未初始化。ROOT common canonical物理锁为唯一入口，不建独立冲突锁。

先新源冻结→完整原生168 batch4/8/尾批合成门→等A本机新CACHE_READY→实际四payload逐SHA和semantic geometry验证→W32八视图270一次2160（PREACCESS排他、attempt journal、不自动重试）。只读等A RACE2160按旧轨迹重建，B逐atom校验其旧SHA。两模型新index相同但不伪称旧index byteSHA，四payload/评分几何严格历史相同。

固定appearance_k1_lambda0.5对原RACE+w32八视图解码；GT只最终全crop scorer，Point最终score.argmax。新RAW在recovery/b，270保存mask独立另进程复数审计齐全才发布RAW_FUSION wrapper，源/配方/原FINAL不变、无搜索。ROOT report读取新recovery包装，不回写旧目录。新增W32=2160，A额外历史RACE重建=2160，总4320如实计，不称独立确认。旧持久PREACCESS缺失不作为绝对无未落盘执行证明。

脚本stage cpu/gate/forward/score/audit/watch均独立；只有watch在无完整raw时按唯一队列正常运行一次。完整raw存在只复核audit/统计，神经PREACCESS存在而READY不全必须停报root，不隐式重试。全部原训练cache/权重/源码冻结保留，大数组只本机/tmp，无临床载荷跨目的地备份。
