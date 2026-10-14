# Amendment 05：已运行等待queue与最终implementation的绑定

记录时间：2026-09-05，FAMILY_READOUT前，任何UTSW全量preaccess/新病例/模型访问前。

UTSW queue于20:41:42在Amendment01后启动并保持等待；随后Amendment02–04继续加固下游builder/证据门。当前PID319038，queue文件从启动至今SHA始终为`a541b298683bebe0f85ba868fb5597f54b9a2865e0e85bcaa63b955c2c40d23c`。

运行中协调器真正调用的family/no-N4 eligibility、Full972等待、RHUH precedence、four-hour和idle-GPU控制逻辑没有在Amendment02–04变化：

- Amendment02只改变builder的cache硬门；
- Amendment03给protocol增加child artifact的`docs()`绑定，并给evaluator增加checkpoint训练cache门；
- Amendment04只给builder增加raw label域门。

未来`prepare/smoke/download/build/evaluate`均作为新Python子进程启动，每个入口首先调用当前冻结器，只接受`EXTERNAL_IMPLEMENTATION_AMENDMENT_04.json`。因此不需要中断/重启只等待的协调器；实际访问只能使用最终实现。

本记录时`EXTERNAL_PREACCESS.json`、`SOURCE_DOWNLOAD.json`、`CACHE_RECEIPT.json`及两个全量`/tmp`目录全部不存在，模型/新UTSW访问仍为0。

