# Amendment 01：实现清单补列queue合同

冻结时间：2026-09-05，条件queue启动前、任何新UTSW病例或模型访问前。

首份`EXTERNAL_IMPLEMENTATION.json`已绑定queue代码，但遗漏列出已通过的`tests/test_utsw_no_n4_paired_external_queue.py`。这不改变协议、代码、cohort、门或测试结果，只是实现manifest覆盖不完整。

原收据保留。冻结器改为生成`EXTERNAL_IMPLEMENTATION_AMENDMENT_01.json`，在原18文件基础上加入queue顺序测试，并重新绑定冻结器自身哈希。后续所有入口只接受新收据。

