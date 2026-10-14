# V1诊断停止：单位声明检查过严

状态：工程计量检查停止，未完成全684例诊断，不作全队列保真结论。固定首例通过，正式顺序已保存2行；第3例标签未显式声明mm，触发原入口单位检查。

四进程预取可能已访问其他fit标签，旧入口未记录每个访问开始事件，不能说只访问了2例。范围仍严格为fit684，0开发/测试/外部数组或性能、0模型forward。原脚本、PREACCESS、FIRST_CASE、部分RAW_CASES和FAILURE保留，不覆盖。

后续header-only检查发现：(mm,mm)501，(mm,unknown)123，(unknown,mm)46，(unknown,unknown)14；全部canonical shape与数值affine一致、数值轴步长(1,1,1)。未知单位不是标签错误，但不能自动当成毫米。

采取已说明的[V2单位修订](../2026-09-08_fit_spatial_roundtrip_v2/AMENDMENT.md)：完整队列用原生体素坐标距离，保留所有Dice/体积/组件算法，加逐worker访问日志。不是根据结果换算子或删病例。见[失败收据](FAILURE.json)、[header审计](HEADER_UNITS_AUDIT.json)、[V2协议](../2026-09-08_fit_spatial_roundtrip_v2/PREREG.md)。
