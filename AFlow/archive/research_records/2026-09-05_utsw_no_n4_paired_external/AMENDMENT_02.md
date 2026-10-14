# Amendment 02：移除未预注册的逐例image-different硬门

冻结时间：2026-09-05，FAMILY_READOUT前，任何UTSW全量preaccess/新病例/模型访问前。

实现审计发现cache builder把`images_different`放入362/362逐例硬门。PREREG只冻结了：全量label/geometry相同；固定BT0001/2/3按已归档桥接逐例复现（其中image不同）。若某个其余病例的N4数学上恰好输出相同数组，不能把来源/cache判为失败。

修订后：

- 362/362逐例硬门只要求label exact与geometry exact；
- canonical/no-N4完整`images.f32` SHA必须整体不同；
- 固定3例仍须各自与归档canonical/no-N4 image SHA exact，因此其N4差异仍被验证；
- 每例`images_different`继续作为描述字段，不参与未预注册的停止门。

cohort、源、N4、双下载、标签、模型、阈值、视图、bootstrap、性能门和访问顺序均不变。原两份implementation收据保留，新收据为`EXTERNAL_IMPLEMENTATION_AMENDMENT_02.json`。

