# Amendment 001：独立cache复算固定SimpleITK单线程

时间：2026-09-03 UTC

状态：在任何模型构造、GPU smoke、训练或train-only inner指标访问前冻结

## 发生了什么

V1 cache builder的24个构建worker均由`_worker_init`显式设置`OMP/ITK/MKL/OPENBLAS=1`并调用`SimpleITK.ProcessObject_SetGlobalDefaultNumberOfThreads(1)`。972/972主构建完成，候选整文件标签SHA已通过与Control一致门，并写出cache index。

随后预注册的5例独立复算在第一例`BraTS-GLI-02114-100`停止：image未逐字节一致。检查发现V1 `verify()`在父进程调用`prepare_row()`前没有复用worker的SimpleITK单线程设置，因此比较了“单线程N4构建”与“父进程默认多线程N4复算”。

## 只读诊断

在独立进程显式设置相同四个线程环境变量并调用SimpleITK全局单线程后，重新处理同一train-role病例：

- cache slot：137；
- image exact：true；最大绝对差`0.0`；差异体素`0`；
- label exact：true。

这证明主cache由冻结单线程worker正确生成；失败边界是验证器环境不匹配，不是百分位公式、N4构建、病例、标签或cache内容错误。未构造模型，未访问任何inner metric/fixed validation/outer/confirmation/test270/UCSD75。

## 修正范围

只在`verify()`开头加入与`_worker_init`完全相同的四个线程环境变量及SimpleITK全局单线程调用。以下全部不变：

- raw输入、972病例顺序、N4 shrink2/4×50；
- brain crop、NumPy linear 1st/99th percentile、clip后总体z-score与128³插值；
- 已生成的cache数据与index；
- split、seed、epoch、模型、teacher、loss、门槛和访问边界；
- trainer/evaluator/freezer/runner hashes。

不重建主cache，因为其24个worker从开始到结束已使用正确单线程合同；修正后对该冻结cache重新执行原定seed20261110随机5例独立逐字节复算。若任一失败，方法在模型前停止。
