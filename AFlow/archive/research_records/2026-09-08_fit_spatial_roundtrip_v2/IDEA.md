# Experiment idea: 原生体素坐标下的训练标签保真诊断

Status: amended diagnostic, before V2 array access. V1 stopped for unit declarations, not fidelity findings.

## 1. 现在的问题是什么？

fit684/313的脑区缩成128³，几何变形和细节保留尚未量化。V1发现单位声明不统一：14例影像/标签都unknown，不能把所有距离当毫米。

## 2. 一句话直觉

把答案缩小再放大，用所有病例都有的原生像素格子测误差，不把缺单位的格子强称毫米。

## 3. 为什么选择这个idea？

全684 header的数值shape/affine相同、轴长均1，但单位缺失；计数、Dice和连接关系不需要毫米声明。旧160³模型负结果不回答当前表示保真。V1失败与本修订均保留，未查看低保真结果挑方法。

## 4. 具体怎么做？

fit原生WT标签按旧image-only crop与nearest-exact128重建并核cache exact；最近邻/三线性>0.5分别还原到原生数组，报告Dice、体积、以原生体素为单位的边界距离及26邻接组件追踪。只读影像header、不读影像像素；无模型。

## 5. 唯一变化与固定项

相对V1只统一表面计量单位并完善访问日志；所有重采样、阈值、病例、标签定义及组件算法不变。原core不改，新目录输出；dev/validation/outer/confirmation/test/external数组或性能均不访问。

## 6. 成功、失败和停止规则

全684缓存byte parity、身份/数值几何、首例跨进程重复须通过；发现裁剪/取样损失不排例、不停止描述。表面距离只称native voxel，原单位原样列出；无bootstrap或模型晋级。不自动恢复V1，不改变任何已关闭模型协议。

## 7. 成本与风险

4 CPU workers、30分钟与64MiB报告限额，0 GPU/模型forward、不建大cache。连通块不是临床病灶，GT往返不是模型上限；访问日志记录预取，失败保留partial。

## 8. 最终结果

V2待执行；V1已保存2行，额外预取访问仅可确定限于fit集合。最终结果另存，不覆盖冻结七问。
