# N4单位敏感性：源码与train-only数值诊断

状态：诊断预案；不改变正在运行的no-N4实验，不读取E/F预测/指标，不读取任何分割标签。

已确认环境SimpleITK2.5.5、ITK5.4.6，和SimpleITK v2.5.5官方SuperBuild绑定一致。此前背景解码工程中，第三份train图只改变线性强度编码便令完整N4输入链最大差0.057679；该等价性方案已经关闭，不放宽门。

诊断在相同3份训练图（Full972 manifest按case_id排序首3）上固定四种条件：原图重复运行两次、原图精确乘0.5、精确乘2。均用原N4 shrink2/4×50 iterations和相同单线程，记录迭代/收敛轨迹、正值支持、校正后标准化128³张量差。重复用于区分运行随机性与单位敏感性；2的幂缩放避免另一次z-score编码的舍入误差成为唯一解释。结果不能选择这轮no-N4方法参数，也不构成RHUH可评测的放行证据。

版本核对已经排除最初的最高档边界怀疑：早先查看的v5.4.0只有idx<E.size()-1分支，但本机对应v5.4.6已明确增加else correctedPixel=E.back()。因此不能把旧版本源码缺口归因为本机的问题，不构建这个已存在的修复，也不启动修复版模型评测。继续做上面固定的重复与倍数缩放数值诊断，以区分确定性、收敛轨迹及单位敏感；不预先认定具体内部步骤。

来源：[SimpleITK v2.5.5依赖](https://github.com/SimpleITK/SimpleITK/blob/v2.5.5/SuperBuild/External_ITK.cmake)、[ITK v5.4.6 N4实现](https://github.com/InsightSoftwareConsortium/ITK/blob/v5.4.6/Modules/Filtering/BiasCorrection/include/itkN4BiasFieldCorrectionImageFilter.hxx)。
