# Cohort freeze amendment 01: Python boolean literal

首次freeze已从哈希绑定inventory计算出`95_Glioma`的95例/190个T1CE+seg路径，但在构造最终access字典时因使用JSON的`false`而非Python的`False`报NameError；输出文件不存在，payload/数组/model均未访问。

唯一修复是六个access布尔值的Python语法大小写。Publisher目录选择、病例ID/路径/大小、排序、排除51例OtherNeoplasms和全部门不变。
