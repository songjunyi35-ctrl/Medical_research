# Source amendment 01: public labeled Training33 only

原计划要求CPTAC39（Training33+Validation6）每例均有T1CE+seg。固定首个Validation目录`BraTS2021_00447`只有flair/t1/t1ce/t2，没有seg，因此39人cohort按门失败；`EXECUTION_01_FAILURE.json`保留。

该失败发生在payload/label-array/model前，只说明Validation6标签仍未公开，不否定Training33的公开seg。新来源定义在再次目录审计前固定：

- 只纳入官方映射`Segmentation Cohort=Training`的33位CPTAC患者；
- 要求33/33各5个正大小NIfTI且唯一t1ce/seg；
- 明确列出Validation6并要求6/6各4影像、0 seg，永久排除；
- 选择完全由公开标签可用性决定，不读像素/模型；
- 通过只冻结source cohort，不自动授权下载或评测。

证据身份为`BraTS2021 public Training33 used as external-to-this-project`，不是原challenge validation/test、不是hidden blind。模型若未来评测，必须从未用BraTS2021/CPTAC训练，并单独预注册Africa-informed假设。
