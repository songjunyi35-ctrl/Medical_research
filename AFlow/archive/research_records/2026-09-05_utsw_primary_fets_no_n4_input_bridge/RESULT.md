# UTSW官方主FeTS图no-N4输入桥接结果

## 结论

固定3例的主FeTS no-N4输入桥接通过：BT0001/2/3两份独立官方副本生成的128³图像、WT标签和crop geometry均3/3 exact，所有源/输出数值门通过。这证明输入配方可确定地复现，不是模型成绩，也不证明其余359个人工病例都可用。

## 为什么选这版图

UTSW论文把Federated Tumor Segmentation（FeTS）配准、去颅骨和初始分割作为主流程，把Advanced Normalization Tools（ANTs）skull-strip列为额外版本。因此在任何模型前固定无`_ants`的`brain_t1ce.nii.gz`，而不是用分割Dice倒选图。

全网格答案固定为`rtumorseg_manual_correction.nii.gz`。UTSW的标签4代表enhancing tumor，和当前post-treatment BraTS中需排除的resection cavity语义不同，所以桥接先把所有positive UTSW标签合为WT，再调用共享no-N4输入链。

## 固定处理和结果

每例执行canonical RAS→finite/非负/黑背景0门→正值脑support bbox→脑内population z-score→图像三线性、标签nearest-exact缩到128³。

- 源图三例均min=0、无负值；positive脑体素1,370,681/1,519,442/1,470,146。
- image-label canonical shape均240×240×155，affine最大绝对差均0。
- 源WT体素150,416/9,487/123,107，所有positive标签均进入WT。
- crop extent分别126×169×152、133×175×137、137×170×148。
- copy A/B输出image exact 3/3、label exact 3/3、geometry exact 3/3；输出finite且WT非空。

## 实现失败边界

原冻结脚本首次执行被共享BraTS专用`case_id`正则拒绝`BT0001`，在128³转换与QC文件前停止。源码确认该函数只验证/返回字符串后，Amendment 01仅允许固定BT0001/2/3通过ID元数据检查，其他外部ID仍拒绝；数值路径未改。首次失败收据和原脚本SHA均保留。

## 直觉、成本和下一步

“按官方主版本而非模型分数选图，并复用no-N4链”的工程直觉在固定3例得到支持。代价为5项合成合同、约7秒CPU、0GPU/0forward，复用已有约56MB双副本。

当前不下载全362、不建cache、不跑模型。若no-N4 Stage A与后续Full972门通过且RHUH优先链完成或有充足窗口，才可另立全362官方主FeTS+`r`标签的下载/cache/外部评测协议；UTSW术前SRI24与训练的治疗后MNI域差必须明确标注，不能与post-treatment排名混排。

## 证据

- [七问/最终填写](IDEA.md)
- [预注册](PREREG.md)
- [实现冻结](IMPLEMENTATION.json)
- [首次实现失败](EXECUTION_01_FAILURE.json)
- [ID修订](AMENDMENT_01.md)
- [修订实现冻结](AMENDMENT_01_IMPLEMENTATION.json)
- [原始输入QC](INPUT_QC.json)，SHA-256 `7573bed6d03f194a429d7eacb4ffb3b711feeda15e91a47539aaf025020deb59`
- [机器结果](RESULT.json)

