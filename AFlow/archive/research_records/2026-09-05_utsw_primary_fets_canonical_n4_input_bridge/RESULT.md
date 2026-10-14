# UTSW主FeTS图canonical-N4输入桥接结果

## 结论

固定3例canonical-N4 input-only门通过。两个独立Python进程、每进程SimpleITK单线程，对BT0001/2/3生成的128³ image、WT label和geometry全部3/3 exact；相对已归档no-N4输入，label/geometry保持3/3 exact、image按预期3/3不同。

这证明未来可以为same-seed canonical和no-N4模型构建严格配对的UTSW输入cache；没有模型成绩，也不授权全362下载或评测。

## 输入与唯一变化

- 固定主FeTS `brain_t1ce.nii.gz`和全网格`rtumorseg_manual_correction.nii.gz`。
- UTSW所有positive标签先合为WT，避免把术前ET标签4按post-treatment RC语义排除。
- 相对已通过的UTSW no-N4桥接，唯一变化是加入historical canonical正值mask N4；RAS、brain-support bbox、source population z-score、128³插值全部相同。

## 工程结果

- copy A/B child报告SHA：`a1776d09…`/`28ce4887…`。
- 跨进程：image/label/geometry exact=`3/3`、`3/3`、`3/3`。
- 对no-N4：label/geometry exact=`3/3`、`3/3`；image different=`3/3`。
- 三例crop geometry未被N4改变：126×169×152、133×175×137、137×170×148。
- N4输出image SHA分别`5ebb343d…`、`0172a073…`、`7c2714d5…`；label SHA与no-N4相同。

## 直觉、代价和限制

支持“给canonical对照按原配方补光，同时保持标签与几何配对”的工程直觉。两进程总墙钟381.9秒，0GPU、0forward、不新增源文件；3项合成合同通过。

固定3例来自同一术前数据源，只证明实现可重复。UTSW仍是SRI24术前队列，而训练是MNI post-treatment；不能把未来分数与现有test270混排。当前不访问其余359人工病例，不加载checkpoint。

## 下一步

只有在no-N4 Stage A全门通过、条件Full972完成，并先处理已排队的RHUH post-treatment外部链后，才考虑另立全362双下载、canonical/no-N4双cache、same-seed配对外部评测协议。若no-N4未晋级，本桥接仅保留为工程资产。

## 证据

- [七问/最终填写](IDEA.md)
- [预注册](PREREG.md)
- [实现冻结](IMPLEMENTATION.json)
- [预访问收据](PREACCESS.json)
- [copy A](COPY_A_QC.json)
- [copy B](COPY_B_QC.json)
- [INPUT_QC](INPUT_QC.json)，SHA-256 `1cec4ba98fae05e54caca2a1eb0a14681758e79ed6ac35ec47b0a28a58353be0`
- [机器结果](RESULT.json)

