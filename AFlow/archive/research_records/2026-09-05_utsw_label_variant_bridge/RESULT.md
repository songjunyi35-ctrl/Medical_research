# UTSW人工标签双版本固定3例桥接结果

## 结论

双传输与数组有效性全部通过，但预注册要求“四种配对中恰好一个三例全过”，实际有两个胜者，因此本实现按门失败并停止。不能在结果后把“唯一pair”改成“唯一label”，也不能用模型Dice选择两种T1ce影像。

## 输入与唯一变化

- 固定病例：BT0001、BT0002、BT0003，均在官方TSV中标为人工细化。
- 每例固定两份T1ce、两份人工标签和一份FeTS标签；15文件单份28,880,493 bytes。
- 唯一变化是访问这些固定样本的两份官方传输副本并检查几何/标签；训练、checkpoint、旧cohort、阈值、TTA、SAM和后处理均未访问或改变。

## 传输与有效性

- 两份独立官方传输15/15大小与SHA-256 exact；每份28,880,493 bytes。
- 所有影像finite且非空；所有标签finite、整数exact、含背景0及positive标签。
- 标签值：BT0001/2三份标签均为`{0,1,2,4}`；BT0003两份人工标签为`{0,2}`，FeTS为`{0,1,2}`。这些只描述，不参与配对选择。

## 四种冻结配对

三例全部shape/affine exact的有两个：

1. `brain_t1ce` × `rtumorseg_manual_correction`
2. `brain_t1ce_ants` × `rtumorseg_manual_correction`

两份T1ce均为`240×240×155`并与`r`标签affine最大绝对差0。非`r`标签在BT0001/2只是`74×93×63`、`24×28×36`的裁剪块，affine差97/82；BT0003非`r`标签偶然已为完整`240×240×155`。因此`r`版本是三例一致的全网格人工标签；但两种skull-strip影像不能靠几何消歧。

描述性地，`r`人工标签相对FeTS二值Dice为0.997294、0.996428、0.928655；这不是模型表现，也没有用于选择文件。

## 直觉是否支持

部分支持。双下载和“把标签叠到影像网格”成功识别了全网格`r`标签；但“几何会给出唯一image-label pair”不成立。当前失败只关闭这个唯一pair实现，不否定UTSW362人工来源。

## 成本、访问与下一步

- 墙钟约35秒，0GPU、0模型forward；`/tmp`双副本约56MB。
- 已访问固定3例的影像与标签数组；其余622例、所有旧validation/outer/confirmation/test270/UCSD/RHUH均未访问。
- 不下载全包、不跑模型。若继续，必须另立并在任何模型前冻结“依据官方FeTS/ANTs发布语义与训练预处理谱系选择影像”的输入桥接，不能复用本结果放宽唯一pair门。

## 证据

- IDEA SHA-256：`f3e2d69ed75893f91f8fd86b4d1692b5fe1b5760cd9f2921cc727e674b3edcd5`
- PREREG SHA-256：`2667210f9afaa1d09d874a41a5b6fa9661cb95133f85622237c1114e5f4fb8d7`
- [实现冻结](IMPLEMENTATION.json)
- [预访问收据](PREACCESS.json)
- [双传输收据](TRANSFER_RECEIPT.json)
- [原始几何/标签QC](RAW_QC.json)
- [机器结果](RESULT.json)

