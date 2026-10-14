# UTSW异常label的非r原始manual探针

冻结时间：2026-09-06，任何35份非`r`标签下载前。

固定病例：全362诊断中“含label3但不含4”的30例，加“含label5”的5例，共35例；ID SHA=`1d9ef9657137e46beca9879727204a1be814d85578b765a30eca32b71a434bc7`。每例只取`tumorseg_manual_correction.nii.gz`，35路径总556,450 bytes，path/size SHA=`d83a940bc0e8a142be08da56c1f6ea110bf95f311786b4b80584e147340897e3`。

每份用独立公开授权下载到两个新目录，35/35 official size与双SHA exact后才读取数组。只记录unique值、shape、affine与对应`r`值域；不读图像、不重采样、不计算Dice/N4/cache/model。

- 非`r`若全为FeTS标准`{0,1,2,4}`子集而`r`出现3/5，支持“全图r交付引入数值伪值”的来源机制，仍需新协议才能使用positive union。
- 非`r`若也出现3/5或其他值，则数值语义未解决，UTSW模型路线继续关闭。
- 原baseline v1无论如何保持失败，不覆盖。

