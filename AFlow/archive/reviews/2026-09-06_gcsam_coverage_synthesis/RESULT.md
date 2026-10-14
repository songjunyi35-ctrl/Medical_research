# GCSAM外部coverage倾向：五cohort保存计数综合

状态：post-hoc、各cohort分开；0新forward/cohort重访/选择。

用Full identity与GCSAM identity逐例保存的Dice、predicted/target voxel counts计算precision与recall。UCSD raw没有显式intersection，按Dice×(pred+target)/2反推；其余使用保存整数intersection。每队列单独10,000次配对bootstrap，不pool。

| cohort | 预测体素差（95% CI） | Recall差（95% CI） | Precision差（95% CI） | Point差 |
|---|---:|---:|---:|---:|
| UCSD75 | `+285` `[-504,+1097]` | `+0.0157` `[-0.0009,+0.0359]` | `+0.0054` `[-0.0085,+0.0205]` | 0 |
| Africa95 | `+11,720` `[+8,383,+15,159]` | `+0.0809` `[+0.0547,+0.1071]` | `−0.00380` `[-0.0468,+0.0380]` | −3 |
| CPTAC33 | `+3,845` `[+2,507,+5,292]` | `+0.0223` `[+0.0109,+0.0362]` | `−0.0235` `[-0.0347,−0.0134]` | 0 |
| WAW-pre152 | `+1,299` `[+246,+2,349]` | `+0.00674` `[-0.00838,+0.0210]` | `−0.00190` `[-0.0224,+0.0183]` | −1 |
| WAW-post31 | `+2,386` `[-1,238,+6,281]` | `+0.00770` `[-0.0272,+0.0408]` | `+0.00345` `[-0.0164,+0.0248]` | −1 |

五处预测体积和recall点估计全部为正；precision仅UCSD/WAW-post点估计正且都不确定，Point从未增加。Recall区间只在Africa/CPTAC全正；预测体素区间在Africa/CPTAC/WAW-pre全正。CPTAC明确以precision下降换recall上升，Dice仅小正且CI跨0。

因此可说“原冻结GCSAM checkpoint在外部五队列有更宽coverage/recall的描述性倾向”，不可说optimizer因果、普遍Dice优越或三指标更好。Full/GCSAM训练fit、epoch、seed与选择不同；不同cohort标签/阶段也不同。不得据此做volume router、阈值或checkpoint选择。

Precision空预测约定按既有Africa审计固定为1（GT均非空：empty prediction没有false-positive但recall为0）。V1误用0，使Africa precision差误写`−0.0143`；不影响Dice、recall、预测体积、Point或任何模型。V1原样保留并由V2取代。

证据：V2 [机器结果](RESULT_V2.json) SHA `8598f270157f6991a75baf18f68bf3bbe6d2691a416f75ccd05dc5013c08aaa9`、[图](COMPARISON_V2.png) SHA `947495c7cc581c8046797f9148eb83aa15f5a4d57b9336b3d96bebc968f274f4`；superseded V1 [机器结果](RESULT.json) SHA `570dd74f3866662c5a4d74099127bdc85e3e53a160e8ca8f0f1c3914cb797ad5`。

## V3补记（2026-09-06 Fable session）

加入UCSD-BraTS2024test56后为六外部cohort：GCSAM−Full的recall与预测体积点估计6/6为正、precision仅2/6为正、Point从未增加。新近域队列precision差`−0.0226`事后CI全负、recall跨0、总体配对Dice区间整体为负——在远域大WT（Africa）能兑现的coverage倾向，靠近训练分布时变成纯precision代价。仅描述；不授权optimizer因果、router或threshold。V2保留；见[RESULT_V3.json](RESULT_V3.json)。
