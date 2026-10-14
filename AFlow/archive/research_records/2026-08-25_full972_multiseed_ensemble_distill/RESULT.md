# Full972三seed等权logit ensemble结果

## 结论

两个新seed完整训练并正常收敛；固定seed32/40/41各1/3的logit ensemble在fixed validation108达到Scan/Subject/Point=`0.745841/0.735684/104/108`，相对当前seed32提高`+0.005829/+0.004418/+1 point`，88/108 scans、41/50 subjects改善。

但Subject paired 95% CI下界为`-0.005660`，低于预注册`-0.003`门，因此`distillation_authorized=false`。不能看到结果后剔除较弱seed40、改变权重或启动三teacher student蒸馏。

## 唯一变化与固定项

Seed40/41完全复用Full972 epoch168训练协议，只改变随机seed。Ensemble公式在训练前冻结为`(z32+z40+z41)/3`，threshold0、无TTA/SAM/postprocess。没有按validation/test选择seed或权重。

## Validation108 / 50 subjects

| 方法 | Scan | Subject | Point |
|---|---:|---:|---:|
| 三seed等权ensemble | `0.745841` | `0.735684` | 104/108 |
| seed32 | `0.740013` | `0.731266` | 103/108 |
| seed40 | `0.733408` | `0.722941` | 101/108 |
| seed41 | `0.736000` | `0.724980` | 103/108 |

Ensemble-seed32 Scan=`+0.005828931`，95% CI `[-0.000312847,+0.011144778]`；Subject=`+0.004417695`，95% CI `[-0.005659707,+0.011348983]`；Point`+1/108`。

直觉部分支持：三个seed错误高度互补，ensemble点估计明显更高；但subject层不确定性仍宽，严格门不支持继续花费约3.6小时蒸馏。

## 成本与访问

两个新seed两卡并行约3.61h，每卡20.29GiB；workspace约217MB；ensemble推理3×。新validation108已访问；本方法未访问新test、outer或confirmation。研究发生在test270已揭示后，属于post-test research。

Raw report SHA `9eed1ece022a0c0cbd370c41b73f64666c2049f4af6f331a146e6bf8229b15ea`；pre-access SHA `c52b40ba1ab4534e88bf185cb2fefbac3302caf25efac4f330d18e038e73dc36`；seed40/41 checkpoint SHA `48c366ef9f5d3ba5c667482deda6df90fa7cd4bf03c314017e15c13f161dfd38` / `00db7860b7951d1f4b0affde6cad558f679421ba93dfe69157c31bd497b19ddb`。

## 下一决定

停止当前三seed等权ensemble分支，不调权重、不剔除seed、不蒸馏、不访问新test。保留checkpoint和报告作为随机方差与互补证据。
