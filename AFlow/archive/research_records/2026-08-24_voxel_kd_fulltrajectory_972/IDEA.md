# Experiment idea: 972病例从随机初始化参与完整voxel-KD轨迹

Status: frozen after train-only common-inner split generation and before runner implementation, GPU model construction, or inner training.

## 1. 现在的问题是什么？

历史KD200用756例取得fixed validation108 Scan/Subject/Point `0.734384/0.724434/101/108`。972病例late continuation在增加预算后达到`0.734259/0.724956/100/108`，相对same-run 752-draw control提高`+0.000779/+0.000954`，证明数据与预算均有效；但新增216例只在history168之后出现32轮，可能太晚改变特征表示。

## 2. 一句话直觉

新增教材不能只在期末补发；让756/972两组学生都从开学第一天按同样课时学习，才能判断多出的216页是否真正有用。

## 3. 为什么选择这个idea？

- 972数据多样性、减少replacement重复、增加draw预算都相对matched control产生过正Dice信号。
- 当前所有972实验都从756例history168盆地继续，未测试新增病例从随机初始化参与encoder/decoder形成。
- 先前loss、teacher、adapter和late-continuation负结果没有否定full trajectory。
- Fixed validation108已反复访问，不能再用于选择epoch；因此新建只来自train的公共subject-disjoint inner tune。

## 4. 具体怎么做？

阶段A inner selection：从原756公共池按subject抽108 scans / 49 subjects作为两臂共同inner tune。Control fit为剩余648 scans / 297 subjects；candidate fit为972池扣除同一tune后的864 scans / 395 subjects。两臂从相同随机初始化开始，每轮均请求972、消费968、121步，使用相同history/EMA teacher和DiceCE+voxel-KL，跑200轮；每轮只访问common inner tune。只按candidate epoch101–200最大inner Scan选择唯一epoch，Subject exact tie、早epoch再tie；control用同一epoch作比较。

阶段B final：若inner promotion通过，用新固定seed从随机初始化分别在full756/full972上重训到同一个selected epoch；两臂每轮仍968 effective draws/121 steps。完成并冻结checkpoint后才一次性报告fixed validation108。

## 5. 唯一变化与固定项

- 两臂唯一变化：eligible fit/full病例池`648→864`（inner）和`756→972`（final）。
- 固定：common inner tune、每轮968 draws/121 steps、模型4.70M、随机初始state、history/EMA teacher、KD公式、batch8、增强、AdamW、LR schedule、seed、epoch、阈值0、无TTA/SAM/postprocess、1×推理。

## 6. 成功、失败和停止规则

- Inner selection只用common train-inner，candidate在epoch101–200按Scan→Subject→早epoch选择。
- Inner promotion要求candidate严格超过same-epoch control Scan/Subject，Point不低于control超过1/108，Scan/Subject paired 95% CI下界均>=-0.003；失败则不跑final。
- Final validation promotion要求candidate严格超过same-epoch/full-trajectory control和archived KD200 Scan/Subject，Point>=101/108，四CI安全门全过；才允许test270。
- Hash/split/initial-state/step不一致、NaN/Inf、teacher gradient、peak>24GiB、inner单臂>6h、final单臂>6h、reload非exact立即停止。

## 7. 成本与风险

- Inner两卡并行预计4.5–5h；若过门，final再约3.5–4.5h；总墙钟约8–10h。
- 每卡约20.3GiB；inner保存两臂best/checkpoint和history，final两个checkpoint，总新增预计<500MB。
- 推理仍单模型1×。
- 最大风险：从随机初始化做强teacher KD可能早期压制监督学习；或者candidate inner未超过control，从而在阶段A提前停止。

## 8. 最终结果（完成后填写）

- Inner：待运行。
- Final validation：仅inner过门后允许。
- Test：仅final全部门通过后允许。
