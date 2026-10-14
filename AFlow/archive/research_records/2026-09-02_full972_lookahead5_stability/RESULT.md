# Full972 Lookahead5稳定优化结果

## 结论

Lookahead-AdamW在完全配对的fresh train-only inner108上低于原AdamW：Scan/Subject/Point=`0.738678/0.716695/104/108`，Control=`0.742575/0.721451/104/108`，差值`-0.003897/-0.004757/0`。六项promotion门只通过Point与工程配对两项，两个点估计门和两个CI安全门失败；`full972_final_authorized=false`。按预注册停止，没有运行Full972 Stage B，也没有访问fixed validation108、outer、confirmation或test270。

通俗地说：每5步把模型拉回慢权重，最后确实让训练集loss更低，但面对没见过的病例反而分得更差。“走得平稳”没有转化成“泛化更正确”。

## 方法与唯一变化

两臂在新subject-clean split的864 fit/392 subjects上使用同一seed20261071、完全相同初始state、每轮sampler序列、首增强batch、4.70M SegResNet、history/EMA teacher、DiceCE+voxel-KL、AdamW内部超参、late-cosine和固定epoch168。

- Control：普通AdamW走20328步，部署最终fast weights。
- Candidate：相同AdamW先走fast step；每`k=5`步执行`slow ← slow + 0.5×(fast−slow)`并把fast同步回slow。共20328 fast steps、4065 slow updates，末尾3个未同步fast steps不部署，checkpoint使用latest slow weights。

唯一变化是AdamW外是否包论文默认`k=5, alpha=0.5`的Lookahead。数据、模型、teacher、loss、batch8、972 requested/968 consumed/121 steps、增强、LR、epoch、阈值0、无TTA/SAM/postprocessing和1×推理固定。

## 工程与配对证据

- Fresh split：864 fit/392 subjects；108 inner/52 subjects；case/subject重叠0。
- V1 smoke训练与配对通过，但checkpoint说明文字错误地写formal同步次数；在formal前只修metadata为动态实际计数并完整重跑V2。
- V2 smoke：初始state、6轮sampler、sampling weights、首增强image/label exact；Candidate 6 fast/1 slow/1 trailing/final-slow合同通过。
- Formal：两臂初始state、sampling weights、全部168轮sampler SHA、首增强image/label exact。
- Candidate准确完成20328 fast steps、4065 slow updates、3 trailing fast steps并最终加载slow；所有reload/deploy/finite/memory/time/access门通过。
- Control/Candidate约`12,206.02s/12,164.24s`，peak`21,796,646,912/21,815,463,936` bytes。

## Fresh train-only inner108 / 52 subjects

| 方法 | Scan Dice | Subject Dice | Point |
|---|---:|---:|---:|
| AdamW Control | **`0.742574776`** | **`0.721451177`** | `104/108` |
| Lookahead5 slow | `0.738677675` | `0.716694527` | `104/108` |
| Candidate − Control | `-0.003897102` | `-0.004756650` | `0` |

- Scan paired 95% CI：`[-0.009837,+0.001641]`；51/108 scans改善、56变差、1持平。
- Subject paired 95% CI：`[-0.011494,+0.001552]`；27/52 subjects改善、24变差、1持平。
- Bootstrap 10,000次，seed20261073。

## Promotion与科学解释

通过：Point不下降；九项工程/配对门全部通过。

失败：Candidate Scan与Subject均未超过Control；Scan/Subject CI下界均低于冻结`-0.003`。

因此不进入Full972 final。最可信解释是：Lookahead默认`k=5, alpha=0.5`从训练早期持续减小有效更新幅度。它最终把fit864 train Dice/loss推到比Control更好的方向，但这个方向对fresh inner产生了约`0.004–0.005`的泛化损失。LAWA6只在末期做小幅平均，三个seed都得到微小正收益；Lookahead把平均反馈贯穿全程，反而过度约束了当前AdamW+late-cosine配方。

这只否定当前Full972-style T1ce/WT、Lookahead默认`k=5,alpha=0.5`、slow-weight evaluation、epoch168实现，不证明所有Lookahead设置不可能。但本inner已揭示，按冻结规则不得搜索`k/alpha`、改评测fast weights、延长epoch或调LR/阈值。

## 成本、访问与证据

- 双卡formal墙钟约3.39h，总约6.77 GPU-hours。
- Candidate额外slow参数约18MB；peak只比Control多约18.8MB。
- Workspace约217M，不建新cache，不删除材料。
- 推理仍4.70M参数、1×forward。
- Fresh train-only inner已访问；fixed validation、outer、confirmation、test270均未访问。
- IDEA SHA`b71bd5c2a018ee638d3517fa179f42fbe21fa2d197336f7a451888bbec814f36`；PREREG SHA`cbbdcfdb8a3424e573dbec531edad84b8344abc8827113128eaa4d34720807d9`。
- Raw inner SHA`6d30ec02e87f68623cbb3eec45eb09be1e81ad3784fb04a4860a48f194f0bcc3`；preaccess SHA`44b2e0005e6d32655733409d22c730762040a029154487e58c3f7d22987ea2cd`。
- Candidate checkpoint SHA`179408fd2b6e3a0105ed259c780e76c9b7081cac71fbb917dcc5c51a91e3e2af`。

## 下一决定

关闭Lookahead5分支，不运行Stage B、不访问test、不调`k/alpha/slow-fast选择/epoch/LR/seed/threshold`。结合GCSAM与LAWA6，当前结论是：末期同盆地平均有可复现但很小的正收益；把slow平均反馈贯穿全训练并不能解决seed稳定性。后续若继续内部development，必须离开优化器/权重平均轴并建立新fresh证据。

Evidence links: [七问](IDEA.md)、[预注册](PREREG.md)、[raw inner](../../../workspace/BraTS/sammed3d_t1ce_wt/full972_lookahead5_stability_20260902/STAGE_A_INNER.json)、[Candidate checkpoint](../../../workspace/BraTS/sammed3d_t1ce_wt/full972_lookahead5_stability_20260902/formal_lookahead5_candidate/raw_lookahead5_candidate_epoch168.pth)。
