# T1ce/WT intensity-aware uncertainty residual 结果（2026-08-17）

## 结论

本轮没有把全局灰度 hard threshold 直接叠到现有 benchmark 输出上。那样会在已经揭示的
selector/confirmation/test 上产生新的阈值自由度，也不适合 T1ce 中增强、坏死和水肿并存的
异质 WT。实际检验的是更保守的可学习方案：只在 plain locator 不确定的体素上，让一个小型
residual head 使用归一化 T1ce、局部强度对比和梯度幅值修正 logit。

结果为正式负结果。新的 subject-clean inner split 上，intensity residual 相对参数完全匹配的
zero-intensity control：

- scan-macro Dice：`-0.002638`；95% paired bootstrap CI
  `[-0.008138,+0.003299]`；
- subject-balanced Dice：`-0.003331`；95% CI
  `[-0.011072,+0.005886]`；
- automatic point hit：`-0.370370`。

预注册 5 个门槛失败 3 个，`all_inner_gates_passed=false`。因此 outer selector 没有授权、
没有读取；已揭示 confirmation 和 270-case test 也完全没有访问。当前 benchmark checkpoint
仍是 direct plain locator epoch 99，不被本轮替换。

## 证据隔离与预注册

完整协议在 `INTENSITY_RESIDUAL_PREREG_20260817.md`。只使用原 756-case train role 重新做
subject-clean 划分：648 fit cases / 301 subjects，108 inner-tune cases / 45 subjects，case 和
subject overlap 均为 0。fit/inner manifest SHA-256 分别为：

- `947a5364fd2550f5e63af835612a9b97157f7c51eee28c5333de81260dfea19e`；
- `711f441addbc28208ee07e22f9fd14f4345313c123f809b81f1fb02d8826b3f8`。

三臂使用相同 seed `20260821`、batch 8、最多 100 epoch、Dice + 0.1 CE、AdamW
`lr=2e-4`、raw logit threshold `0`。三臂为：

1. `plain`：原 direct locator；
2. `residual_control`：具有与 intensity arm 相同的 3,065 个额外参数，但三个显式强度通道
   恒为 0；
3. `intensity_residual`：显式输入 clipped normalized T1ce、7-voxel local z-contrast 和
   gradient magnitude。

修正形式为 `final_logit = base_logit + 4p(1-p) * residual`，只让 residual 在 locator
不确定区起主要作用；最后一层零初始化。这样 `intensity - control` 才是灰度信息本身的主因果
对比，而不是“多了一个 head/多了参数”的混合对比。

## 严格重载结果

| arm | best epoch | scan Dice | subject Dice | point hit | params |
|---|---:|---:|---:|---:|---:|
| plain | 79 | `0.728322` | `0.713027` | `0.953704` | 4,700,897 |
| residual control | 93 | `0.729545` | `0.717166` | `0.888889` | 4,703,962 |
| intensity residual | 98 | `0.726907` | `0.713835` | `0.518519` | 4,703,962 |

三份最佳 checkpoint 均独立重载全部 108 cases，训练内嵌指标与重载结果的最大绝对差为
`0.0`。checkpoint SHA-256：

- plain：`2685e995ef24cda21248267b0139277acb0875fc20652a5cacb96a40474d004a`；
- control：`97c4a77cf59d7d6f5c92ca5638ead557b018da5422109a8f407a8eb19b9cd80a`；
- intensity：`6194718b2d392f6150c2b3489c1a4389a6eda694698c340064b6115130d05966`。

预注册门槛：

| gate | observed | pass |
|---|---:|---:|
| intensity - control scan >= `+0.003` | `-0.002638` | no |
| intensity - control subject >= `+0.003` | `-0.003331` | no |
| intensity 不比 plain scan 低超过 `0.001` | `-0.001415` | no |
| intensity 不比 plain subject 低超过 `0.001` | `+0.000808` | yes |
| point-hit 相对 control 下降 <= `0.02` | `-0.370370` | no |

## 为什么灰度分支没有提高 benchmark

失败不是因为 residual head 完全没有容量。control head 自身把它的 base output 从
`0.725778`修正到`0.729545`，平均 `+0.003767`；intensity head 只从`0.725107`修正到
`0.726907`，平均 `+0.001800`。intensity 的平均 residual 绝对幅度为 `0.927685`，约为
control 的 `0.322130` 的 2.88 倍，说明显式强度让修正更激进，但没有让修正更正确。

逐病例看，intensity 相对 control 仅 41/108 改善、67/108 变差，中位差
`-0.005318`。两臂的峰值坐标在 108/108 cases 都发生变化；其中 43 例由 control 命中变成
intensity 未命中，只有 3 例反向改善。这与 T1ce 的本质一致：WT 不是一个单调灰度区域，
scanner、subject、治疗后组织和不同肿瘤亚区都会改变绝对与局部强度。强度线索可以帮助局部
边界，却不适合接管全局最高置信位置。

因此，“再加一个全局灰度阈值就能大幅提高很多指标”目前没有证据，且本轮更灵活的学习式
替代已经失败。继续从本 inner split 搜 kernel、强度 cut-off 或 residual scale 会把它变成新的
调参集，不应进行。

## 次要观察与下一步边界

参数匹配 control 相对 plain 在这个 inner split 为 scan `+0.001223`、subject
`+0.004139`；这是主判定之后的 exploratory 对比，case/subject 95% CI 分别为
`[-0.004963,+0.007757]` 和 `[-0.003085,+0.012458]`，均跨 0。它不能晋级 selector，
也不能声称新 record。若以后研究 residual-only regularization，必须另写预注册并使用新的
subject-clean split 或外部数据，不得拿当前 inner/selector/confirmation/test 调参。

更合理的灰度研究方向是先限定到不会改变 locator 全局峰值的区域级边界校准，例如分别建模
enhancing/core/edema 的多亚区概率，再合并成 WT；但这属于新假设，不能作为本轮失败后的
同 split 补救实验。

## 核心收据

- 决策：
  `workspace/BraTS/sammed3d_t1ce_wt/intensity_residual_inner_20260817/seed20260821/INNER_DECISION.json`
  （SHA-256 `0925eb07e42941321fee7faeca19fb54f7499bb67a952006577bcb94dd4cbfef`）；
- plain reload：`INNER_RELOAD_plain.json`
  （`bf7a7da5c8f649da5d7e31baa3809d2c97bcf2520bdbd2c0540c8b2d418a8a1b`）；
- control reload：`INNER_RELOAD_residual_control.json`
  （`b708d32c2e5de19ce95a7943a6f827f294d63080fe12ac709e8491d63c36855d`）；
- intensity reload：`INNER_RELOAD_intensity_residual.json`
  （`3f76e886d623f9d599e8dfe94b0ec0b778f22005514bbf036e2dc000f2cf9694`）。

证据边界：这是新的 subject-clean development inner split 上的单 seed 机制检验，不是外部
验证、confirmation 或 test。它能判定当前 intensity-aware residual 假设不晋级，不能证明
所有可能的 MRI 强度建模都无效。
