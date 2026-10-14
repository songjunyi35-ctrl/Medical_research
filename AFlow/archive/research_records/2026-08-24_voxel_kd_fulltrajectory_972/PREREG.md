# 972病例完整voxel-KD轨迹预注册

Status: frozen before implementation, model construction, or inner training.

## Frozen split

- Split manifest SHA `c4fe9066ecb35fd7e86fada4493b59aec03d4b1b21c95f9cbce5c2719532c60b`。
- `control_fit`:648 scans / 297 subjects，SHA `724aaa38fcea4c3f9ad11d7b7828aeb625142911eb4d9c3d64bf71e197f10d61`。
- `candidate_fit`:864 scans / 395 subjects，SHA `6533098a1b0b6e0e7ea37bf166de0b2b0744d5404ec6dac91840e9cfaf8a6f3b`。
- `common_inner_tune`:108 scans / 49 subjects，SHA `87dd75450fb1258b2f5ff2c87ae6635cff3077bb5dfb8109b95cb51681da8bef`。
- Tune来自公共756池；与两fit在case和subject均零重叠。Candidate fit=control fit+216 extra cases。

## Inner training

- Arms:`inner648_control`,`inner864_candidate`。
- Random initialization seed`20260931`; require exact initial state SHA between arms。
- Both use972 requested/968 consumed draws,121 steps/epoch,200 epochs。
- Model AutomaticPromptLocator 4,700,897 params；history/EMA teacher fixed；loss Dice+0.1 BCE+0.3 voxel-KL,T2。
- LR2e-4 through epoch100 then cosine to1e-5 at200；batch8,workers4,shift8,AdamW historical defaults,clip5。
- Validate common inner tune every epoch；save only running best pre100 and post100 per arm plus full metric history。
- Select candidate among101–200 by `(Scan,Subject,-epoch)`；compare control metric at the same epoch。
- Inner gate: candidate Scan/Subject strictly greater；Point >= control_hits-1；paired Scan/Subject CI lower>=-0.003；engineering gates all pass。

## Final retraining

- Only inner gate authorizes final。
- Arms:`full756_control`,`full972_candidate`；seed`20260932`；same selected epoch from candidate inner。
- Both request972/consume968/121 steps each epoch；same model/init/teacher/loss/LR/augmentation/optimizer。
- Full manifests SHA `a520439c5c4b7b6a630a43312c0ce0b2af8d29f671d1e7bea129084bdde74d16` / `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`。
- Freeze two final checkpoints before fixed validation108 access。

## Final promotion

- Candidate must beat full756 control and archived KD200 Scan/Subject；Point>=101/108；four paired bootstrap CI lower bounds>=-0.003。
- Only all gates authorize one reused internal test270；otherwise stop without tuning。

## Costs and access

Inner/final each arm peak<=24GiB and elapsed<=6h。At freeze: common-inner generated from train only；fixed validation108=false，outer=false，confirmation=false，test270=false。
