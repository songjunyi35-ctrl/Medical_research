# Full972 width16 vs width24完整轨迹预注册

Status: frozen before implementation and model construction.

## Inner stage

- Arms:`width16_control`,`width24_candidate`。
- Both use candidate-fit864 SHA `6533098a1b0b6e0e7ea37bf166de0b2b0744d5404ec6dac91840e9cfaf8a6f3b` and common-inner108 SHA `87dd75450fb1258b2f5ff2c87ae6635cff3077bb5dfb8109b95cb51681da8bef`。
- Seed20260960；200 epochs；972 requested/968 consumed/121 steps；same teacher/KD/LR/augmentation/AdamW。
- Width16 params4,700,897；width24 params10,574,161。
- Candidate selection on inner epochs101–200 by `(Scan,Subject,-epoch)`；compare control at same epoch。
- Inner gate: candidate Scan/Subject strictly higher；Point within1；two CI lower>=-0.003；engineering gates。

## Final stage

- Only inner pass authorizes final full972 pair。
- Seed20260961；both full972 manifest SHA `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`；same selected epoch and968 draws/121 steps。
- Final width24 must beat matched width16 and archived seed32 validation`0.740012535/0.731266217/103` on Scan/Subject；Point>=101；four CI lower>=-0.003。

## Resources and access

- Peak hard limit32GiB；inner/final each arm elapsed<=8h；disk addition<1GB；no new cache。
- At freeze:new inner=false,new fixed validation=false,new test=false,outer=false,confirmation=false。
- Post-test research label required; no test-based model or hyperparameter selection。
