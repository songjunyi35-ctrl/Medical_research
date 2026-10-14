# Full972多seed ensemble/distillation预注册

Status: frozen before new model construction.

## Frozen seed training

- Arms:`full972_seed20260940`,`full972_seed20260941`。
- Full972 manifest SHA `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`；cache index SHA `7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd`。
- Native artifact/weight SHA `32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31` / `81d3fa6a907f791c4d0b97efe95ece5e057d15b85673463cadbeecc2b3f148ec`。
- Inner-selected epoch receipt SHA `6b88f43291360eedba292165c6a968110a1e618a47a622e2bcda72792db3cca3`；stop epoch168。
- Each epoch972 requested/968 consumed/121 steps；same model/teacher/DiceCE+0.3 voxel-KL T2/LR/augmentation/AdamW。
- Seeds fixed exactly20260940 and20260941；initial states must differ from each other and seed32。

## Frozen ensemble

- Checkpoints: existing seed32 SHA `f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b` plus two future frozen seed40/41 SHAs。
- Formula:`z=(z32+z40+z41)/3`；no weight/grid/seed selection。
- Validation108 only after both checkpoints and evaluator SHA are frozen。
- Promotion: Scan/Subject strictly exceed seed32；Point>=101；candidate-seed32 Scan/Subject CI lower>=-0.003；engineering/access gates。

## Conditional distillation

- Only ensemble promotion authorizes one new random-init Full972 student。
- Teacher is exact three-seed mean logits computed online；student epoch168, same972 data/968 draws/121 steps。
- Student validation must beat seed32 and KD200 Scan/Subject, Point>=101 and four CI gates。

## Evidence scope

Research begins after test270 was revealed. Validation/test evidence must be labeled post-test research. Test cannot select seeds, weights, epoch, loss, threshold or postprocessing. At freeze: new validation=false, new test=false, outer=false, confirmation=false。
