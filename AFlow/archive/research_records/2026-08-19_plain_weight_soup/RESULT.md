# Fixed parameter weight soup result

The fixed single-model interpolation `0.75 × historical epoch168 parameters + 0.25 × EMA168 parameters` failed catastrophically on validation108:

| metric | historical | parameter soup | difference |
|---|---:|---:|---:|
| scan Dice | `0.731601` | `0.431343` | `-0.300258` |
| subject Dice | `0.722121` | `0.430487` | `-0.291633` |
| point hit | `0.925926` | `0.796296` | `-0.129630` |

All 108 cases and all 50 subjects worsened. The two independently trained checkpoints occupy different weight-space basins; their successful output-space logit interpolation does not imply that parameter interpolation is valid. Per preregistration, no additional alpha values and no test evaluation were attempted.

- Soup checkpoint SHA-256: `2a5897b68ab631cd9585406d0f781b65e4dd8384bad19139f55aa18e509313af`
- Validation report SHA-256: `d1108004d3da70c3010bad4f93595ac1cab8cfbefba2ac72b8e2dd98ebbde805`
- Test access: false

The next recommended single-model route is fixed-ensemble knowledge distillation rather than additional weight interpolation.
