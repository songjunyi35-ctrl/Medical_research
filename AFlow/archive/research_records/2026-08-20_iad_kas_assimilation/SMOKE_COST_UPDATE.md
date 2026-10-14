# KAS paired A100 smoke cost update

- Physical hardware: two NVIDIA A100-SXM4-40GB, one independent process per GPU.
- Both arms use global batch8 and matched first-batch data/augmentation hashes.
- Control/KAS peak allocated memory: `21,683,973,120 / 24,070,016,000` bytes (`20.19 / 22.42 GiB`), below the frozen32GiB gate.
- Control/KAS median synchronized step: `0.344102 / 0.360362s`.
- Mechanical 32-epoch estimates: `1035.06 / 1083.97s`; parallel wall estimate about18.1min.
- KAS weighted loss initially dominates supervised by about12.6x. This is a mechanism risk, not a technical gate; accepted-paper gamma remains frozen and is not retuned after smoke.
- No validation, outer, confirmation or test access occurred.
