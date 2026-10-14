# 训练标签引导的每病例最佳老师蒸馏预注册

Status: frozen before implementation, GPU model construction, formal training, or fixed-validation access.

## Objective and arms

- Objective: 测试training-label-guided per-case best-teacher routing能否把Full972三seed互补知识压入1×student，同时避免固定等权平均的负尾。
- `anchor_repeat3_control`: Full972 seed20260932 teacher真实forward三次，target为相同anchor raw logit。
- `gt_best_of3_candidate`: Full972 seed20260932/40/41各forward一次；每病例按GT soft-Dice loss argmin选择一份完整raw logit。
- 唯一arm差异是teacher target routing；teacher forward count固定为3。
- This hypothesis is informed by fresh train-only inner Stage A summary, not by test270 per-case outputs. Fixed validation is reused development evidence; test is not authorized.

## Frozen inputs

- Full972 manifest SHA-256 `6728d8d5e6846f90f6e5d0ad47acc4f2706eed42763ace4e228abcbbd6a4cf65`, 972 scans / 444 subjects.
- Train cache index SHA-256 `7373cb306a01ec1ec311a584a088844439de84ff7289ed819127de6270839cfd`.
- Native WT-volume artifact SHA-256 `32be09856d71e78f512792c8bcf8f5f9ee137e6ba1e4358862ff2a61ba828c31`.
- Full972 teachers:
  - seed20260932 checkpoint SHA-256 `f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b`.
  - seed20260940 checkpoint SHA-256 `48c366ef9f5d3ba5c667482deda6df90fa7cd4bf03c314017e15c13f161dfd38`.
  - seed20260941 checkpoint SHA-256 `00db7860b7951d1f4b0affde6cad558f679421ba93dfe69157c31bd497b19ddb`.
- Fresh consensus Stage A raw inner report SHA-256 `33fe034e829cf7498cd35083c12f3181a47d5db7eabd753f482612edcf15ef5d`; it motivates routing but is not an evaluation cohort for this experiment.
- Archived Full972 seed32 validation report SHA-256 `30cd65c78ad1ed2a2b5748652f48ec05cde3b54468b44e66eba77d062d3ce129`; metrics `0.7400125352044901/0.7312662169195357/103/108`.

## Frozen routing formula

- For each teacher raw logit`z_j`, compute`p_j=sigmoid(z_j)` at temperature1.
- Per-case soft Dice loss over all 128³ voxels:
  `L_j=1-(2*sum(p_j*y)+1e-6)/(sum(p_j)+sum(y)+1e-6)`.
- Candidate index is hard detached`argmin_j L_j`; gather the selected teacher's entire raw-logit volume.
- No per-voxel routing, confidence/entropy term, trainable gate, routing temperature, teacher weight, or tie override. `torch.argmin` first-index behavior resolves exact ties.
- Control runs anchor seed20260932 three separate times in eval/no-grad and uses its raw logit.

## Frozen student protocol

- Student seed`20261040`; exact same initial state in both processes.
- Plain `AutomaticPromptLocator`, init_filters16, dropout0.1, 4,700,897 parameters.
- Full972, native-volume weighted replacement, batch8, 972 requested/968 consumed/121 steps each epoch.
- Fixed stop epoch168 with 200-epoch late-cosine denominator; AdamW LR`2e-4→1e-5`, weight decay`1e-5`, shift8, clip5.
- Loss: sigmoid DiceCE`1:0.1` plus binary voxel KL weight`0.3`, temperature`2`, standard`T²` factor.
- Threshold0; point=raw-logit argmax; no TTA, SAM, connected components, or postprocessing.

## Engineering gates

- Synthetic contracts: per-case soft Dice exact; argmin/gather exact; tie selects first; routing detached; control repeated logits exact; three-forward count exact; nonfinite/shape mismatch fail.
- CUDA smoke: paired initial state, same first augmented batch and student first-pass loss under identical synthetic teacher target; candidate routing finite/nonempty; teacher gradients absent; checkpoint reload/deploy exact.
- Formal pairing: all epoch sampler-index SHA values exact; both arms 168 epochs/121 steps/968 draws; teacher count3; routing counts sum to batch cases; no teacher gradient.
- Peak allocated`<=24GiB`; each arm`<=6h`; no new cache; planned addition`<150MB`.

## Fixed validation gate

- Formal outputs and a one-shot evaluator must be frozen in a pre-access receipt before reading fixed validation108.
- Candidate selected checkpoint is fixed epoch168; no checkpoint/epoch selection.
- Candidate must strictly exceed matched Control and archived Full972 seed32 on Scan and Subject; Point`>=103/108`.
- Candidate-vs-Control and Candidate-vs-archived four paired Scan/Subject 95% CI lower bounds`>=-0.003`, 10,000 draws, seed`20261041`.
- Any failure sets`candidate_authorized_for_test=false` and prohibits routing/teacher/seed/KD/epoch/threshold changes.
- Test270 requires a new receipt and explicit user authorization even if all validation gates pass.

## Sources and adaptation boundary

- https://doi.org/10.1016/j.knosys.2025.113196
- https://repositori.urv.cat/repositori/getDocument/imarina%3A9452339?ds=DocumentPrincipal&mime=application%2Fpdf
- https://www.nature.com/articles/s41598-026-35627-x
- https://github.com/AhmeddEmad7/Brain-Tumor-Segmentation-Advancing-Generalizability

The 2025 paper adaptively weights teachers using GT Dice loss and the 2026 paper routes by known tumor population. This experiment preregisters a project-specific hard best-teacher, output-only, same-architecture/seed-diversity adaptation; it is not a reproduction.

