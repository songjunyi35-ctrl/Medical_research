# DPD-SAM 自动提示、DPG 与 CLS 架构落地审计

Date: 2026-07-22  
Scope: architecture audit only; no GPU execution; no sealed-test read; no shared training-code edit

## 1. Evidence and decision

Primary local source:

- `article/DPD-SAM.pdf`
- SHA-256: `ce49d74a94dcd3529c394a39458ea8583ac8d8226aa0d02de099112171473b4d`

Audited local implementation:

- `sammed3d/dpdsam_binary_model.py`, SHA-256
  `b104031706ee1fe9f6b1031842c95cc8b2b056613171d144d74aba2a5d611cc7`
- `sammed3d/t1ce_binary.py`, SHA-256
  `89759e0d81d91d3f2a588ed1883823bbf48877e2d560defe4b956e239ac436c1`
- `scripts/train_dpdsam_t1ce_b0.py`, SHA-256
  `d07c25793a9288af70449e6bf069d2a1ccdad08ca982bd11bf4a63c5464f62d3`

Exact-title and `DPD-SAM` searches did not locate an author repository. The PDF has no
code link. Consequently, a bit-exact reconstruction is currently impossible; every
unreported choice below is explicitly marked as an inference.

The current static-token B0 is a useful local control but is not the paper's DPD-SAM and
not demonstrably the paper's `SAM-Med3D 74.1` baseline. The closest executable path is:

1. keep B0 unchanged as the case-independent prompt control;
2. add a fit-only frozen global descriptor and offline K-means++ registry;
3. synthesize an image-dependent rank-1 prompt prefix from four domain bases;
4. train the prompt generator, domain adapter, and SAM mask decoder while freezing the
   image encoder;
5. only after the no-diffusion DPG arm works, attach conditional diffusion refinement.

## 2. Paper facts

### 2.1 Task and data

- input is one T1ce volume, `C=1`;
- output is one automatic binary residual-tumor mask;
- all volumes are normalized, bias-field corrected, skull stripped, and resized to
  `128 x 128 x 128`;
- training augmentation uses TorchIO affine (rotation up to 15 degrees and translation
  up to 10%), elastic deformation, and Gaussian noise;
- split is patient-level 8:2 train/test, with about 20% of the training portion reserved
  for validation;
- five runs share the same split and differ by random seed;
- AdamW, LR `4e-4`, weight decay `0.1`, 200 epochs, batch size 2;
- final mask threshold is 0.5 with no post-processing.

The PDF calls the target a single-channel residual-tumor ground truth but does not map it
to BraTS raw labels. Local `{1,2,3}` foreground with label 4 excluded is a reconstruction
assumption, not a recovered paper fact. It must remain visible in every artifact.

### 2.2 Latent Domain Mining (LDM)

- SAM-Med3D's image encoder is frozen;
- an additional learnable `[CLS]` token is said to be prepended to patch embeddings and
  propagated through Transformer blocks;
- its final representation `z_cls` is the global descriptor;
- descriptors from training cases are clustered offline with K-means++;
- `K=4` is the reported optimum;
- centroids and pseudo-domain labels are not updated by gradients during joint training;
- inference computes a descriptor for each case and uses it for domain adaptation.

### 2.3 Dynamic Prompt Generator (DPG)

For latent domain `k`, the paper defines:

`P_k = P_shared * (U_k V_k^T)`

where `*` is elementwise multiplication, `P_shared` is `N x D`, `U_k` is `N x 1`, and
`V_k` is `D x 1`. This is rank 1. A two-layer MLP plus Softmax maps `z_cls` to four
domain weights `w`. The instance prompt is:

`P_weighted = sum_k w_k P_k`.

It is concatenated to sparse prompt embeddings as a continuous visual prefix. No manual
point or box is required. Adapter weights are supervised against offline pseudo labels:

`L_adapter = CE(w, y_pseudo)`.

### 2.4 Trainable and frozen modules

The paper explicitly says:

- frozen: SAM image encoder and offline LDM assignments/centroids;
- jointly optimized: shared prompt base, rank-1 domain factors, Prompt Adapter, SAM mask
  decoder, and later the conditional diffusion network;
- warm-up phase, epochs 1-50: diffusion frozen; optimize segmentation and adapter loss;
- refinement phase, epochs 51-200: activate diffusion and jointly optimize the remaining
  trainable path.

The prose also says LoRA is applied to the SAM mask decoder, but gives no rank, layer
targets, alpha, or whether base decoder weights remain trainable. Algorithm 1 directly
updates the mask decoder `M`. This is unresolved, not evidence for encoder LoRA.

### 2.5 Losses

- `L_seg = lambda_dice * DiceLoss(M_initial,M_gt) + lambda_bce * BCE(M_initial,M_gt)`;
- `L_diff` is noise-prediction MSE;
- `L_adapter` is pseudo-domain cross-entropy;
- `L_total = L_seg + alpha L_diff + beta L_adapter`;
- reported optimum: `alpha=0.1`, `beta=0.1`;
- `lambda_dice` and `lambda_bce` are not disclosed.

### 2.6 Conditional Diffusion Refinement (CDR)

- ground-truth binary mask is treated as a continuous 3D volume;
- DDPM training timesteps `T=1000`; DDIM inference steps `S=10`;
- conditions are T1ce anatomy `X`, SAM initial mask `M_initial`, and global domain feature
  `z_cls`;
- denoiser is a four-level 3D U-Net with channels 64/128/256/512;
- stages use two `3x3x3` convolutions, InstanceNorm, and LeakyReLU with slope 0.01;
- bottleneck uses channel attention;
- the three deepest decoder levels use cross-attention with spatial features as query and
  `z_cls` as key/value;
- skip paths use `1x1x1` convolution and addition.

### 2.7 Paper ablation signal

BraTS2024 DSC/HD95 in Table V:

| Arm | DSC | HD95 |
|---|---:|---:|
| frozen Base | 74.1 | 14.2 |
| Base + CDR | 78.2 | 9.8 |
| Base + Adapter | 78.5 | 12.8 |
| Base + Adapter + LDM | 81.3 | 11.5 |
| Base + Adapter + LDM + DPG | 84.6 | 9.8 |
| full + CDR | 86.8 | 5.0 |

If accepted as reported, this table says dynamic prompt/domain modeling accounts for most
of the DSC gain, while CDR supplies a smaller final DSC gain and a large boundary gain.
Implementation priority should therefore be CLS/LDM/DPG before diffusion.

## 3. Unresolved paper details

These are blockers to an exact claim, not permission to guess silently:

1. **CLS bootstrap contradiction.** The descriptor is called learnable, yet clustering is
   offline before training and the joint-trainable list does not include the CLS module.
   The paper does not explain how a useful CLS token is learned before clustering.
2. **Native encoder incompatibility.** Local SAM-Med3D `ImageEncoderViT3D` processes a
   five-dimensional voxel grid and its attention includes 3D window partitioning and
   relative positions. It has no sequence-level CLS slot. A token cannot simply be
   prepended without redefining every encoder block and CLS positional semantics.
3. **Prompt length `N` is absent.** Prompt embedding width is locally fixed by the mask
   decoder at 384, but the number of prefix tokens is unknown.
4. **Adapter MLP width/activation/dropout and DPG initialization are absent.**
5. **Decoder-LoRA details are absent and conflict with the algorithm's full decoder
   update wording.**
6. **Diffusion input is internally inconsistent.** Equation 8 requires noisy mask `M_t`
   plus conditions `X` and `M_initial`; Table II calls the input only the two-channel
   concatenation `[X,M_initial]`. A denoiser cannot predict the added noise without
   receiving `M_t` through some path.
7. **Segmentation supervision is inconsistent.** Equation 10 defines `L_seg` on
   `M_initial`, while the prose also says the recovered diffusion mask is supervised by
   Dice+BCE. The combination/reduction is not specified.
8. **Residual-tumor raw labels, normalization convention, bias-correction implementation,
   skull-strip implementation, diffusion beta schedule, and metric empty-case policy are
   not disclosed.**
9. **The automatic prompt protocol for the reported SAM-Med3D baseline is not described.**
   Therefore the local B0 cannot be asserted to reproduce `74.1` merely because both are
   automatic and use SAM-Med3D.

## 4. Why the static token can have a low ceiling

The current B0 uses one learned vector of shape `1 x 384`, identical for every case. It
is injected as the sole sparse token and decoded through official mask token 0.

Its likely limitations are structural:

1. **No image-conditioned prompt path.** Every case receives the same prompt. Scanner
   style, contrast, cavity morphology, motion artifacts, and weak-enhancement patterns can
   affect image embeddings, but cannot alter the prompt representation.
2. **It encodes target identity only.** One token can learn “residual tumor,” but cannot
   select among four learned domain bases or interpolate boundary cases.
3. **No pseudo-domain supervision.** There is no auxiliary signal forcing the prompt path
   to separate acquisition variation from the segmentation target.
4. **No prompt prefix capacity.** B0 has one token; DPD-SAM uses an undisclosed `N`-token
   prefix with a shared anatomical template and domain-specific rank-1 modulation.
5. **No refinement model.** Any small missed component, hole, or jagged boundary must be
   solved by the SAM decoder in one pass.
6. **Possible target mismatch.** If the paper's residual mask is not local `{1,2,3}`, a
   lower local result may reflect task definition rather than prompt quality. T1ce alone
   may be especially weak for some voxels in a broad `{1,2,3}` union.
7. **Current training scope differs from the paper.** B0 trains a full mask decoder and
   static token; optional B1 adds encoder Q/V LoRA. The latter violates the paper's frozen
   image-encoder statement and does not substitute for a DPG.

These points explain why B0 is a necessary control but not a sufficient main architecture.
They do not prove an empirical ceiling until paired fit/validation runs exist.

## 5. Closest executable architecture

### 5.1 Keep B0 as the control

Do not mutate `StaticPromptBinarySAM`. Its contract is clean and useful:

- `[B,1,128,128,128]` T1ce input;
- no GT/manual prompt in `forward`;
- frozen image and prompt encoders;
- one static target token;
- official mask token 0;
- train mask decoder and target token.

Disable encoder LoRA for the paper-aligned comparison.

### 5.2 Descriptor and offline LDM

Because true CLS propagation is not compatible with the unmodified encoder and its
pre-clustering training is unspecified, use two explicitly named alternatives:

**L0 deterministic descriptor, recommended first:** capture the final pre-neck frozen
encoder patch grid `[B,8,8,8,768]`, apply LayerNorm and global mean pooling to produce
`z_global [B,768]`. Extract fit cases once in eval/FP32, run K-means++ with `K=4`, and
persist descriptors, centroids, pseudo labels, case IDs, preprocessing/checkpoint hashes,
K-means seed and implementation version. This is a `CLS surrogate`, not paper-exact CLS.

**L1 sidecar learned CLS, experimental:** keep every original patch block unchanged, and
update a sidecar learned query by cross-attention over flattened patch tokens after each
block. It never changes patch tokens, so the source image embeddings remain checkpoint
compatible. This is closer to a learned global token but is still not bidirectional
prepend-through-encoder behavior. It also needs a declared pre-clustering learning rule,
which the paper does not supply. Do not make L1 primary without author code or a frozen
bootstrap protocol.

A true prepended CLS would require reimplementing all twelve 3D blocks, window padding,
relative position logic, and checkpoint loading. It creates more uncertainty than
fidelity and is not the minimum reconstruction.

### 5.3 Rank-one DPG and adapter

Proposed tensor contract:

- `K=4`;
- decoder prompt width `D=384`;
- shared prompt `P_shared [N,384]`;
- `U [4,N,1]`, `V [4,384,1]`;
- domain bases `P [4,N,384]`;
- Adapter: two-layer MLP from descriptor dimension to four logits;
- `w=softmax(logits) [B,4]`;
- weighted prefix `P_weighted [B,N,384]`;
- sparse prompt sequence is exactly `P_weighted`; there are no point/box embeddings;
- dense prompt remains the frozen `no_mask_embed` expansion;
- mask decoder returns token-0 binary logits.

`N` is unknown. Use an ablation that preserves causal interpretation:

- `D1`: `N=1`, dynamic DPG; isolates case-dependent domain interpolation from B0;
- `D2`: one predeclared larger `N` only after D1 passes. `N=4` is a reasonable minimal
  engineering choice but must be labeled an assumption, not a paper value.

The Adapter receives the frozen descriptor, not GT or site metadata. Offline pseudo labels
are used only by `L_adapter` in training. At inference, the MLP Softmax weights synthesize
the prompt automatically.

### 5.4 Decoder adaptation

Primary reconstruction choice: freeze image encoder and prompt encoder; train DPG,
Adapter, mask-decoder base weights except the unused IoU head, matching Algorithm 1's
`Update ... M` wording. Keep source image-encoder hashes unchanged.

Secondary diagnostic only: decoder Q/V LoRA plus output upscaling/hypernetwork heads.
Do not use the existing encoder Q/V LoRA as a proxy. Until the paper's decoder LoRA rank
and target layers are known, neither arm can be called exact.

### 5.5 CDR reconciliation

Implement CDR only after D2 produces a valid automatic initial mask. To reconcile the
paper's equation with its two-channel condition table:

1. feed `[X,M_initial]` through a two-channel condition stem;
2. feed noisy mask `M_t` through a separate one-channel mask stem;
3. add or concatenate the two level-1 features before the 64/128/256/512 U-Net;
4. inject `z_global` through cross-attention at decoder levels 3, 2, and 1;
5. predict one-channel noise;
6. use soft `sigmoid(M_initial)` as the differentiable spatial condition during training;
7. run DDIM for 10 steps and threshold the recovered mask at 0.5.

The separate `M_t` stem is an explicit implementation inference required by the diffusion
equation. A literal two-channel denoiser that never sees `M_t` is invalid.

For phase-2 segmentation supervision, predeclare one reconciliation. The least aggressive
choice is to retain Dice+BCE on `M_initial` as Eq. 10 states and use MSE for diffusion.
An additional Dice+BCE on the recovered mask should be a separate ablation because the
paper does not specify its weight or combination.

## 6. Training and loss schedule

### Phase A: offline fit-only registry

1. Use only training/fit cases; validation and sealed test never participate in clustering.
2. Freeze source encoder and preprocessing.
3. Extract L0 descriptors.
4. Run K-means++ at fixed `K=4` and fixed seed.
5. Persist immutable pseudo labels and centroids; report cluster sizes and distance
   distributions without looking at clinical/site/test labels.

### Phase B: epochs 1-50

Train DPG, Adapter, and mask decoder with:

`L_warm = DiceBCE(M_initial,M_gt) + 0.1 CE(w,y_pseudo)`.

Equal Dice/BCE weights are the current local assumption because the PDF omits lambdas.

### Phase C: epochs 51-200

Activate CDR and train:

`L_total = DiceBCE(M_initial,M_gt) + 0.1 L_diff + 0.1 L_adapter`.

Use AdamW, LR `4e-4`, weight decay `0.1`, batch size 2. The image encoder stays in eval
mode and frozen. Validation selects a checkpoint using predeclared binary DSC; do not tune
the 0.5 threshold or add post-processing.

## 7. Minimal paired matrix

Run in this order after a separate protocol authorizes compute:

| Arm | Prompt | LDM | Decoder | CDR | Purpose |
|---|---|---|---|---|---|
| B0 | static `N=1` | none | train base | no | local static control |
| D0 | descriptor-conditioned generic MLP prompt | none | train base | no | test image-conditioned prompting alone |
| D1 | rank-1 DPG, `K=4,N=1` | L0 | train base | no | isolate dynamic domain interpolation |
| D2 | rank-1 DPG, `K=4,N=4` assumption | L0 | train base | no | prompt-prefix capacity |
| D3 | D2 | L0 | train base | yes | full closest reconstruction |

All arms must share patient split, preprocessing, augmentation draws, training budget,
selection rule, five seeds, and automatic no-GT inference. Report DSC/IoU/HD95 plus
prediction/target empty counts. Do not compare to paper values until the residual label
mapping and metric empty-case policy are resolved.

The paper's ablation suggests D1/D2 should be established before paying the cost and
complexity of D3.

## 8. Future code ownership plan

No shared training code was edited in this audit. A later implementation should add new
modules instead of changing the audited B0 control:

1. `sammed3d/dpdsam_dynamic_prompt.py`
   - frozen descriptor hook/wrapper;
   - immutable `LatentDomainRegistry` buffers;
   - `RankOneDomainPromptGenerator`;
   - `DomainAffinityAdapter`;
   - `DynamicDomainPromptBinarySAM` and metadata/checkpoint validation.
2. `sammed3d/dpdsam_diffusion.py`
   - timestep embedding and beta schedule;
   - separate noisy-mask and two-channel condition stems;
   - four-level 3D U-Net, bottleneck channel attention, domain cross-attention;
   - DDPM loss and deterministic DDIM-10 inference.
3. `scripts/extract_dpdsam_domain_registry.py`
   - fit-only feature extraction and K-means++ artifact generation;
   - strict case/checkpoint/preprocessing/code hashes.
4. `scripts/train_dpdsam_t1ce_dynamic.py`
   - two-phase schedule, resume, five-seed receipts, and no sealed-test capability.
5. `tests/test_dpdsam_dynamic_prompt.py` and `tests/test_dpdsam_diffusion.py`
   - prompt changes when descriptor/domain weights change;
   - rank-1 basis algebra and gradients;
   - adapter CE uses immutable pseudo labels;
   - image encoder state/hash never changes;
   - batch-safe prefix injection and token-0 output;
   - denoiser output depends on `M_t`, `X`, `M_initial`, and `z_global` independently;
   - DDIM reproducibility, checkpoint round trip, and GT firewall.

## 9. Stop gates

- Stop if B0 cannot overfit a tiny train-only subset; DPG will not fix a broken target or
  decoder path.
- Stop if D0/D1 prompts are identical across deliberately different descriptors.
- Stop if D1 domain weights collapse to one cluster or adapter accuracy remains at chance
  while cluster sizes are nondegenerate.
- Stop if any image-encoder parameter or buffer changes.
- Stop if D1 improves only validation but not train-only paired diagnostics under the same
  budget; inspect pseudo-label/descriptor stability before DPG expansion.
- Do not start CDR until D2 produces non-empty, spatially meaningful initial masks.
- Stop CDR if predicted noise is invariant to `M_t`; that would reproduce the paper's
  table ambiguity rather than a valid diffusion model.
- Do not read sealed test, tune threshold, or add post-processing under this architecture
  audit.

## 10. Final judgment

The current static token is too weak to represent DPD-SAM's central mechanism because it
is case-independent and has no domain-discovery supervision. The first meaningful
scientific upgrade is not diffusion; it is a fit-only frozen descriptor plus K=4
rank-one dynamic prompt interpolation. The local 3D encoder cannot support the paper's
claimed prepended CLS without substantial undocumented redesign, so the recommended
first implementation uses a clearly labeled deterministic global-pooling CLS surrogate.
This preserves checkpoint behavior, isolates the DPG hypothesis, and avoids claiming
paper fidelity that the available assets cannot support.
