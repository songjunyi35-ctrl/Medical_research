# Historical-best + EMA168 logit ensemble result

## Outcome

The validation-selected fixed logit blend produced a new project-high point estimate on the reused internal test270:

`blended_logit = 0.75 * historical_epoch168_logit + 0.25 * EMA168_logit`

The EMA weight `0.25` was selected from the preregistered candidate set `{0.25, 0.5, 0.75}` on the historical fixed validation108 before test access. Raw blended logit threshold remained zero, and the automatic point was the blended-logit argmax. There was no TTA, postprocessing, SAM, or test-based weight adjustment.

## Validation selection

| EMA weight | scan Dice | subject Dice | point hit |
|---:|---:|---:|---:|
| `0.00` historical | `0.731601219` | `0.722120565` | `0.925926` |
| **`0.25`** | **`0.734171097`** | **`0.724489716`** | `0.944444` |
| `0.50` | `0.732048303` | `0.722623021` | **`0.953704`** |
| `0.75` | `0.723725266` | `0.714813875` | `0.944444` |
| `1.00` EMA168 | `0.718824903` | `0.710217848` | `0.944444` |

Selected `w=0.25` minus historical:

- scan `+0.002569878`, paired 95% CI `[+0.001632,+0.003473]`
- subject `+0.002369150`, paired 95% CI `[+0.001165,+0.003527]`
- point hit `+0.018519`
- 82/108 cases and 39/50 subjects improved

All preregistered gates passed.

## Reused internal test270

| metric | historical true-128³ epoch168 | fixed ensemble | difference | paired 95% CI |
|---|---:|---:|---:|---:|
| scan Dice | `0.741203949` | **`0.745054064`** | **`+0.003850115`** | `[+0.003171,+0.004565]` |
| subject Dice | `0.723179486` | **`0.727344609`** | **`+0.004165123`** | `[+0.003141,+0.005269]` |
| point hit | `0.929630` | **`0.955556`** | **`+0.025926`** | — |

On test, 222/270 cases improved, 46 worsened, and two tied; 100/119 subjects improved and 19 worsened. Median case delta was `+0.003169388`. Point transitions were 249 hit→hit, 2 hit→miss, 9 miss→hit, and 10 miss→miss.

This is the current highest project point estimate on the reused internal test set. It is not a pristine blind test, external confirmation, or public BraTS SOTA claim. The fixed test result will not trigger weight, checkpoint, or threshold tuning.

## Cost and interpretation

The gain requires two 4.70M-parameter locator forward passes and one linear logit blend. Locator inference compute is therefore approximately doubled versus the historical single locator, although there is no additional training, TTA, SAM, or postprocessing. The likely mechanism is error diversification: EMA168 is weaker alone on Dice but has higher point hit and complementary per-case errors; a small `0.25` contribution smooths the historical logits without replacing the stronger model.

## Artifacts

- Historical checkpoint SHA-256: `f7c55b500fbfb1470f120cf682c8fa443995ba6002237d4d8132e3601b8e3be1`
- EMA168 checkpoint SHA-256: `4ea97dacb9a538993e1c16593bc56b03ce6dc3df2f457c1f0426439c6252197e`
- Validation report SHA-256: `66681f5d811f25c6fc67eeafa12b7356b5e0962f98c20be66d5e60b53254460a`
- Test report SHA-256: `b49cf72a7d749afd28bc3009d0f7ad27b63ce5642424cc8eeb183e4eec7dbc7c`
- Test pre-access receipt SHA-256: `eb10a8a01752666f7789621d7ce010772b909f713792273038de6adf218ef8f8`
- Synthetic/temporary-data contract regression: 23 passed

Outer selector and confirmation were not accessed during this ensemble stage; test was accessed only for the validation-selected fixed candidate.
