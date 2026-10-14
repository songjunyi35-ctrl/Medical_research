# Amendment 003: 让smoke报告顶层自包含采样合同

Status: frozen after V2 GPU smoke computation passed and before formal training or validation/test access.

## Audit finding

Both V2 arms passed all compute gates: four finite steps, positive student gradients, no teacher gradients, exact checkpoint reload, peak allocated`21,788,257,792 B`, and formal extrapolation about1928 seconds. Candidate sampler gates also passed.

A post-smoke read-only audit found that the checkpoint payload contains the patched sampler contract and the top-level `SMOKE.json` contains `sampler_gates`, but the inherited smoke report schema has no top-level `sampling_contract` field for the patcher to replace. The formal report schema already has that field. No validation, outer, confirmation, or test data were accessed.

## Frozen fix

After the inherited run returns and before rewriting the final arm report, explicitly attach:

- `report["sampling_contract"] = sampler_contract()`;
- `report["method_documents"] = method_documents()`.

Then apply the existing recursive patch and write the final report. This changes only report completeness; it does not change RNG, indices, DataLoader, model, optimizer, loss, checkpoint tensors, memory, runtime, selection, or promotion gates.

## Verification

- Add a synthetic assertion that both arm contracts can be attached at the top level.
- Rerun both four-step smokes into new V3 directories because the evidence schema changed.
- Require final top-level strategy, counts, hashes, epoch audits, method document hashes, sampler gates, and checkpoint SHA to be present and consistent before formal.

Access flags at amendment freeze: smoke=true, formal=false, validation=false, outer=false, confirmation=false, test270=false.
