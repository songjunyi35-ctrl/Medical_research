"""No-patient regression checks for immutable recipe/budget and resume guards."""
import tempfile
from pathlib import Path
from scripts.race_v2_20261029 import common as c
from scripts.race_v2_20261029 import coordinator as pipeline

def main():
    original_record=c.R;original_pipeline_record=pipeline.R;checks=[]
    with tempfile.TemporaryDirectory(prefix='aflow_race_v2_freeze_checks_',dir='/tmp') as folder:
        c.R=Path(folder);pipeline.R=c.R
        recipe=dict(source_epoch=164,source_checkpoint_sha256=c.SOURCE_SHA,effective_batch=8,
                    microbatch_per_gpu=2,world_size=2,updates_per_epoch=122,train_cases_per_epoch=972,learning_rate=2e-5)
        budget=dict(status='passed',new_epochs=80,new_updates=9760,effective_batch=8,
                    microbatch_per_gpu=2,world_size=2,updates_per_epoch=122,examples_per_epoch=972)
        c.write(c.R/'RECIPE.json',recipe);c.write(c.R/'BUDGET_DECISION.json',budget);(c.R/'IDEA.md').write_text('synthetic freeze fixture\n')
        c.write(c.R/'SMOKE_DDP.json',dict(bf16_deployment_exact=True));c.write(c.R/'SMOKE_SOURCE_FREEZE_V3.json',dict(synthetic_fixture=True))
        frozen=dict(source_sha256={'scripts/race_v2_20261029/model.py':c.sha(c.ROOT/'scripts/race_v2_20261029/model.py')},
              source_checkpoint_sha256=c.SOURCE_SHA,numerical_function_ast_sha256=c.numerical_functions(),
              smoke_gate_sha256=c.sha(c.R/'SMOKE_DDP.json'),smoke_source_freeze_sha256=c.sha(c.R/'SMOKE_SOURCE_FREEZE_V3.json'),bf16_deployment_exact=True,
              recipe_sha256=c.sha(c.R/'RECIPE.json'),budget_sha256=c.sha(c.R/'BUDGET_DECISION.json'),idea_sha256=c.sha(c.R/'IDEA.md'))
        c.write(c.R/'EXECUTION_FREEZE.json',frozen);c.verify_sources();checks.append('valid synthetic freeze accepted')
        for file in ('RECIPE.json','BUDGET_DECISION.json','IDEA.md'):
            path=c.R/file;prior=path.read_bytes();path.write_bytes(prior+b' ')
            try:c.verify_sources();raise AssertionError('mutation accepted '+file)
            except RuntimeError:pass
            path.write_bytes(prior);checks.append(file+' mutation rejected')
        prior=c.sha(c.R/'EXECUTION_FREEZE.json');mtime=(c.R/'EXECUTION_FREEZE.json').stat().st_mtime_ns
        pipeline.freeze()
        assert c.sha(c.R/'EXECUTION_FREEZE.json')==prior and (c.R/'EXECUTION_FREEZE.json').stat().st_mtime_ns==mtime
        checks.append('existing execution freeze validated without rewriting timestamps or budget')
        for i in (1,2):c.write(c.R/f'ENGINEERING_RECOVERY_{i}.json',dict(cumulative_engineering_recoveries=i))
        assert pipeline.recovery_count()==2
        try:pipeline.record_recovery('synthetic_third_attempt');raise AssertionError('third recovery accepted')
        except RuntimeError:pass
        assert len(list(c.R.glob('ENGINEERING_RECOVERY_*.json')))==2
        checks.append('persistent recovery count survives attach and rejects third recovery')
        bad=dict(frozen);bad['numerical_function_ast_sha256']={'scripts/race_v2_20261029/runner.py':{'update':'wrong'}}
        c.write(c.R/'EXECUTION_FREEZE.json',bad)
        try:c.verify_sources();raise RuntimeError('bad AST accepted')
        except AssertionError:pass
        checks.append('numerical update AST mutation rejected')
        c.R=original_record;pipeline.R=original_pipeline_record
    c.write(original_record/'FREEZE_CHECKS.json',dict(at=c.now(),status='passed',checks=checks,
            patient_forwards=0,training_updates=0,source_sha256={str(Path(__file__).relative_to(c.ROOT)):c.sha(__file__),
            'scripts/race_v2_20261029/common.py':c.sha(c.ROOT/'scripts/race_v2_20261029/common.py'),
            'scripts/race_v2_20261029/coordinator.py':c.sha(c.ROOT/'scripts/race_v2_20261029/coordinator.py')}))
    print(checks)

if __name__=='__main__':main()
