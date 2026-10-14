"""Close the conditional full864 branch without training or benchmark access."""
from datetime import datetime,timezone
import json
from scripts import weak_elastic_full864_protocol as p


def main():
    source=p.source.RECORD/'RESULT.json';result=json.loads(source.read_text());verification=json.loads((p.source.RECORD/'VERIFICATION_RECEIPT.json').read_text())
    assert verification['status']=='passed';p.checked(source,verification['result_sha256'])
    assert result['status']=='completed_reused_development_gate_failed' and result['selected_arm'] is None
    assert not (p.RECORD/'TRAIN_PREACCESS.json').exists() and not (p.RECORD/'TEST_PREACCESS.json').exists()
    data={'status':'not_admitted_primary_development_gates_failed','created_utc':datetime.now(timezone.utc).isoformat(),'source_result_sha256':p.sha(source),'source_verification_sha256':p.sha(p.source.RECORD/'VERIFICATION_RECEIPT.json'),'source_selected_arm':None,'full864_models_trained':0,'new_test_forwards':0,'former_development180_consumed_as_training':False,'fit864_manifest_metadata_only':{'path':str(p.FIT),'sha256':p.sha(p.FIT)},'prepared_code_is_not_execution':True,'decision':'close_conditional_full864_elastic_branch_no_failed_gate_test'}
    p.write_new(p.RECORD/'RESULT.json',data)
    text='# 全864弱形变分支：未准入、未执行\n\n主研究四臂完成后两配方均未过开发门，selected_arm=None，因此不启动full864重训，也不访问test270。不是full864模型负结果，因为没有训练该模型。\n\n只提前准备过七问、代码、合成测试和864清单；旧dev180没有被训练消费，仍可按固定684/180角色用于独立备用研究。\n\nfull864新训练0、模型前向0、test0；无新checkpoint。准备代码、清单和负结果边界保留。\n\n[主研究结果](../2026-09-10_weak_elastic/RESULT.md)、[解释](../2026-09-10_weak_elastic/INTERPRETATION.md)、[本分支机器结果](RESULT.json)。\n'
    with (p.RECORD/'RESULT.md').open('x') as f:f.write(text)
    print(json.dumps(data,indent=2))


if __name__=='__main__':main()
