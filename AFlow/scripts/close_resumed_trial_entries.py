"""Close failed fixed recipes, without pretending the timed window has ended early."""
from datetime import datetime,timezone
import json
from scripts.structured_background_protocol import ROOT,sha,write_new
from scripts.require_development_mask_verification import require_masks


def main():
    session=ROOT/'archive/research_records/2026-09-10_resumed_eight_hour'
    records={}
    for study in ('small_rotation','image_blur'):
        rec=ROOT/f'archive/research_records/2026-09-10_{study}';require_masks(rec);result=json.loads((rec/'RESULT.json').read_text())
        assert result['selected_arm'] is None and not (rec/'TEST_PREACCESS.json').exists()
        records[study]={'result_sha256':sha(rec/'RESULT.json'),'mask_verification_sha256':sha(rec/'DEV_MASK_VERIFICATION.json'),'test_not_admitted':True}
    rec=ROOT/'archive/research_records/2026-09-10_image_blur'
    write_new(rec/'TEST_NOT_ADMITTED.json',{'status':'not_admitted_development_joint_gates_failed','created_utc':datetime.now(timezone.utc).isoformat(),'result_sha256':sha(rec/'RESULT.json'),'mask_verification_sha256':sha(rec/'DEV_MASK_VERIFICATION.json'),'new_test_forwards':0,'fixed_seed11_test_recipe_not_executed':True,'no_sigma_probability_seed_epoch_threshold_search':True})
    book=session/'EXECUTION_ORDER.md';old=book.read_text()
    with (session/'EXECUTION_ORDER.md.BEFORE_TRIAL_CLOSURE').open('x') as f:f.write(old)
    prefix='''# 当前覆盖说明：本轮模型实验已经结束

旋转四模型与模糊两模型均完成168轮，正式开发及实际mask核验已归档，两种配方均未过联合门。全864旋转和固定seed11模糊测试均未准入，0新test。以下执行顺序保留为过程历史，不能再启动或重复任何本窗口训练/开发/测试入口。当前八小时窗口的剩余时间只用于证据与文档收尾；最终窗口是否结束以RESULT.json及其finished_utc为准，未生成前不能冒称已到八小时。

未来若获新工作窗口，先读当前交接与RESEARCH_SYNTHESIS.md，另注册新的目标编码/匹配对照研究；不要修改这次失败门或复用已经关闭的脚本当新实验。新fit子区标签cache已独立核验，但没有训练区域编码模型。

---

'''
    book.write_text(prefix+old)
    write_new(session/'TRIALS_CLOSED.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'records':records,'all_six_models_completed':True,'new_test_forwards':0,'timed_window_not_marked_complete_by_this_script':True,'execution_order_sha256':sha(book)})
    print('fixed model trials closed; timed window still awaits its declared end',flush=True)


if __name__=='__main__':main()
