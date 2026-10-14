"""Mechanically derive isolated test tools; executing this does not access test."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'archive/research_records/2026-09-10_resumed_eight_hour'


def write_derived(name,destination,text,receipt):
    src=ROOT/name;dst=ROOT/destination
    if dst.exists():raise FileExistsError(dst)
    with dst.open('x') as f:f.write(text)
    receipt[name]={'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'destination':destination,'destination_sha256':hashlib.sha256(dst.read_bytes()).hexdigest()}


def main():
    receipt={}
    names=['scripts/freeze_weak_elastic_test270.py','scripts/evaluate_weak_elastic_test270.py']
    for name in names:
        text=(ROOT/name).read_text()
        text=text.replace('weak_elastic_test270_helpers','resumed_test270_helpers').replace('weak_elastic','small_rotation').replace('full_elastic','full_rotation')
        text=text.replace('verify_saved_test270_panel','verify_resumed_test270_panel').replace('test_saved_test270_panel_verification','test_resumed_test270_panel_verification')
        text=text.replace('archive_verified_test270_panel','archive_resumed_test270_panel').replace('test_test270_panel_archival_rendering','test_resumed_test270_panel_archival_rendering')
        text=text.replace('2026091031','2026091071').replace('datetime(2026,9,10,11,28,19,','datetime(2026,9,10,21,24,55,')
        text=text.replace("['max_lipschitz_bound']<=.250001","['max_observed_degrees']<=15.000001")
        text=text.replace('gates=h.gates(results,contrasts)',"gates=h.gates(results,contrasts,{'candidate_flip8':'full_rotation_flip8','candidate_identity':'full_rotation_identity','control_flip8':'full_control_flip8'})")
        if name.startswith('scripts/freeze'):
            text=text.replace("    p.check_freeze()", "    assert not (p.ROOT/'archive/research_records/2026-09-10_image_blur/TEST_PREACCESS.json').exists(), 'one new test panel per resumed window'\n    p.check_freeze()",1)
            text=text.replace("    p.write_new(output,", "    files+=['scripts/weak_elastic_test270_helpers.py','scripts/resumed_test270_helpers.py']\n    p.write_new(output,",1)
        write_derived(name,name.replace('weak_elastic','small_rotation'),text,receipt)
    name='scripts/verify_saved_test270_panel.py';text=(ROOT/name).read_text()
    text=text.replace("choices=['weak_elastic_full864','fractional_occupancy']","choices=['small_rotation_full864','image_blur']")
    start=text.index('    gates={');end=text.index('    assert gates==',start)
    text=text[:start]+'''    gates={'candidate_overall_scan_lead':candidate['scan_dice']>constants.OVERALL_SCAN,
           'candidate_identity_1x_scan_lead':raw['results'][role['candidate_identity']]['scan_dice']>constants.SINGLE_SCAN,
           'candidate_flip8_above_matched_control':candidate['scan_dice']>control['scan_dice'],
           'candidate_subject_above_matched_control':candidate['subject_dice']>control['subject_dice'],
           'subject_above_old_overall':candidate['subject_dice']>constants.OVERALL_SUBJECT,
           'point_at_least265':candidate['point_hits']>=265,
           'point_at_least_matched_control':candidate['point_hits']>=control['point_hits'],
           'scan_cluster_ci_positive':contrast['scan_cluster_95ci'][0]>0,
           'subject_ci_positive':contrast['subject_95ci'][0]>0}
'''+text[end:]
    write_derived(name,'scripts/verify_resumed_test270_panel.py',text,receipt)
    name='scripts/archive_verified_test270_panel.py';text=(ROOT/name).read_text()
    text=text.replace('weak_elastic_full864','small_rotation_full864').replace('fractional_occupancy','image_blur').replace('full_elastic','full_rotation').replace('subject_elastic','subject_rotation').replace('2026091031','2026091071')
    text=text.replace('全864弱形变','全864小角度旋转').replace('弱形变','小角度旋转').replace('两臂仅新增形变与否','两臂仅新增旋转与否')
    text=text.replace("'control':'fit684原硬标签对照','candidate':'fit684体积占比标签'","'control':'fit684患者均衡对照','candidate':'fit684图像模糊增强'")
    text=text.replace("        sampling='native_volume'\n        motivation=", "        sampling='subject_equal_within_native'\n        motivation=")
    a=text.index("        motivation='仅把训练WT标签");b=text.index("\n        development=",a)
    text=text[:a]+"        motivation='只对20%训练图像追加sigma0.5—1的轻度模糊，原硬标签、患者均衡抽样、模型、损失、128网格、阈值和部署不变。两seed开发全门后固定seed11测试，不选最好seed；仅fit684/313，无教师。'"+text[b:]
    text=text.replace('weak_elastic_p0.2_max2vox','input_grid_rotation_p0.2_each_axis_15degrees').replace('exact_cell_occupancy_target','image_only_blur_p0.2_sigma0.5_1').replace('original_nearest_exact_hard_target','original_augmentation_subject_sampling')
    write_derived(name,'scripts/archive_resumed_test270_panel.py',text,receipt)
    for name,target,old,new in [('tests/test_saved_test270_panel_verification.py','tests/test_resumed_test270_panel_verification.py','scripts.verify_saved_test270_panel','scripts.verify_resumed_test270_panel'),('tests/test_test270_panel_archival_rendering.py','tests/test_resumed_test270_panel_archival_rendering.py','scripts.archive_verified_test270_panel','scripts.archive_resumed_test270_panel')]:
        write_derived(name,target,(ROOT/name).read_text().replace(old,new),receipt)
    with (REC/'TEST_ENTRY_DERIVATION.json').open('x') as f:json.dump(receipt,f,indent=2)


if __name__=='__main__':main()
