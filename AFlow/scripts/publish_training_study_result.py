"""Publish verified results; never changes scores, gates, or selection."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
from scripts.structured_background_protocol import ROOT,checked,sha,write_new
from scripts.publish_research_card import publish

NAMES={'small_rotation':'小角度输入网格旋转','image_blur':'答案标签不变的轻度图像模糊'}
GATE_NAMES={'subject_gain':'相对native患者Dice至少增加0.003','scan_positive':'相对native扫描Dice增加','point_safety':'相对native点命中减少不超过1例','subject_ci_safety':'患者Dice区间安全','scan_cluster_ci_safety':'扫描Dice患者簇区间安全','own_subject_positive':'胜过自身患者抽样对照的患者Dice','own_scan_positive':'胜过自身患者抽样对照的扫描Dice','own_point_nonnegative':'不降低自身患者抽样对照点命中'}


def interpret(result):
    output={}
    for key,m in result['metrics'].items():
        c=result['comparisons'][key]['versus_own_sampling_control']
        supported=c['scan_cluster_95ci'][0]>0 and c['subject_95ci'][0]>0
        negative=c['scan_cluster_95ci'][1]<0 or c['subject_95ci'][1]<0
        wording='两项自身对照配对区间均为正' if supported else ('至少一项自身对照区间整体为负' if negative else '自身对照的分割变化仍有区间不确定性')
        output[key]={'failed_gates':[GATE_NAMES.get(k,k) for k,v in result['gates'][key].items() if not v],
                     'versus_own_control_scan_delta':c['scan_delta'],'versus_own_control_subject_delta':c['subject_delta'],'versus_own_control_point_delta':c['point_delta'],
                     'own_scan_interval':c['scan_cluster_95ci'],'own_subject_interval':c['subject_95ci'],'dice_interval_interpretation':wording}
    return output


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study',choices=list(NAMES),required=True);args=ap.parse_args()
    rec=ROOT/f'archive/research_records/2026-09-10_{args.study}'
    if (rec/'INTERPRETATION.json').exists():raise FileExistsError('already published')
    r=json.loads((rec/'RESULT.json').read_text());v=json.loads((rec/'VERIFICATION_RECEIPT.json').read_text())
    assert v['status']=='passed';checked(rec/'RESULT.json',v['result_sha256']);checked(rec/'RAW_DEVELOPMENT.json',v['raw_sha256'])
    assert r['status'] in ('completed_reused_development_supported','completed_reused_development_gate_failed')
    from scripts.require_development_mask_verification import require_masks
    require_masks(rec)
    selected=r['selected_arm'];details=interpret(r)
    for c in r['checkpoints'].values():checked(ROOT/c['path'],c['sha256'])
    raw=json.loads((rec/'RAW_DEVELOPMENT.json').read_text());gpu_hours=sum(c['seconds'] for c in r['checkpoints'].values())/3600;peak=max(c['peak_bytes'] for c in r['checkpoints'].values())/1024**3
    common='图像/标签共同追加p0.2、每轴±15°输入网格旋转；原采样层内其他因素固定' if args.study=='small_rotation' else '仅20%训练图像追加sigma0.5—1高斯模糊，原硬标签字节不变，患者均衡抽样固定'
    next_step=('主开发已过门；仍需全864重训工程/预算准入，不能直接用开发分数宣称超榜' if args.study=='small_rotation' else '开发已过门；仍需固定seed11的独立测试预访问准入，不选择较好seed') if selected else '精确配方关闭，不改参数/门或择seed救结果；不准入其测试分支'
    title=NAMES[args.study]
    report={'created_utc':datetime.now(timezone.utc).isoformat(),'study':args.study,'result_sha256':sha(rec/'RESULT.json'),'verification_sha256':sha(rec/'VERIFICATION_RECEIPT.json'),'selected_arm_unchanged':selected,'details':details,'new_forwards':0,'next_decision':next_step}
    write_new(rec/'INTERPRETATION.json',report)
    lines=[f'## {title}：'+('✅ 开发联合门通过，尚非超榜' if selected else '❌ 固定开发联合门未通过'),'',
           '**一句话：** '+('训练时轻微转动整个输入，检验朝向鲁棒性。' if args.study=='small_rotation' else '练习略失焦图像，答案不动，检验是否减少对细碎亮点的依赖。'),
           '**出发点：** native开发Scan0.746762/0.744177；患者均衡0.753475/0.755906，但点命中未稳定改善。具体负结果不等于达到上限。',
           '**具体做法：** '+common+'。fit684/313，reused dev180/78；原4.70M网络、初始化、主损失、168轮、基础增强随机流、阈值0/argmax/1×固定，无教师。','',
           '**结果：**','', '| 候选 | Scan Dice | Subject Dice | Point /180 |','|---|---:|---:|---:|']
    for key,m in r['metrics'].items():lines.append(f"| {key} | {m['scan_dice']:.9f} | {m['subject_dice']:.9f} | {m['point_hits']} |")
    lines+=['','**是否支持直觉：** 分别看同采样对照的面积变化和点命中；下列结论不由native复合收益替代自身对照。']
    for key,d in details.items():
        lines.append(f"\n- {key}：自身对照Scan差{d['versus_own_control_scan_delta']:+.6f}、Subject差{d['versus_own_control_subject_delta']:+.6f}、Point差{d['versus_own_control_point_delta']:+d}；{d['dice_interval_interpretation']}。失败项：{'；'.join(d['failed_gates']) or '无'}。")
    lines+=['', '同一78人反复复用，两个训练seed不是独立队列；10,000次患者簇配对95%区间不作多重选择校正。工程导出/随机流通过不等于泛化成功。完整各对比区间和原门见RESULT.json；不把正点估计都称显著。', '',
            f"**代价/下一步：** 新训练{gpu_hours:.3f} GPU小时（单卡占用墙钟之和），最高allocated显存{peak:.3f}GiB；本次新开发{raw['new_forwards']}前向/{raw['seconds']:.2f}秒，无新影像cache，部署1×。{next_step}。", '',
            '本记录尚无新test/outer/confirmation/fixed validation/external性能。旧同test270/119 overall Scan仍0.757269662（Full972知识蒸馏模型固定八镜像），1次前向Scan仍0.752003176；开发180分数不能与它们直接相减。历史训练源官方字节parity未证明限制保留。', '',
            f'[七问](archive/research_records/2026-09-10_{args.study}/IDEA.md)、[结果](archive/research_records/2026-09-10_{args.study}/RESULT.md)、[机器结果与CI](archive/research_records/2026-09-10_{args.study}/RESULT.json)、[原始开发](archive/research_records/2026-09-10_{args.study}/RAW_DEVELOPMENT.json)、[独立验证](archive/research_records/2026-09-10_{args.study}/VERIFICATION_RECEIPT.json)。']
    for key,c in r['checkpoints'].items():lines.append(f"\n- [{key} checkpoint]({c['path']})：SHA `{c['sha256']}`。")
    text='\n'.join(lines)+'\n'
    with (rec/'INTERPRETATION.md').open('x') as f:f.write(text)
    (rec/'CARD.md').write_text(text);publish(str(rec.relative_to(ROOT)))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
