"""Time-stamped, artifact-derived status for the currently authorized window."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import tempfile
from scripts.publish_research_card import publish

ROOT=Path(__file__).resolve().parents[1]
REC=ROOT/'archive/research_records/2026-09-10_resumed_eight_hour'
START=datetime(2026,9,10,13,39,55,tzinfo=timezone.utc)
END=datetime(2026,9,10,21,39,55,tzinfo=timezone.utc)


def atomic(path,body):
    with tempfile.NamedTemporaryFile('w',dir=path.parent,delete=False,encoding='utf-8') as f:f.write(body);temp=Path(f.name)
    temp.replace(path)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--publish',action='store_true');args=ap.parse_args()
    now=datetime.now(timezone.utc);studies={}
    for name in ['small_rotation','image_blur','small_rotation_full864']:
        record=ROOT/f'archive/research_records/2026-09-10_{name}';work=ROOT/f'workspace/BraTS/sammed3d_t1ce_wt/{name}_20260910'
        runs={}
        if work.exists():
            for path in sorted(work.glob('seed*/progress.json')):
                r=json.loads(path.read_text());h=r['history'];done=(path.parent/'TRAIN_DONE.json').exists()
                runs[path.parent.name]={'epochs':len(h),'fixed_epochs':168,'elapsed_seconds':h[-1]['elapsed_seconds'] if h else 0,'complete':done}
        value={'runs':runs,'training_started':(record/'TRAINING_STARTED.json').exists(),'training_done':(record/'TRAINING_DONE.json').exists(),'training_failure':(record/'TRAINING_FAILURE.json').exists(),'result_exists':(record/'RESULT.json').exists(),'test_preaccess_exists':(record/'TEST_PREACCESS.json').exists(),'test_result_exists':(record/'RAW_TEST270.json').exists()}
        if value['result_exists']:
            r=json.loads((record/'RESULT.json').read_text());value['result_status']=r.get('status');value['selected_arm']=r.get('selected_arm');value['metrics']=r.get('metrics',{})
        studies[name]=value
    diagnostics={}
    for name in ['letterbox_fidelity','development_error_anatomy']:
        record=ROOT/f'archive/research_records/2026-09-10_{name}'
        diagnostics[name]={'completed':(record/'RESULT.json').exists(),'failure':(record/'DEVELOPMENT_FAILURE.json').exists() or (record/'FAILURE.json').exists() or (record/'CACHE_FAILURE.json').exists()}
    development_forwards=0;test_forwards=0
    for name in ['small_rotation','image_blur','development_error_anatomy']:
        path=ROOT/f'archive/research_records/2026-09-10_{name}/RAW_DEVELOPMENT.json'
        if path.exists():development_forwards+=json.loads(path.read_text())['new_forwards']
    for name in ['small_rotation_full864','image_blur']:
        path=ROOT/f'archive/research_records/2026-09-10_{name}/RAW_TEST270.json'
        if path.exists():test_forwards+=json.loads(path.read_text())['new_forwards']
    leaderboard=json.loads((ROOT/'AFLOW_BRATS_LEADERBOARD.json').read_text())['test270_direct_locator_rank_by_scan']
    best=leaderboard[0];single=next(r for r in leaderboard if r.get('locator_forwards')==1)
    snapshot={'created_utc':now.isoformat(),'window_start_utc':START.isoformat(),'window_end_utc':END.isoformat(),'elapsed_seconds':(now-START).total_seconds(),'remaining_seconds':max(0,(END-now).total_seconds()),'studies':studies,'diagnostics':diagnostics,'completed_raw_development_forwards':development_forwards,'completed_raw_test_forwards':test_forwards,'partial_or_failed_forwards_not_counted':True,'current_test270_overall':best,'current_test270_one_forward':single}
    atomic(REC/'STATUS.json',json.dumps(snapshot,indent=2,ensure_ascii=False)+'\n')
    lines=[f'## 新八小时实际状态（{now.strftime("%Y-%m-%d %H:%M:%S UTC")}）','',
           '用户新授权窗口13:39:55—21:39:55 UTC；这不是旧03:28—11:28轮次的延长记账。', '',
           '**一句话：** 优先检验训练时小角度旋转；过门才全864同榜比较，否则按预先固定条件检验仅图像模糊。',
           '**出发点：** 旧模型分割覆盖与最高分点稳定性不同步；几种具体负结果不等于上限。',
           '**具体做法：** 原fit684/313与reused dev180/78，原硬评分/阈值/argmax不变，旋转双采样层×两seed固定168轮。成功全864分支会将原dev转为训练，未启动前角色不变。', '',
           '**结果/进度：**']
    for name,info in studies.items():
        status='未正式启动'
        if info['training_started']:status='已启动，以下为保存进度'
        if info['training_done']:status='训练完成'
        if info['training_failure']:status='训练失败，保留现场，不按部分结果选优'
        if info['result_exists']:status=f"结果已归档：{info['result_status']}，入选层={info.get('selected_arm')}"
        runs='；'.join(f"{k}: {v['epochs']}/168" for k,v in info['runs'].items())
        lines.append(f'- {name}：{status}。{runs}')
    lines+=['',f"已完成raw报告的新开发前向{development_forwards}，新test前向{test_forwards}；不包含未完成或失败时的部分计数。",'',
            f"当前同test270/119 overall Scan最高{best['scan_dice']:.9f}（{best['method']}）；单次前向Scan最高{single['scan_dice']:.9f}（{single['method']}）。开发分数不与它们直接混排。",'',
            '**是否支持直觉：** 只有完整开发/独立核验后才判断；不得由训练Dice或工程测试通过推断超榜。保形标签诊断0.965432→0.937514，存在短轴细节代价，未准入模型训练。',
            '**代价/下一步：** 两A100固定训练、推理仍1×；全864或固定测试分支另计1×/8×。正式计算21:24:55前结束，余15分钟归档；未获准分支不启动。', '',
            '[新窗口](archive/research_records/2026-09-10_resumed_eight_hour/SESSION.md)、[机器实时状态](archive/research_records/2026-09-10_resumed_eight_hour/STATUS.json)、[旋转七问](archive/research_records/2026-09-10_small_rotation/IDEA.md)、[模糊备用](archive/research_records/2026-09-10_image_blur/IDEA.md)、[旧点记录更正](archive/research_records/2026-09-10_resumed_eight_hour/POINT_RECORD_CORRECTION.md)。',
            '以下带旧时间戳的“已结束/仍运行”等段落均保留历史，以本窗口及对应机器报告为准。']
    atomic(REC/'CARD.md','\n'.join(lines)+'\n')
    if args.publish:publish(str(REC.relative_to(ROOT)))
    print(json.dumps({k:v for k,v in snapshot.items() if not k.startswith('current_test270')},indent=2))


if __name__=='__main__':main()
