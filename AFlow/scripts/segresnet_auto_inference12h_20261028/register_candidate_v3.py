"""Register and account for actual padded input voxels in group-view recipes."""
import argparse,re
import numpy as np
from scripts.segresnet_auto_inference12h_20261028.register_candidate_v2 import main as register
from scripts.segresnet_auto_inference12h_20261028.common import *


def main(candidate):
    register(candidate)
    folder=R/'candidates'/candidate;result=read(folder/'RESULT.json');path=folder/'RECIPE.json';seen=set()
    while True:
        assert path not in seen;seen.add(path);recipe=read(path)
        if 'aggregation' in recipe:break
        path=ROOT/recipe['base_recipe_path']
    aggregation=recipe['aggregation'];total=result['locator_forwards']
    if aggregation['kind']=='group_mix':
        groups=aggregation.get('group_sources',[aggregation['group']]);terms=[];group_forwards=0
        if aggregation['alpha']!=0:
            for group in groups:
                descriptor=R/'jobs'/group/'SOURCE.json';source=read(descriptor);spec=source['spec'];count=len(source['views'])
                padded=[int(n)+int(e) for n,e in zip(CANVAS,spec.get('padding_extra',[0,0,0]))];shape=list(spec.get('resize_shape',padded))
                assert len(shape)==3 and not spec.get('roi_margin')
                group_forwards+=count;terms.append(dict(role='transformed_group',group=group,forwards=count,network_canvas=shape,input_voxels=count*int(np.prod(shape)),descriptor_sha256=sha(descriptor)))
        base_forwards=total-group_forwards;assert base_forwards>=0
        if base_forwards:terms.append(dict(role='untransformed_core',forwards=base_forwards,network_canvas=list(CANVAS),input_voxels=base_forwards*int(np.prod(CANVAS))))
        voxels=sum(t['input_voxels'] for t in terms);equivalent=voxels/128**3
        cost=dict(at=now(),neural_case_forwards=total,terms=terms,total_input_voxels_per_scan=voxels,input_voxel_equivalents_per128cubed=equivalent,scope='deployed selected recipe, not search budget',limitations='input voxel accounting only; not architecture FLOPs, GPU latency or CPU postprocessing time',source_sha256=sha(Path(__file__)))
        write(folder/'DEPLOYMENT_COST.json',cost)
        board=read(ROOT/'AFLOW_BRATS_LEADERBOARD.json');row=next(r for r in board['test270_tuned_native_rank_by_scan'] if r['method']==result['method'])
        row.update(forward_equivalents=equivalent,forward_cost_note=cost['limitations']+'; sum actual padded/resized group canvases plus required native base views',deployment_cost_path=str((folder/'DEPLOYMENT_COST.json').relative_to(ROOT)),deployment_cost_sha256=sha(folder/'DEPLOYMENT_COST.json'))
        write(ROOT/'AFLOW_BRATS_LEADERBOARD.json',board)
    else:board=read(ROOT/'AFLOW_BRATS_LEADERBOARD.json')
    path=ROOT/'AFLOW_BRATS_LEADERBOARD.md';text=path.read_text();a=text.index('<!-- TEST270_TUNED_NATIVE -->');b=text.index('<!-- END_TEST270_TUNED_NATIVE -->',a);block=text[a:b];first=next(re.finditer(r'^\| 1 \|',block,re.M)).start()
    table='\n'.join(f"| {v['rank']} | {v['method']} | {v['scan_dice']:.9f} | {v['subject_dice']:.9f} | {v['point_hits']}/270 | {v['forward_equivalents']:.3f}× | [报告]({v['report_path']}) |" for v in board['test270_tuned_native_rank_by_scan'])+'\n'
    prefix=block[:first];marker='AUTO_INFERENCE12H_NATIVE_SCOPE_20261028'
    note=f'<!-- {marker} -->\n本轮auto12h新行由用户授权直接按270反馈选全局规则，属于当前test调参；上方五镜像固定后测试的描述只对应历史方案。相位新行的体素等价量计入实际额外填充画布，CPU后处理另计。\n<!-- END_{marker} -->\n\n'
    if re.search(rf'<!-- {marker} -->.*?<!-- END_{marker} -->',prefix,re.S):prefix=re.sub(rf'<!-- {marker} -->.*?<!-- END_{marker} -->',lambda _:note.strip(),prefix,flags=re.S)
    else:
        at=prefix.index('| 排名');prefix=prefix[:at]+note+prefix[at:]
    path.write_text(text[:a]+prefix+table+'\n'+text[b:])
    print(dict(candidate=candidate,neural_forwards=total,actual_voxel_equivalents=next(r for r in board['test270_tuned_native_rank_by_scan'] if r['method']==result['method'])['forward_equivalents']))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);a=p.parse_args();main(a.candidate)
