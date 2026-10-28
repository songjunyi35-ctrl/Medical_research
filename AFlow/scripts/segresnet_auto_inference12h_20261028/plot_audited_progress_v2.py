"""Static scientific plot of audited same-checkpoint results only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.segresnet_auto_inference12h_20261028.common import R,ROOT,read,write,now,sha


def main():
    candidates=[]
    for p in (R/'candidates').glob('*/RESULT.json'):
        row=read(p)
        if not row.get('all270_masks_independently_verified'):continue
        candidates.append(dict(candidate=p.parent.name,forwards=row.get('locator_forwards',5),scan=row['metrics']['scan_dice'],subject=row['metrics']['subject_dice'],hits=row['metrics']['point_hits'],result_path=str(p.relative_to(ROOT)),result_sha256=sha(p)))
    raw=read(ROOT/'archive/research_records/2026-10-28_segresnet_w32_e171_test270/RESULT.json')
    selected=read(ROOT/'archive/research_records/2026-10-28_segresnet_inference_selected_test270/RESULT.json')
    # Historical scalars are immutable reference records, not new executions.
    points=[dict(candidate='Original',forwards=1,scan=.7732280404479416,subject=.7600269199377683,hits=256),dict(candidate='Prior5view',forwards=5,scan=.7809951982288449,subject=.7683062793066104,hits=255)]+candidates
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),constrained_layout=True)
    axes[0].scatter([p['forwards'] for p in points],[p['scan'] for p in points],s=35,color='#53768c',alpha=.7)
    costs=sorted({p['forwards'] for p in points});best=[max([p for p in points if p['forwards']<=c],key=lambda p:p['scan']) for c in costs]
    axes[0].step(costs,[p['scan'] for p in best],where='post',color='#c1683c',linewidth=2)
    for c in costs:
        p=max([p for p in points if p['forwards']==c],key=lambda p:p['scan'])
        axes[0].annotate(f"{p['scan']:.6f}\npoint {p['hits']}/270",(p['forwards'],p['scan']),xytext=((4,-35) if c==5 else ((-6,-38) if c==7 else ((-6,12) if c>=8 else (4,6)))),ha=('right' if c>=7 else 'left'),textcoords='offset points',fontsize=8)
    axes[0].set_ylim(min(p['scan'] for p in points)-.0005,max(p['scan'] for p in points)+.0012)
    axes[0].set(xlabel='Neural forwards per scan (CPU postprocessing extra)',ylabel='Mean scan Dice',title='Audited accuracy / neural cost')
    latest=max(candidates,key=lambda p:p['scan']);labels=['Original','Prior5view','Current best'];rows=[points[0],points[1],latest]
    x=range(3);axes[1].bar(x,[100*(p['scan']-points[0]['scan']) for p in rows],color=['#a5afb5','#6d94a8','#c1683c'],width=.55)
    axes[1].set_xticks(list(x),labels);axes[1].set(ylabel='Dice percentage points vs original',title='Same w32 / epoch171 / native270')
    for i,p in enumerate(rows):axes[1].text(i,100*(p['scan']-points[0]['scan'])+.015,f"{p['scan']:.6f}\npoint {p['hits']}/270",ha='center',fontsize=8)
    axes[1].set_ylim(0,max(100*(p['scan']-points[0]['scan']) for p in rows)+.18)
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Reused270 /119 subjects: test-tuned, post-selection descriptive evidence',fontsize=10)
    out=R/'figures';out.mkdir(exist_ok=True);fig.savefig(out/'audited_accuracy_cost.svg');fig.savefig(out/'audited_accuracy_cost.png',dpi=170);plt.close(fig)
    svg=out/'audited_accuracy_cost.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    write(out/'PLOT_DATA.json',dict(at=now(),points=points,current_best=latest,selection_warning='All current-study configurations selected on reused270; no independent confirmation',source='auditedRESULT.json files'))
    print(out/'audited_accuracy_cost.svg')
if __name__=='__main__':main()
