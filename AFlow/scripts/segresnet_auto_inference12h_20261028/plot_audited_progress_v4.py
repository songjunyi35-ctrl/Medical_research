"""Audited accuracy/cost with separate views for low-cost and phase recipes."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts.segresnet_auto_inference12h_20261028.common import *


def main():
    candidates=[]
    for path in sorted((R/'candidates').glob('*/RESULT.json')):
        row=read(path)
        if not row.get('all270_masks_independently_verified'):continue
        m=row['metrics'];candidates.append(dict(candidate=path.parent.name,forwards=row.get('locator_forwards',5),scan=m['scan_dice'],subject=m['subject_dice'],hits=m['point_hits'],result_path=str(path.relative_to(ROOT)),result_sha256=sha(path)))
    references=[]
    for name,label,cost in [('2026-10-28_segresnet_w32_e171_test270','Original',1),('2026-10-28_segresnet_inference_selected_test270','Prior5view',5)]:
        path=ROOT/'archive/research_records'/name/'RESULT.json';m=read(path)
        references.append(dict(candidate=label,forwards=cost,scan=m['scan_dice'],subject=m['subject_dice'],hits=m['point_hits'],result_path=str(path.relative_to(ROOT)),result_sha256=sha(path)))
    points=references+candidates;current=max(candidates,key=lambda p:p['scan']);low=[p for p in points if p['forwards']<=16];high=[p for p in candidates if p['forwards']>32]
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),constrained_layout=True)
    ax=axes[0];ax.scatter([p['forwards'] for p in low],[p['scan'] for p in low],s=30,color='#53768c',alpha=.65)
    costs=sorted({p['forwards'] for p in low});best=[max((p for p in low if p['forwards']<=c),key=lambda p:p['scan']) for c in costs]
    ax.step(costs,[p['scan'] for p in best],where='post',color='#c1683c',linewidth=2)
    for cost in costs:
        p=max((p for p in low if p['forwards']==cost),key=lambda p:p['scan'])
        # Keep the compact1/5/7/8 comparison readable; other points remain visible.
        if cost not in [1,5,7] and p['candidate']!=current['candidate']:continue
        offset=(4,-36) if cost==5 else ((-6,-36) if cost==7 else ((-6,12) if cost>=8 else (4,6)))
        ax.annotate(f"{p['scan']:.6f}\npoint {p['hits']}/270",(cost,p['scan']),xytext=offset,ha='right' if cost>=7 else 'left',textcoords='offset points',fontsize=8)
    ax.set(xlabel='NN forwards per scan',ylabel='Mean scan Dice',title='Audited recipes: up to16 forwards');ax.set_xticks(costs)
    ax.set_ylim(min(p['scan'] for p in low)-.0005,max(p['scan'] for p in low)+.0011)
    ax=axes[1]
    if high:
        for label,key,color in [('Calibrated view groups','raw','#809ca9'),('Groups + components','cc','#416d86'),('Groups + morphology','holes','#678763')]:
            rows=[p for p in high if ('holes' if 'holes' in p['candidate'] else ('cc' if '_cc' in p['candidate'] else 'raw'))==key]
            if rows:ax.scatter([p['forwards'] for p in rows],[p['scan'] for p in rows],s=38,color=color,label=label)
        ax.axhline(current['scan'],color='#c1683c',linestyle='--',linewidth=1.5,label=f"Current best: {current['forwards']} forwards")
        ax.scatter([current['forwards']],[current['scan']],marker='*',s=100,color='#c1683c',zorder=5)
        ax.set_xscale('log');ax.set_xlim(max(1,current['forwards']*.75),max(p['forwards'] for p in high)*1.4)
        ticks=sorted(set([current['forwards']]+[p['forwards'] for p in high]));ax.set_xticks(ticks,[str(int(t)) for t in ticks])
        ax.set_ylim(min(p['scan'] for p in high)-.00015,max(current['scan'],max(p['scan'] for p in high))+.00015)
        ax.legend(loc='lower left',fontsize=7)
    else:ax.text(.5,.5,'View-group masks awaiting audit',ha='center',transform=ax.transAxes)
    ax.set(xlabel='NN forwards per scan (log scale)',ylabel='Mean scan Dice',title='High-cost view-group controls')
    ax=axes[2];rows=references+[current];labels=['Original','Prior5view','Current best'];values=[100*(p['scan']-references[0]['scan']) for p in rows]
    ax.bar(range(3),values,color=['#a5afb5','#6d94a8','#c1683c'],width=.55);ax.set_xticks(range(3),labels)
    for i,p in enumerate(rows):ax.text(i,values[i]+.02,f"{p['scan']:.6f}\npoint {p['hits']}/270",ha='center',fontsize=8)
    ax.set(ylabel='Dice percentage points vs original',title='Same w32 / epoch171 / native270',ylim=(0,max(values)+.18))
    for ax in axes:ax.grid(axis='y',alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Reused270 /119 subjects: test-tuned, post-selection descriptive evidence',fontsize=11)
    fig.supxlabel('Phase forwards use a larger padded canvas; actual input-voxel costs are recorded separately. CPU postprocessing is additional.',fontsize=8)
    out=R/'figures';out.mkdir(exist_ok=True);fig.savefig(out/'audited_accuracy_cost.svg');fig.savefig(out/'audited_accuracy_cost.png',dpi=170);plt.close(fig)
    svg=out/'audited_accuracy_cost.svg';svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    write(out/'PLOT_DATA.json',dict(at=now(),points=points,current_best=current,low_cost_max_forwards=16,phase_panel_min_forwards=32,source_sha256=sha(Path(__file__)),selection_warning='All current-study recipes selected on reused270; no independent confirmation'))
    print(svg)

if __name__=='__main__':main()
