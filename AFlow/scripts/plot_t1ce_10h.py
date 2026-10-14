"""Scientific progress figure from saved reports only; no patient/model access."""
import sys,json,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'archive/research_records/2026-09-21_t1ce_test270_10h'
OUT=ROOT/'workspace/BraTS/segresnet_t1ce_wt/test270_10h_20260921'

def main():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(11,4.2))
 colors={'A':'#176b9b','B':'#ca6521'}
 for c,label in [('A','A: native patches'),('B','B: target weights')]:
  values=[json.loads(p.read_text()) for p in sorted((OUT/c).glob('validation_e*.json'))]
  if values:axes[0].plot([v['epoch'] for v in values],[v['metrics']['scan_dice'] for v in values],'-o',markersize=3,label=label,color=colors[c])
  tests=[json.loads(p.read_text()) for p in (R/'reports').glob(c+'_*.json')];tests.sort(key=lambda d:d['gate_epoch'])
  if tests:axes[1].plot([v['gate_epoch'] for v in tests],[v['metrics']['scan_dice'] for v in tests],'-o',markersize=4,label=label,color=colors[c])
 axes[0].axhline(.751178640,color='#555',ls='--',lw=1,label='Prior full-native combination (historical)')
 if (R/'REFERENCE_PANEL.json').exists():
  baseline=json.loads((R/'REFERENCE_PANEL.json').read_text())['strongest_scan'];axes[1].axhline(baseline,color='#555',ls='--',lw=1,label='Frozen strongest reference')
 axes[1].axhline(.80,color='#428044',ls=':',lw=1,label='Absolute target 0.80')
 for ax,title,n in zip(axes,['Validation: fixed native raster','Target gates: test-tuned'],[108,270]):
  ax.set_title(title);ax.set_xlabel('Additional training epochs' if n==108 else 'Gate epoch (validation-selected checkpoint)');ax.set_ylabel(f'Mean Scan Dice ({n} scans)');ax.grid(alpha=.2);ax.legend(fontsize=7)
 fig.suptitle('T1ce 10-hour campaign | No independent confirmation claim',fontsize=12)
 fig.text(.5,.01,datetime.datetime.now(datetime.timezone.utc).strftime('Updated %Y-%m-%d %H:%M UTC')+' | B gate25 selected initialization (epoch0); A gates selected checkpoints15/40/90.',ha='center',fontsize=8)
 fig.tight_layout(rect=(0,.045,1,.95));fig.savefig(R/'LIVE_CURVES.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
