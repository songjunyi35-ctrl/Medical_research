"""Snapshot fit-only learning curves; never access development arrays or metrics."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scripts import structured_background_protocol as p


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,choices=p.SEEDS,required=True)
    ap.add_argument('--prefix',type=Path,required=True);args=ap.parse_args()
    if args.prefix.with_suffix('.json').exists() or args.prefix.with_suffix('.png').exists():
        raise FileExistsError('snapshot already exists')
    snapshot={};source_hashes={}
    for arm in p.ARMS:
        path=p.WORK/f'seed{args.seed}_{arm}'/'progress.json'
        body=path.read_bytes();data=json.loads(body)
        assert data['development_metrics_accessed'] is False
        snapshot[arm]=data['history'];source_hashes[arm]=hashlib.sha256(body).hexdigest()
    fig,axes=plt.subplots(1,3,figsize=(13,3.6),layout='constrained')
    for arm,color in [('control','#4477aa'),('candidate','#ee7733')]:
        rows=snapshot[arm];x=[r['epoch'] for r in rows]
        for ax,key,title in zip(axes,['primary_loss','train_dice','aux_loss'],['Primary WT training loss','Augmented fit Dice','Auxiliary RC training loss']):
            ax.plot(x,[r[key] for r in rows],label=arm,color=color,lw=1.5)
            ax.set(xlabel='Epoch',title=title);ax.grid(alpha=.15)
    axes[0].legend();fig.suptitle(f'Seed {args.seed}: training only; no development evaluation yet')
    args.prefix.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.prefix.with_suffix('.png'),dpi=150);plt.close(fig)
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'seed':args.seed,'scope':'augmented sampled fit data only, not development performance','source_progress_sha256':source_hashes,'epochs':{a:len(v) for a,v in snapshot.items()},'history_snapshot':snapshot,'plot_sha256':p.sha(args.prefix.with_suffix('.png'))}
    args.prefix.with_suffix('.json').write_text(json.dumps(receipt,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'epochs':receipt['epochs'],'plot_sha256':receipt['plot_sha256']}))


if __name__=='__main__':main()
