"""Logged first-batch label byte parity, plus the unchanged-by-code label contract."""
import argparse
from datetime import datetime,timezone
import json
from scripts import image_blur_protocol as p


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--final',action='store_true');args=ap.parse_args()
    p.check_freeze();results={}
    for seed in p.SEEDS:
        path=p.WORK/f'seed{seed}_subject_blur'/('TRAIN_DONE.json' if args.final else 'progress.json')
        r=json.loads(path.read_text());h=r['history'];old=json.loads(p.control_path(seed,'subject_blur').read_text())['history'][:len(h)]
        assert len(h)>0 and (not args.final or len(h)==168)
        assert all(a['first_batch']['labels_sha256']==b['first_batch']['labels_sha256'] for a,b in zip(h,old,strict=True))
        changed=sum(a['first_batch']['image_sha256']!=b['first_batch']['image_sha256'] for a,b in zip(h,old,strict=True))
        assert changed>0
        results[str(seed)]={'epochs':len(h),'recorded_first_batch_labels_all_byte_equal':True,'first_batches_with_changed_images':changed,'report_path':str(path),'report_sha256':p.sha(path)}
    output=p.RECORD/('LABEL_PARITY_FINAL.json' if args.final else 'LABEL_PARITY_'+datetime.now(timezone.utc).strftime('%H%M%S')+'.json')
    p.write_new(output,{'status':'passed_logged_label_parity','created_utc':datetime.now(timezone.utc).isoformat(),'results':results,'scope':'all logged first batches, not a stored byte log of every batch; ImageBlur returns incoming labels object unchanged by construction','script_sha256':p.sha(__file__),'new_forwards':0,'development_access':False,'test_access':False})
    print(json.dumps({'path':str(output),'results':results},indent=2))


if __name__=='__main__':main()
