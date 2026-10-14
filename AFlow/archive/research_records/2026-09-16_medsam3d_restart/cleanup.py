import pathlib, hashlib, json, shutil, datetime
ROOT=pathlib.Path(__file__).resolve().parents[3]
OUT=pathlib.Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()

def strings(x):
 if isinstance(x,dict):
  for v in x.values(): yield from strings(v)
 elif isinstance(x,list):
  for v in x: yield from strings(v)
 elif isinstance(x,str): yield x

leader=json.loads((ROOT/'AFLOW_BRATS_LEADERBOARD.json').read_text())
protected={s for s in strings(leader) if len(s)==64}
for value in list(strings(leader)):
 if value.endswith('.json'):
  report=ROOT/value
  if report.is_file():
   try:
    protected.update(s for s in strings(json.loads(report.read_text())) if len(s)==64)
   except (ValueError, OSError): pass
protected.add('42f96a0a2ade9d98d9db897a11a5a285f7a6351a7693c6884d6ac97a29dfde52')
rows=[]
for p in sorted((ROOT/'workspace/BraTS/sammed3d_t1ce_wt').rglob('*')):
 if not p.is_file() or p.is_symlink() or p.suffix not in ('.pt','.pth','.ckpt'): continue
 rel=str(p.relative_to(ROOT)); digest=sha(p)
 formal_sam=any(part.startswith('sammed3d_') and part!='sammed3d_t1ce_wt' for part in p.parts) and 'smoke' not in rel
 keep=digest in protected or formal_sam
 rows.append(dict(path=rel,bytes=p.stat().st_size,sha256=digest,action='keep' if keep else 'delete',reason='leaderboard hash, rebuild dependency, or formal SAM checkpoint' if keep else 'obsolete non-ranked branch or smoke checkpoint'))
for split in ('train','validation'):
 p=ROOT/f'workspace/BraTS/cache/t1ce_wt_refiner_graymean_20260813/{split}/sam_logits.f16'
 receipt=json.loads((p.parent.parent/'MIGRATION_RECEIPT.json').read_text())
 digest=sha(p)
 assert digest==receipt['splits'][split]['logits_sha256']
 assert (ROOT/f'workspace/BraTS/cache/t1ce_wt_gtcrop_n4_128/{split}/images.f32').is_file()
 index=p.parent/'index.json'
 rows.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=digest,index_sha256=sha(index),action='delete',reason='rebuildable logits; source cache, coarse checkpoint and rebuild script retained'))
assert any(r['sha256']=='42f96a0a2ade9d98d9db897a11a5a285f7a6351a7693c6884d6ac97a29dfde52' and r['action']=='keep' for r in rows)
for digest in ('f7df2a1678564268e899d20411e55c0163df480498273721a7e5b1b71c80d17b','acf2fce112370e3cdb58fe295642680d5db1db3ceb619defdc47824ef4016d0d'):
 assert any(r['sha256']==digest and r['action']=='keep' for r in rows)
plan=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),authorization='2026-09-16 user requested disk cleanup and retaining best experiments',disk_before=shutil.disk_usage(ROOT)._asdict(),files=rows)
(OUT/'CLEANUP_PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
print(json.dumps(dict(files=len(rows),delete_files=sum(r['action']=='delete' for r in rows),delete_gib=sum(r['bytes'] for r in rows if r['action']=='delete')/2**30,keep_files=sum(r['action']=='keep' for r in rows))))
