import pathlib,json,hashlib,shutil,datetime
root=pathlib.Path(__file__).resolve().parents[3]
out=pathlib.Path(__file__).resolve().parent
plan=json.loads((out/'CLEANUP_PLAN.json').read_text())
receipt_path=out/'CLEANUP_RECEIPT.json'
assert not receipt_path.exists(), 'Cleanup already executed; do not overwrite receipt'
receipt={'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'deleted':[],'disk_before':shutil.disk_usage(root)._asdict()}
for row in plan['files']:
 p=root/row['path']
 assert p.is_relative_to(root/'workspace/BraTS') and p.is_file() and not p.is_symlink()
 assert p.stat().st_size==row['bytes']
 if row['action']=='keep': continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 assert h.hexdigest()==row['sha256'],str(p)
 p.unlink()
 receipt['deleted'].append(row)
 receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
receipt['disk_after']=shutil.disk_usage(root)._asdict()
receipt['deleted_bytes']=sum(r['bytes'] for r in receipt['deleted'])
receipt['status']='complete'
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({k:v for k,v in receipt.items() if k!='deleted'},indent=2))
