"""Two-GPU continuation of RACE-Seg; global8, original loss/optimizer/LR."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import fcntl
import sys
import threading
import time
import traceback
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Subset
from monai.losses import DiceCELoss
from scripts.race_seg_20261028.common import *
from scripts.race_seg_20261028 import train as original
from scripts.race_seg_20261028.data import RaceDataset, augment_batch
from scripts import segresnet_pretrained_native_common as native


class DistributedRace(DistributedDataParallel):
    @property
    def enabled(self): return self.module.enabled
    @property
    def prototypes(self): return self.module.prototypes
    @property
    def strength(self): return self.module.strength


def initialize():
    rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(rank)
    dist.init_process_group('nccl', timeout=datetime.timedelta(minutes=5))
    assert dist.get_world_size() == 2
    model = original.setup('race')
    torch.set_num_threads(8)
    return rank, model


def weight_hash(model):
    h = hashlib.sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def check_sync(model):
    hashes = [None, None]
    dist.all_gather_object(hashes, weight_hash(model))
    assert hashes[0] == hashes[1], 'DDP parameters diverged'
    return hashes[0]


def finish_success(rank, mode):
    # All compute/checkpoints are committed before teardown. This environment
    # can stall while Python destroys CUDA/DDP objects; never repeat updates
    # because successful cleanup stalled. Match the historical bounded exit.
    dist.barrier()
    write(R/f'{mode}_EXIT_RANK{rank}.json',dict(at=now(),compute_complete=True,
          bounded_cleanup_seconds=10,checkpoint_work_not_repeated=True))
    def force_exit():
        sys.stdout.flush();sys.stderr.flush();os._exit(0)
    timer=threading.Timer(10,force_exit);timer.daemon=True;timer.start()
    dist.destroy_process_group()
    force_exit()


def local_augmented(batch, epoch, iteration, rank):
    # Reuse the exact full-batch CPU augmentation and random stream, then take
    # disjoint halves. Replicated CPU reads do not duplicate model samples.
    augmented = augment_batch(batch, SEED+epoch*100000+iteration*10)
    n = len(augmented['image'])
    assert n in (4, 8)
    start = rank*(n//2)
    return {key:value[start:start+n//2] for key,value in augmented.items()}


def prepared_batches(loader, epoch, rank):
    """One ordered CPU lookahead; explicit RNG and CPU-only augmentation."""
    with ThreadPoolExecutor(max_workers=1) as pool:
        source=iter(loader)

        def submit(batch, iteration):
            n=len(batch['image']);start=rank*(n//2)
            ids=batch.get('case_id',[])[start:start+n//2]
            return pool.submit(local_augmented,batch,epoch,iteration,rank),n,ids

        first=next(source,None)
        if first is None:return
        pending,n,ids=submit(first,1);iteration=1
        while True:
            local=pending.result()
            following=next(source,None)
            if following is not None:
                future,next_n,next_ids=submit(following,iteration+1)
            yield iteration,n,ids,local
            if following is None:break
            pending,n,ids=future,next_n,next_ids;iteration+=1


def update(model, opt, fn, local, epoch, iteration):
    # Original step averages each rank's two microbatches. DDP averages rank
    # gradients, giving the same global8 sample mean (tail4: two per rank).
    values = original.step(model, opt, fn, local, epoch, iteration, 2, augment=False)
    keys = list(values)
    tensor = torch.tensor([values[k] for k in keys], device='cuda', dtype=torch.float64)
    dist.all_reduce(tensor)
    return dict(zip(keys, (tensor/2).cpu().tolist()))


@torch.inference_mode()
@native.deterministic_validation
def evaluate(model, dataset, loader, epoch, rank):
    model.eval(); cases=[]; gates=[]
    for batch in loader:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            output = model(batch['image'].cuda(), batch['reflection'].cuda(), batch['valid'].cuda())
        cid = batch['case_id'][0]
        item = native.case_metrics(output.segmentation_logits[0].float(), batch['label'][0].cuda(), dataset.geometry[cid])
        item.update(case_id=cid, subject_id=batch['subject_id'][0]); cases.append(item)
        gates.append(float(output.gate.float().mean()))
    gathered = [None, None]
    dist.all_gather_object(gathered, (cases, gates))
    report = None
    if rank == 0:
        cases = sorted([c for rows,_ in gathered for c in rows], key=lambda c:c['case_id'])
        assert len(cases) == len({c['case_id'] for c in cases}) == 108
        metrics = native.summary_ci(cases)
        metrics['point_hits'] = sum(c['automatic_point_hits_gt'] for c in cases)
        report = dict(at=now(), epoch=epoch, metrics=metrics, per_case=cases,
                      gate_mean=sum(sum(g) for _,g in gathered)/108,
                      grid='native1mm crop excludes padding', threshold=.5, test_access=False)
    result = [report]; dist.broadcast_object_list(result, src=0)
    model.train()
    return result[0]


def smoke(rank, bare):
    model = DistributedRace(bare, device_ids=[rank], broadcast_buffers=False)
    opt = original.make_optimizer(bare)
    fn = DiceCELoss(sigmoid=True, lambda_dice=1., lambda_ce=.1)
    generator = torch.Generator().manual_seed(1231)
    x = torch.randn(8,1,*CANVAS, generator=generator)
    y = torch.zeros_like(x); y[:,:,48:80,80:100,64:80] = 1
    classes = y.byte(); classes[:,:,85:100,90:110,65:85] = 2
    matrix = torch.eye(4).repeat(8,1,1); matrix[:,0,0]=-1; matrix[:,0,3]=CANVAS[0]-8
    batch = dict(image=x,label=y,classes=classes,valid=torch.ones_like(x),reflection=matrix)
    times=[]
    synthetic_batches=[batch]*4+[{k:v[:4] for k,v in batch.items()}]
    for iteration,n,ids,local in prepared_batches(synthetic_batches,20,rank):
        native.seed_all(SEED+rank+iteration*10)
        if iteration==1:
            direct=local_augmented(batch,20,1,rank)
            assert all(torch.equal(v,direct[k]) for k,v in local.items()), 'prefetch changed augmented values'
            del direct
        torch.cuda.synchronize(); started=time.monotonic()
        metrics = update(model,opt,fn,local,20,iteration)
        torch.cuda.synchronize(); times.append(time.monotonic()-started)
        assert all(int(s['step'])==iteration for s in opt.state.values())
    digest = check_sync(bare)
    path=TMP/f'ddp200_smoke_rank{rank}.pth'
    original.checkpoint(path,dict(model_state_dict=bare.state_dict(),optimizer_state_dict=opt.state_dict()))
    saved=torch.load(path,map_location='cpu',weights_only=False)
    assert all(torch.equal(v.detach().cpu(),saved['model_state_dict'][k]) and bool(v.isfinite().all()) for k,v in bare.state_dict().items())
    assert all(int(s['step'])==5 for s in saved['optimizer_state_dict']['state'].values())
    peak=torch.cuda.max_memory_reserved()/2**30
    assert peak<=38, 'DDP memory ceiling exceeded'
    rows=[None,None]
    dist.all_gather_object(rows,dict(rank=rank,peak_reserved_gib=peak,seconds_per_update=sum(times[1:4])/3,
                                   updates=5,tail4=True,strict_reload=True,parameter_sha256=digest,metrics=metrics))
    path.unlink()
    if rank==0:
        write(R/'SMOKE_DDP200.json',dict(at=now(),status='passed',patient_forwards=0,
              effective_batch=8,microbatch_per_gpu=2,ranks=rows,base_numerical_functions_unchanged=True,
              prefetched_augmented_values_exact=True))
    finish_success(rank,'SMOKE_DDP200')


def train(rank, bare):
    assert read(R/'SMOKE_DDP200.json')['status']=='passed'
    budget=read(R/'BUDGET_DECISION.json')
    assert budget['epochs']==200 and budget['active_arms']==['race'] and budget['world_size']==2
    record=R/'race'; target=TMP/'runs/race'
    lock=None
    if rank==0:
        lock=(record/'RUN.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    opt=original.make_optimizer(bare)
    saved=torch.load(target/'latest.pth',map_location='cpu',weights_only=False)
    assert saved['arm']=='race' and saved['freeze_sha256']==sha(R/'EXECUTION_FREEZE.json')
    bare.load_state_dict(saved['model_state_dict'],strict=True)
    opt.load_state_dict(saved['optimizer_state_dict'])
    first=saved['epoch']+1; updates=saved['updates']; best=saved['best']; best_epoch=saved['best_epoch']
    elapsed_before=saved['elapsed_seconds']
    assert updates==(first-1)*122 and all(int(s['step'])==updates for s in opt.state.values())
    assert all(torch.equal(v.detach().cpu(),saved['model_state_dict'][k]) for k,v in bare.state_dict().items())
    write(record/f'DDP_RESTORE_RANK{rank}.json',dict(at=now(),start_epoch=first,retained_updates=updates,
          strict_model_reload=True,optimizer_not_reset=True,source_checkpoint_sha256=sha(target/'latest.pth'),
          ddp_freeze_sha256=sha(R/'DDP200_FREEZE.json')))
    del saved
    model=DistributedRace(bare,device_ids=[rank],broadcast_buffers=False)
    check_sync(bare)
    if first>200:
        # Publication-only recovery after the full final optimizer checkpoint.
        if rank==0:
            report=read(record/'validation_e200.json')
            status=read(record/'STATUS.json')
            status.update(at=now(),state='complete',last_complete_epoch=200,target_epochs=200,
                          updates=updates,best_epoch=best_epoch,best_scan=best,
                          validation=report['metrics'],world_size=2,archive_only_recovery=True)
            write(record/'STATUS.json',status)
            original.checkpoint(OUT/'race/latest.pth',dict(model_state_dict=bare.state_dict(),epoch=200,
                  arm='race',freeze_sha256=sha(R/'EXECUTION_FREEZE.json'),ddp_freeze_sha256=sha(R/'DDP200_FREEZE.json')))
            write(record/'DONE.json',status)
        finish_success(rank,'TRAIN_DDP200')
    fn=DiceCELoss(sigmoid=True,lambda_dice=1.,lambda_ce=.1)
    tr=RaceDataset('train'); va=RaceDataset('validation')
    assert len(tr)==972 and len(va)==108
    batches=[]
    # NCCL is not fork safe. Persistent spawn workers avoid both inherited CUDA
    # state and repeated Python/MONAI startup costs at every epoch.
    loader=DataLoader(tr,batch_sampler=batches,num_workers=3,pin_memory=True,prefetch_factor=2,
                      multiprocessing_context='spawn',persistent_workers=True)
    validation_loader=DataLoader(Subset(va,list(range(rank,len(va),2))),batch_size=1,
                      num_workers=2,pin_memory=True,multiprocessing_context='spawn',persistent_workers=True)
    begun=time.monotonic(); model.train()
    for epoch in range(first,201):
        deadline();disk();started=time.monotonic();native.seed_all(SEED+epoch*10+rank)
        lr=native.learning_rate_for_epoch(epoch,base_lr=2e-4,min_lr=1e-5,constant_epochs=100,total_epochs=200)
        for group in opt.param_groups: group['lr']=lr
        order=torch.randperm(972,generator=torch.Generator().manual_seed(SEED+epoch)).tolist()
        batches[:]=[order[i:i+8] for i in range(0,972,8)]
        seen=[]; values=[]
        for iteration,n,case_ids,local in prepared_batches(loader,epoch,rank):
            deadline();seen.extend(case_ids)
            metrics=update(model,opt,fn,local,epoch,iteration);updates+=1;values.append((n,metrics))
            if rank==0 and (iteration==1 or iteration%20==0 or iteration==122):
                write(record/'HEARTBEAT.json',dict(at=now(),phase='training_ddp',epoch=epoch,iteration=iteration,
                      updates=updates,world_size=2,**metrics))
        samples=[None,None];dist.all_gather_object(samples,seen)
        union=[cid for rows in samples for cid in rows]
        assert len(union)==len(set(union))==972 and set(union)==set(tr.case_ids)
        assert updates==epoch*122 and all(int(s['step'])==updates for s in opt.state.values())
        training_seconds=time.monotonic()-started
        if rank==0: write(record/'HEARTBEAT.json',dict(at=now(),phase='validation_ddp',epoch=epoch,updates=updates,world_size=2))
        report=evaluate(bare,va,validation_loader,epoch,rank)
        if report['metrics']['scan_dice']>best:
            best=report['metrics']['scan_dice'];best_epoch=epoch
            if rank==0:
                original.checkpoint(target/'best.pth',dict(model_state_dict=bare.state_dict(),epoch=epoch,arm='race',
                      freeze_sha256=sha(R/'EXECUTION_FREEZE.json'),ddp_freeze_sha256=sha(R/'DDP200_FREEZE.json')))
                tmp=OUT/'race/best.pth.tmp';shutil.copy2(target/'best.pth',tmp);os.replace(tmp,OUT/'race/best.pth')
                write(record/'BEST.json',dict(epoch=epoch,metrics=report['metrics'],checkpoint_sha256=sha(OUT/'race/best.pth')))
        digest=check_sync(bare) if epoch in (first,200) else None
        if rank==0:
            write(record/f'validation_e{epoch:03}.json',report)
            original.checkpoint(target/'latest.pth',dict(model_state_dict=bare.state_dict(),optimizer_state_dict=opt.state_dict(),
                  epoch=epoch,arm='race',updates=updates,best=best,best_epoch=best_epoch,
                  elapsed_seconds=elapsed_before+time.monotonic()-begun,freeze_sha256=sha(R/'EXECUTION_FREEZE.json'),
                  ddp_freeze_sha256=sha(R/'DDP200_FREEZE.json'),world_size=2))
            if epoch in (first,200):
                loaded=torch.load(target/'latest.pth',map_location='cpu',weights_only=False)
                assert all(torch.equal(v.detach().cpu(),loaded['model_state_dict'][k]) and bool(v.isfinite().all()) for k,v in bare.state_dict().items())
                assert all(int(s['step'])==updates for s in loaded['optimizer_state_dict']['state'].values())
                write(record/f'DDP_RELOAD_e{epoch}.json',dict(at=now(),status='passed',updates=updates,
                      parameter_sha256_both_ranks=digest,checkpoint_sha256=sha(target/'latest.pth')));del loaded
            status=dict(at=now(),state='complete' if epoch==200 else 'training',arm='race',last_complete_epoch=epoch,
                  target_epochs=200,updates=updates,best_epoch=best_epoch,best_scan=best,validation=report['metrics'],
                  training_seconds=training_seconds,epoch_seconds=time.monotonic()-started,
                  elapsed_seconds=elapsed_before+time.monotonic()-begun,learning_rate=lr,
                  microbatch=2,effective_batch=8,world_size=2,accumulation_per_rank=2,
                  peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                  train={k:sum(n*v[k] for n,v in values)/972 for k in values[0][1]},test_access=False)
            write(record/'STATUS.json',status)
            with (record/'HISTORY.jsonl').open('a') as f:f.write(json.dumps(status)+'\n')
        dist.barrier()
    if rank==0:
        original.checkpoint(OUT/'race/latest.pth',dict(model_state_dict=bare.state_dict(),epoch=200,arm='race',
              freeze_sha256=sha(R/'EXECUTION_FREEZE.json'),ddp_freeze_sha256=sha(R/'DDP200_FREEZE.json')))
        write(record/'DONE.json',status)
    dist.barrier()
    finish_success(rank,'TRAIN_DDP200')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--smoke',action='store_true');args=parser.parse_args()
    try:
        verify_sources('EXECUTION_FREEZE.json');verify_sources('DDP200_FREEZE.json');deadline();disk()
        rank,model=initialize()
        smoke(rank,model) if args.smoke else train(rank,model)
    except BaseException:
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
