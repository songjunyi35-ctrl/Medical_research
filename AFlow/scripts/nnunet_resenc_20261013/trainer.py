"""Official training recipe with explicit split, native full-volume selection, and durable state."""
import copy
import random
import time
import numpy as np
import torch
import torch.distributed as dist
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from scripts.nnunet_resenc_20261013.common import *

def restore_margin(logits,plans_manager,configuration_manager,props):
    spacing=[props['spacing'][i] for i in plans_manager.transpose_forward]
    restored=configuration_manager.resampling_fn_probabilities(logits,props['shape_after_cropping_and_before_resampling'],
        configuration_manager.spacing,spacing)
    if isinstance(restored,np.ndarray):restored=torch.from_numpy(restored)
    margin=(restored[1].float()-restored[0].float()).cpu().numpy()
    canvas=np.full(props['shape_before_cropping'],-1e4,dtype=np.float32)
    canvas[tuple(slice(a,b) for a,b in props['bbox_used_for_cropping'])]=margin
    return canvas.transpose(plans_manager.transpose_backward).transpose(2,1,0)

def storage_guard(additional_bytes=0):
    used=sum(p.stat().st_size for directory in [R,OUT] if directory.exists() for p in directory.rglob('*') if p.is_file())
    if used+additional_bytes>8*2**30:raise RuntimeError('new persistent storage exceeds8GiB')
    if shutil.disk_usage(ROOT).free-additional_bytes<20*2**30:raise RuntimeError('write would breach20GiB reserve')

class AFlowTrainer(nnUNetTrainer):
    def __init__(self,plans,configuration,fold,dataset_json,device=torch.device('cuda')):
        super().__init__(plans,configuration,fold,dataset_json,device)
        self.smoke=os.environ.get('AFLOW_RESENC_SMOKE')=='1'
        if self.smoke:
            self.preprocessed_dataset_folder=str(TMP/'synthetic/data')
            self.preprocessed_dataset_folder_base=str(TMP/'synthetic')
            self.output_folder_base=str(TMP/'synthetic/results')
        else:
            self.output_folder_base=str(OUT/'model')
        self.output_folder=str(Path(self.output_folder_base)/'fold_0')
        Path(self.output_folder).mkdir(parents=True,exist_ok=True)
        self.log_file=str(Path(self.output_folder)/'training.log')
        self.num_epochs=1000 if self.smoke else int(read(R/'EXECUTION_FREEZE.json')['epochs'])
        self.actual_optimizer_steps=0
        self.attempted_updates=0
        self.best_native_scan=-1.
        self.best_native_epoch=None
        self.run_started=time.monotonic()

    def do_split(self):
        if self.smoke:return [f'synthetic{i}' for i in range(6)],[f'synthetic{i}' for i in range(6,8)]
        verify_manifests()
        split=read(PRE/'splits_final.json')
        expected={'train':[x['case_id'] for x in rows('train')],'val':[x['case_id'] for x in rows('validation')]}
        if split != [expected]:raise RuntimeError('fixed split changed')
        return expected['train'],expected['val']

    def initialize(self):
        super().initialize()
        self.optimizer.register_step_post_hook(self._step_committed)

    def _step_committed(self,optimizer,args,kwargs):self.actual_optimizer_steps+=1
    def plot_network_architecture(self):pass

    def train_step(self,batch):
        self.attempted_updates+=1
        result=super().train_step(batch)
        if not np.isfinite(result['loss']).all():raise RuntimeError('nonfinite training loss')
        return result

    def save_checkpoint(self,filename):
        if self.local_rank!=0:return
        disk()
        estimated=sum(p.numel()*p.element_size() for p in self.network.parameters())*2+64*2**20
        storage_guard(estimated)
        p=Path(filename);temp=p.with_name(p.name+'.tmp')
        super().save_checkpoint(str(temp))
        checkpoint=torch.load(temp,map_location='cpu',weights_only=False)
        checkpoint['aflow']={'actual_optimizer_steps':self.actual_optimizer_steps,'attempted_updates':self.attempted_updates,
           'best_native_scan':self.best_native_scan,'best_native_epoch':self.best_native_epoch,
           'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),
           'augmentation_worker_rng_not_serialized':True,'engineering_resume_not_bitwise':True}
        torch.save(checkpoint,temp);os.replace(temp,p)

    def load_checkpoint(self,filename_or_checkpoint):
        c=torch.load(filename_or_checkpoint,map_location='cpu',weights_only=False) if isinstance(filename_or_checkpoint,str) else filename_or_checkpoint
        super().load_checkpoint(c)
        a=c['aflow'];self.actual_optimizer_steps=a['actual_optimizer_steps'];self.attempted_updates=a['attempted_updates']
        self.best_native_scan=a['best_native_scan'];self.best_native_epoch=a['best_native_epoch']

    @torch.inference_mode()
    def native_validation(self,epoch):
        import nibabel as nib
        from scripts.segresnet_plain972_common import summary_ci
        self.network.eval();self.set_deep_supervision_enabled(False)
        model=self.network.module if self.is_ddp else self.network
        predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=False,
           perform_everything_on_device=True,device=self.device,verbose=False,verbose_preprocessing=False,allow_tqdm=False)
        predictor.manual_initialization(model,self.plans_manager,self.configuration_manager,None,self.dataset_json,
            self.__class__.__name__,None)
        ids=[r['case_id'] for r in rows('validation')]
        dataset=self.dataset_class(self.preprocessed_dataset_folder,ids)
        world=dist.get_world_size() if self.is_ddp else 1
        local=[];started=time.monotonic()
        for i in range(self.local_rank,len(ids),world):
            cid=ids[i];data,seg,_,props=dataset.load_case(cid)
            tensor=torch.from_numpy(np.asarray(data[:]).copy())
            logits=predictor.predict_sliding_window_return_logits(tensor).cpu()
            margin=restore_margin(logits,self.plans_manager,self.configuration_manager,props)
            target=np.asarray(nib.load(RAW/'labelsTr'/f'{cid}.nii.gz').dataobj)>0
            if margin.shape!=target.shape or not np.isfinite(margin).all():raise RuntimeError('invalid restored output')
            pred=margin>0;tp=int((pred&target).sum());fp=int((pred&~target).sum());fn=int((~pred&target).sum())
            dice=float(np.float32(2*tp)/np.float32(2*tp+fp+fn)) if 2*tp+fp+fn else 1.
            point=np.unravel_index(int(margin.argmax()),margin.shape)
            local.append({'case_id':cid,'subject_id':cid.rsplit('-',1)[0],'locator_dsc':dice,
               'automatic_point':[int(v) for v in point],'automatic_point_hits_gt':bool(target[point]),
               'tp':tp,'fp':fp,'fn':fn})
        gathered=[None]*world
        if self.is_ddp:dist.all_gather_object(gathered,local)
        else:gathered=[local]
        cases=sorted([r for part in gathered for r in part],key=lambda x:x['case_id'])
        assert len(cases)==len({r['case_id'] for r in cases})==108
        metrics=summary_ci(cases);metrics['point_hits']=sum(r['automatic_point_hits_gt'] for r in cases)
        if self.local_rank==0:
            write(OUT/f'validation_e{epoch:04d}.json',{'at':now(),'epoch':epoch,'grid':'native1mm crop, padding excluded',
              'metrics':metrics,'per_case':cases,'seconds':time.monotonic()-started,'test_access':False,
              'selection':'full native Scan; earliest tie','mirroring':False,'threshold':.5})
        self.set_deep_supervision_enabled(True);self.network.train()
        return metrics

    def finish_epoch(self,force_validation=False):
        epoch=self.current_epoch+1
        self.logger.log('epoch_end_timestamps',time.time(),self.current_epoch)
        metrics=None
        if epoch==1 or epoch%25==0 or epoch==self.num_epochs or force_validation:
            metrics=self.native_validation(epoch)
            if metrics['scan_dice']>self.best_native_scan:
                self.best_native_scan=metrics['scan_dice'];self.best_native_epoch=epoch
                self.save_checkpoint(str(Path(self.output_folder)/'checkpoint_best.pth'))
        if epoch==1 or epoch%5==0 or epoch==self.num_epochs or force_validation:
            self.save_checkpoint(str(Path(self.output_folder)/'checkpoint_latest.pth'))
        if self.local_rank==0:
            status={'at':now(),'state':'training','last_complete_epoch':epoch,'target_epochs':self.num_epochs,
              'attempted_updates':self.attempted_updates,'actual_optimizer_steps':self.actual_optimizer_steps,
              'best_epoch':self.best_native_epoch,'best_scan':self.best_native_scan,'validation':metrics,
              'peak_allocated_gib':torch.cuda.max_memory_allocated(self.device)/2**30,
              'peak_reserved_gib':torch.cuda.max_memory_reserved(self.device)/2**30,'new_test_access':False}
            write(OUT/'STATUS.json',status)
            write(R/'PIPELINE_STATUS.json',{**status,'running':True,'training_started':True})
            history={**status,'attempt_pid':os.getpid(),'train_loss':float(self.logger.my_fantastic_logging['train_losses'][-1]),
                'patch_val_loss':float(self.logger.my_fantastic_logging['val_losses'][-1]),
                'patch_pseudo_dice':float(self.logger.my_fantastic_logging['mean_fg_dice'][-1]),
                'learning_rate':float(self.optimizer.param_groups[0]['lr'])}
            with (OUT/'TRAIN_HISTORY.jsonl').open('a') as f:f.write(json.dumps(history,ensure_ascii=False,allow_nan=False)+'\n')
        self.current_epoch+=1

    def stop_loaders(self):
        for loader in [self.dataloader_train,self.dataloader_val]:
            if hasattr(loader,'_finish'):loader._finish()
