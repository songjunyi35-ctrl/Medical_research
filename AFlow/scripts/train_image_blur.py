"""Same hash-bound training engine; isolated additional deformation only."""
import sys
from scripts import image_blur_protocol as p
from scripts import train_structured_background_aux as training
from scripts.train_subject_balanced_sampling import balanced_weights
from scripts.train_t1ce_wt_capg3d import augment_batch as base_augment
from scripts.train_t1ce_wt_capg3d import write_json_atomic
from sammed3d.image_blur import ImageBlur


def main():
    p.check_freeze()
    seed=int(sys.argv[sys.argv.index('--seed')+1])
    arm=sys.argv[sys.argv.index('--arm')+1]
    if seed not in p.SEEDS or arm not in p.ARMS:raise ValueError('unregistered run')
    deformation=ImageBlur(seed)
    def augment(image, labels, max_shift):
        image,labels=base_augment(image,labels,max_shift=max_shift)
        result=deformation(image,labels)
        if deformation.steps % 121 == 0:
            write_json_atomic(p.WORK/f'seed{seed}_{arm}/BLUR_PROGRESS.json',deformation.receipt())
        return result
    training.p=p
    training.augment_batch=augment
    if arm=='subject_blur':training.native_sampling_weights=balanced_weights
    try:
        training.main()
    finally:
        p.write_new(p.WORK/f'seed{seed}_{arm}/BLUR_FINAL.json',deformation.receipt())


if __name__=='__main__':main()
