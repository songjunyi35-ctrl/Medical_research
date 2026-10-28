"""Synthetic CPU augmentation throughput; never reads patient arrays or uses CUDA."""
import hashlib
import time
import torch
from scripts.race_seg_20261028.common import CANVAS, R, now, write
from scripts.race_seg_20261028.data import augment_batch


def digest(batch):
    result = {}
    for key, value in batch.items():
        result[key] = hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()
    return result


def main():
    torch.set_num_threads(2)
    generator = torch.Generator().manual_seed(8241)
    image = torch.randn(8, 1, *CANVAS, generator=generator)
    label = torch.zeros_like(image)
    label[:, :, 48:80, 80:100, 64:80] = 1
    classes = label.byte()
    classes[:, :, 85:100, 90:110, 65:85] = 2
    matrix = torch.eye(4).repeat(8, 1, 1)
    matrix[:, 0, 0] = -1
    matrix[:, 0, 3] = CANVAS[0] - 8
    batch = dict(image=image, label=label, classes=classes,
                 valid=torch.ones_like(image), reflection=matrix)
    results = []
    reference = None
    # Repeat the baseline last to expose effects from concurrent live training.
    for threads in (2, 4, 8, 2):
        torch.set_num_threads(threads)
        seconds = []
        for repeat in range(3):
            begin = time.monotonic()
            output = augment_batch(batch, 1847 + repeat)
            seconds.append(time.monotonic() - begin)
            if repeat == 0:
                hashes = digest(output)
                if reference is None:
                    reference = hashes
                assert hashes == reference, 'CPU thread count changed augmented tensor bytes'
            del output
        results.append(dict(threads=threads, seconds=seconds,
                            mean_seconds=sum(seconds[1:])/2,
                            augmented_sha256=hashes))
    write(R/'CPU_PROFILE.json', dict(at=now(), synthetic_only=True,
          patient_array_access=False, patient_forwards=0, cuda_access=False,
          results=results, exact_augmented_bytes_across_threads=True))


if __name__ == '__main__':
    main()
