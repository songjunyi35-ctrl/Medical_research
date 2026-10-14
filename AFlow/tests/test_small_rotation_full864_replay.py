import torch
from sammed3d.small_rotation import SmallRotation
from scripts.audit_small_rotation_full864_training import rotation_stream


def test_independent_stream_matches_actual_draws():
    torch.set_num_threads(1)
    image=torch.zeros(8,1,8,8,8);label=image.clone();label[:,:,2:6,2:6,2:6]=1
    rotation=SmallRotation(2026091071)
    for _ in range(3):rotation(image,label)
    expected,count=rotation_stream(2026091071,3)
    assert rotation.receipt()['independent_random_stream_sha256']==expected
    assert rotation.receipt()['selected']==count
