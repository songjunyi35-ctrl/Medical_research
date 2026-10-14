# -*- coding: utf-8 -*-
"""
Phase 0 验收：用解耦后的 prompt_runner + RITMPolicy，在 tumor_WT 的若干例上
复现 baseline 的 Dice（同例对比 results/tumor_WT.json）。
"""
import argparse, glob, json, os
import numpy as np
import torch
import torchio as tio

from segment_anything.build_sam3D import sam_model_registry3D
from prompt_runner import run_sammed3d, RITMPolicy

ap = argparse.ArgumentParser()
ap.add_argument("--target", default="tumor_WT")
ap.add_argument("--n", type=int, default=20)
ap.add_argument("--num_clicks", type=int, default=5)
ap.add_argument("--ckpt", default="./ckpt/sam_med3d_turbo.pth")
args = ap.parse_args()
device = "cuda"

# 模型
model = sam_model_registry3D["vit_b_ori"](checkpoint=None).to(device)
model.load_state_dict(torch.load(args.ckpt, map_location=device)["model_state_dict"])
model.eval()

# baseline 每例 max-dice（按 basename 索引）
base_raw = json.load(open(f"results/{args.target}.json"))[args.target]
base = {os.path.basename(k): max(v.values()) for k, v in base_raw.items()}

img_dir = f"data/medical_preprocessed/{args.target}/BraTS2024/imagesTr"
cases = sorted(glob.glob(img_dir + "/*.nii.gz"))[: args.n]
transform = tio.Compose([
    tio.ToCanonical(),
    tio.CropOrPad(mask_name="label", target_shape=(128, 128, 128)),
])

mine, theirs = [], []
print(f"{'case':<28} {'mine':>7} {'base':>7} {'diff':>7}")
for img_path in cases:
    lbl_path = img_path.replace("imagesTr", "labelsTr")
    subj = transform(tio.Subject(image=tio.ScalarImage(img_path),
                                 label=tio.LabelMap(lbl_path)))
    image3D = subj.image.data.unsqueeze(0).float()
    gt3D = subj.label.data.unsqueeze(0).float()

    torch.manual_seed(2023)
    _, dice_list, _ = run_sammed3d(model, image3D, gt3D, RITMPolicy(),
                                   num_clicks=args.num_clicks, device=device)
    m = max(dice_list)
    bn = os.path.basename(img_path)
    b = base.get(bn, float("nan"))
    mine.append(m); theirs.append(b)
    print(f"{bn:<28} {m:>7.4f} {b:>7.4f} {m-b:>+7.4f}")

mine = np.array(mine); theirs = np.array(theirs)
print("-" * 56)
print(f"我的均值   : {mine.mean():.4f}  (n={len(mine)})")
print(f"baseline均值: {np.nanmean(theirs):.4f}")
print(f"均值差     : {mine.mean()-np.nanmean(theirs):+.4f}")
print(f"逐例|差|均值: {np.nanmean(np.abs(mine-theirs)):.4f}")
print("(均值差接近 0 即说明解耦未改变行为，Phase 0 通过)")
print("PHASE0_DONE")
