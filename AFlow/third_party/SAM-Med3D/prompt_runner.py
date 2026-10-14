# -*- coding: utf-8 -*-
"""
Phase 0: 把 SAM-Med3D 的推理循环从 validation.py 解耦出来，
点 prompt 的来源改为可插拔的 PromptPolicy（为后续 AFlow 自进化做地基）。

设计目标：用 RITMPolicy 时，逐步逻辑与 validation.py:finetune_model_predict3D 完全一致，
从而能复现 BASELINE_RESULTS.md 里的 Dice。
"""
import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio

from utils.click_method import (
    get_next_click3D_torch_ritm,   # 'default'/'ritm'
    get_next_click3D_torch_2,      # 'random'
)

# ---- 指标（与 validation.py 同实现）----
def compute_iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / union if union > 0 else 0.0

def compute_dice(gt, pred):
    gt = gt.astype(np.uint8); pred = pred.astype(np.uint8)
    inter = (gt & pred).sum()
    s = gt.sum() + pred.sum()
    return 2 * inter / s if s > 0 else 0.0


# ===================== 可插拔 Prompt 策略接口 =====================
class PromptPolicy:
    """workflow 最终产出的对象。next_click 在每个 click step 被调用。"""
    def next_click(self, prev_seg, gt, image, history, step):
        """返回 (batch_points, batch_labels)：
           batch_points: list[Tensor(1,1,3) float 体素坐标]
           batch_labels: list[Tensor(1,1) int64, 1=正点 0=负点]"""
        raise NotImplementedError


class RITMPolicy(PromptPolicy):
    """复现 validation.py 的默认行为：前 2 个 click 用 RITM(误差距离变换)，
       之后(step>1)切换为 random 误差采样。GT-aware。"""
    def next_click(self, prev_seg, gt, image, history, step):
        method = get_next_click3D_torch_2 if step > 1 else get_next_click3D_torch_ritm
        return method(prev_seg, gt)


# ===================== 解耦后的推理循环 =====================
_norm = tio.ZNormalization(masking_method=lambda x: x > 0)

@torch.no_grad()
def run_sammed3d(model, image3D, gt3D, policy, num_clicks=5, crop_size=128, device="cuda"):
    """
    image3D, gt3D: (1,1,D,H,W) tensors（已 CropOrPad 到 crop_size）。gt3D 可为 None(部署模式)。
    返回 (pred_list, dice_list, iou_list)；与 validation 一致，最终取 max(dice_list)。
    """
    # 与 validation.py:231 一致的强度归一化
    img = _norm(image3D.squeeze(dim=1)).unsqueeze(dim=1).float()

    if gt3D is not None:
        prev = torch.zeros_like(gt3D).float().to(device)
    else:
        prev = torch.zeros_like(img).float().to(device)
    low_res = F.interpolate(prev, size=(crop_size // 4,) * 3)

    image_emb = model.image_encoder(img.to(device))

    pred_list, dice_list, iou_list, history = [], [], [], []
    for step in range(num_clicks):
        pts, lbls = policy.next_click(prev_seg=prev.to(device),
                                      gt=None if gt3D is None else gt3D.to(device),
                                      image=img.to(device), history=history, step=step)
        pco = torch.cat(pts, dim=0).to(device)
        pla = torch.cat(lbls, dim=0).to(device)
        history.append((pco, pla))

        sparse, dense = model.prompt_encoder(points=[pco, pla], boxes=None,
                                             masks=low_res.to(device))
        low_res, _ = model.mask_decoder(
            image_embeddings=image_emb.to(device),
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse,
            dense_prompt_embeddings=dense,
            multimask_output=False,
        )
        prev = F.interpolate(low_res, size=img.shape[-3:], mode="trilinear", align_corners=False)

        seg = (torch.sigmoid(prev).cpu().numpy().squeeze() > 0.5).astype(np.uint8)
        pred_list.append(seg)
        if gt3D is not None:
            g = gt3D[0][0].cpu().numpy().astype(np.uint8)
            dice_list.append(round(compute_dice(g, seg), 4))
            iou_list.append(round(compute_iou(seg, g), 4))

    return pred_list, dice_list, iou_list
