# -*- coding: utf-8 -*-
"""SAM-Med3D inference engine for the AFlow adapter.

Wraps the SAM-Med3D 3D promptable model into a few primitive, *non-LLM*
operations that AFlow operators / workflows can compose:

  - get_model()        : load a checkpoint once per (path, device) -> no OOM
  - load_case()        : load one (image,label) pair exactly like validation.py
  - predict_prob()     : run the iterative-click 3D forward, return prob volume
  - ensemble_prob()    : average several prob volumes (self-consistency analog)
  - prob_to_mask()     : threshold
  - largest_cc()       : keep largest connected component (post-process)
  - dice()             : Dice coefficient

The click sampling reuses SAM-Med3D's own click_methods (they sample the next
click from the GT error region), so behaviour matches the official validation.
"""
import os
import sys
import math
import threading

import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import torchio as tio
from torchio.data.io import sitk_to_nib

from sammed3d.paths import (
    SAMMED3D_CHECKPOINT as _SAMMED3D_CHECKPOINT_PATH,
    SAMMED3D_ROOT as _SAMMED3D_ROOT_PATH,
)

# --- make the SAM-Med3D repo importable -------------------------------------
SAMMED3D_ROOT = str(_SAMMED3D_ROOT_PATH)
if SAMMED3D_ROOT not in sys.path:
    sys.path.insert(0, SAMMED3D_ROOT)

from segment_anything.build_sam3D import sam_model_registry3D  # noqa: E402

# SAM-Med3D's `utils/` has no __init__.py (namespace pkg) and the name `utils`
# is collision-prone, so load click_method.py directly by file path.
import importlib.util as _ilu  # noqa: E402

_cm_path = os.path.join(SAMMED3D_ROOT, "utils", "click_method.py")
_spec = _ilu.spec_from_file_location("sammed3d_click_method", _cm_path)
_cm = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cm)
get_next_click3D_torch_ritm = _cm.get_next_click3D_torch_ritm
get_next_click3D_torch_2 = _cm.get_next_click3D_torch_2

CLICK_METHODS = {
    "default": get_next_click3D_torch_ritm,
    "ritm": get_next_click3D_torch_ritm,
    "random": get_next_click3D_torch_2,
}

DEFAULT_CKPT = str(_SAMMED3D_CHECKPOINT_PATH)
CROP_SIZE = 128
MODEL_TYPE = "vit_b_ori"

_MODEL = None
_MODEL_CACHE = {}
_MODEL_LOCK = threading.Lock()
_NORM = tio.ZNormalization(masking_method=lambda x: x > 0)


def resolve_checkpoint(checkpoint: str = None) -> str:
    """Resolve the checkpoint used by inference/eval.

    Explicit arguments win. Otherwise eval can be redirected with
    AFLOW_SAMMED3D_CHECKPOINT (preferred) or SAMMED3D_CHECKPOINT.
    """
    if checkpoint:
        return checkpoint
    return (
        os.environ.get("AFLOW_SAMMED3D_CHECKPOINT")
        or os.environ.get("SAMMED3D_CHECKPOINT")
        or DEFAULT_CKPT
    )


def load_checkpoint_state_dict(checkpoint: str, map_location):
    payload = torch.load(checkpoint, map_location=map_location)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state = payload["model_state_dict"]
    elif isinstance(payload, dict) and "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    if not isinstance(state, dict):
        raise TypeError(f"checkpoint {checkpoint} does not contain a state dict")
    if any(key.startswith("module.") for key in state):
        state = {
            (key[7:] if key.startswith("module.") else key): value
            for key, value in state.items()
        }
    return state


def clear_model_cache():
    """Drop cached model instances. Useful for long-lived eval processes."""
    global _MODEL
    with _MODEL_LOCK:
        _MODEL_CACHE.clear()
        _MODEL = None


def get_model(checkpoint: str = None, device: str = "cuda", force_reload: bool = False):
    """Load a SAM-Med3D model and cache it by checkpoint path and device."""
    global _MODEL
    checkpoint = resolve_checkpoint(checkpoint)
    cache_key = (os.path.abspath(checkpoint), str(device))
    if not force_reload and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    with _MODEL_LOCK:
        if force_reload or cache_key not in _MODEL_CACHE:
            model = sam_model_registry3D[MODEL_TYPE](checkpoint=None).to(device)
            state = load_checkpoint_state_dict(checkpoint, map_location=device)
            model.load_state_dict(state)
            model.eval()
            _MODEL_CACHE[cache_key] = model
            _MODEL = model
    return _MODEL_CACHE[cache_key]


def load_case(image_path: str, label_path: str):
    """Load one case -> (image3D[1,1,128^3], gt3D[1,1,128^3]) float/int tensors.

    Mirrors SAM-Med3D validation.py preprocessing: ToCanonical + CropOrPad to
    128^3 around the label.
    """
    sitk_image = sitk.ReadImage(image_path)
    sitk_label = sitk.ReadImage(label_path)
    if sitk_image.GetOrigin() != sitk_label.GetOrigin():
        sitk_image.SetOrigin(sitk_label.GetOrigin())
    if sitk_image.GetDirection() != sitk_label.GetDirection():
        sitk_image.SetDirection(sitk_label.GetDirection())

    img_arr, _ = sitk_to_nib(sitk_image)
    lbl_arr, _ = sitk_to_nib(sitk_label)
    subject = tio.Subject(
        image=tio.ScalarImage(tensor=img_arr),
        label=tio.LabelMap(tensor=lbl_arr),
    )
    transform = tio.Compose([
        tio.ToCanonical(),
        tio.CropOrPad(mask_name="label", target_shape=(CROP_SIZE, CROP_SIZE, CROP_SIZE)),
    ])
    subject = transform(subject)

    image3D = subject.image.data.clone().detach().unsqueeze(0)  # (1,1,W,H,D)
    gt3D = subject.label.data.clone().detach().unsqueeze(0)      # (1,1,W,H,D)
    gt3D = (gt3D > 0).short()
    return image3D, gt3D


@torch.no_grad()
def predict_prob(image3D, gt3D, num_clicks: int = 5, click_method: str = "default",
                 seed: int = 2023, device: str = "cuda"):
    """Run the iterative-click 3D forward; return the final sigmoid prob volume.

    Returns prob as float32 numpy array of shape (W,H,D) after `num_clicks`
    clicks (the last, most-refined prediction).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = get_model(device=device)

    img = _NORM(image3D.squeeze(dim=1)).unsqueeze(dim=1).to(device).float()
    gt = gt3D.to(device)

    prev_masks = torch.zeros_like(gt).float().to(device)
    low_res = F.interpolate(prev_masks, size=(CROP_SIZE // 4,) * 3)
    image_embedding = model.image_encoder(img)

    method = click_method
    prob = None
    for click_idx in range(num_clicks):
        if click_idx > 1:
            method = "random"
        batch_points, batch_labels = CLICK_METHODS[method](prev_masks.to(device), gt.to(device))
        points = torch.cat(batch_points, dim=0).to(device)
        labels = torch.cat(batch_labels, dim=0).to(device)

        sparse_emb, dense_emb = model.prompt_encoder(
            points=[points, labels], boxes=None, masks=low_res.to(device))
        low_res, _ = model.mask_decoder(
            image_embeddings=image_embedding.to(device),
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        prev_masks = F.interpolate(low_res, size=gt.shape[-3:], mode="trilinear", align_corners=False)
        prob = torch.sigmoid(prev_masks).cpu().numpy().squeeze()
    return prob.astype(np.float32)


def _run_click_sequence_with_embedding(model, image_embedding, gt, num_clicks: int,
                                       click_method: str, seed: int, device: str):
    """Run one seeded click trajectory using a precomputed image embedding."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    prev_masks = torch.zeros_like(gt).float().to(device)
    low_res = F.interpolate(prev_masks, size=(CROP_SIZE // 4,) * 3)

    method = click_method
    probs = []
    for click_idx in range(num_clicks):
        if click_idx > 1:
            method = "random"
        batch_points, batch_labels = CLICK_METHODS[method](prev_masks.to(device), gt.to(device))
        points = torch.cat(batch_points, dim=0).to(device)
        labels = torch.cat(batch_labels, dim=0).to(device)

        sparse_emb, dense_emb = model.prompt_encoder(
            points=[points, labels], boxes=None, masks=low_res.to(device))
        low_res, _ = model.mask_decoder(
            image_embeddings=image_embedding.to(device),
            image_pe=model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_emb,
            dense_prompt_embeddings=dense_emb,
            multimask_output=False,
        )
        prev_masks = F.interpolate(low_res, size=gt.shape[-3:], mode="trilinear", align_corners=False)
        prob = torch.sigmoid(prev_masks).cpu().numpy().squeeze()
        probs.append(prob.astype(np.float32))
    return probs


@torch.no_grad()
def predict_prob_sequence(image3D, gt3D, num_clicks: int = 5, click_method: str = "default",
                          seed: int = 2023, device: str = "cuda"):
    """Run iterative-click inference and return the probability volume after every click.

    This exposes the same trajectory that SAM-Med3D's validation.py evaluates,
    without choosing a mask by ground-truth Dice. Workflows can combine or rank
    these intermediate predictions with GT-free heuristics.
    """
    model = get_model(device=device)

    img = _NORM(image3D.squeeze(dim=1)).unsqueeze(dim=1).to(device).float()
    gt = gt3D.to(device)
    image_embedding = model.image_encoder(img)
    return _run_click_sequence_with_embedding(
        model,
        image_embedding,
        gt,
        num_clicks=num_clicks,
        click_method=click_method,
        seed=seed,
        device=device,
    )


@torch.no_grad()
def predict_prob_multi_seed(image3D, gt3D, seeds, num_clicks: int = 5,
                            click_method: str = "default", device: str = "cuda"):
    """Return final probability volumes for several seeds, reusing image encoding.

    This is equivalent to calling `predict_prob` for each seed, except the costly
    image encoder is run once per case instead of once per seed.
    """
    model = get_model(device=device)
    img = _NORM(image3D.squeeze(dim=1)).unsqueeze(dim=1).to(device).float()
    gt = gt3D.to(device)
    image_embedding = model.image_encoder(img)
    probs = []
    for seed in seeds:
        trajectory = _run_click_sequence_with_embedding(
            model,
            image_embedding,
            gt,
            num_clicks=num_clicks,
            click_method=click_method,
            seed=int(seed),
            device=device,
        )
        probs.append(trajectory[-1])
    return probs


@torch.no_grad()
def predict_prob_sequence_multi_seed(image3D, gt3D, seeds, num_clicks: int = 5,
                                     click_method: str = "default", device: str = "cuda"):
    """Return click trajectories for several seeds, reusing image encoding."""
    model = get_model(device=device)
    img = _NORM(image3D.squeeze(dim=1)).unsqueeze(dim=1).to(device).float()
    gt = gt3D.to(device)
    image_embedding = model.image_encoder(img)
    return [
        _run_click_sequence_with_embedding(
            model,
            image_embedding,
            gt,
            num_clicks=num_clicks,
            click_method=click_method,
            seed=int(seed),
            device=device,
        )
        for seed in seeds
    ]


def ensemble_prob(probs):
    """Average a list of prob volumes (soft-vote self-consistency)."""
    return np.mean(np.stack(probs, axis=0), axis=0)


def transform_prob(prob, smooth_sigma: float = 0.0, gamma: float = 1.0):
    """GT-free probability calibration: optional Gaussian smoothing and gamma curve."""
    p = np.asarray(prob, dtype=np.float32)
    if smooth_sigma and smooth_sigma > 0:
        try:
            from scipy import ndimage
            p = ndimage.gaussian_filter(p, sigma=float(smooth_sigma)).astype(np.float32)
        except Exception:
            pass
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    if gamma and gamma > 0 and abs(gamma - 1.0) > 1e-6:
        p = np.power(p, float(gamma)).astype(np.float32)
    return np.clip(p, 0.0, 1.0).astype(np.float32)


def otsu_threshold(prob, bins: int = 256):
    """Otsu threshold for a probability volume, implemented without extra deps."""
    p = np.asarray(prob, dtype=np.float32).ravel()
    p = p[np.isfinite(p)]
    if p.size == 0:
        return 0.5
    hist, edges = np.histogram(p, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total <= 0:
        return 0.5
    centers = (edges[:-1] + edges[1:]) / 2.0
    weight_bg = np.cumsum(hist)
    weight_fg = total - weight_bg
    mean_bg = np.cumsum(hist * centers) / np.maximum(weight_bg, 1e-12)
    mean_fg = (np.cumsum((hist * centers)[::-1]) / np.maximum(np.cumsum(hist[::-1]), 1e-12))[::-1]
    between = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
    idx = int(np.nanargmax(between))
    return float(centers[idx])


def prob_to_mask(prob, threshold: float = 0.5):
    return (prob > threshold).astype(np.uint8)


def threshold_sweep(prob, thresholds):
    """Create candidate masks from several fixed thresholds."""
    return [prob_to_mask(prob, float(t)) for t in thresholds]


def largest_cc(mask):
    """Keep only the largest connected component of a binary mask."""
    try:
        from scipy import ndimage
    except Exception:
        return mask
    if mask.sum() == 0:
        return mask
    labeled, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, n + 1))
    keep = np.argmax(sizes) + 1
    return (labeled == keep).astype(np.uint8)


def refine_mask(mask, min_component_voxels: int = 0, max_components: int = None,
                close_iters: int = 0, open_iters: int = 0, fill_holes: bool = False):
    """Morphological cleanup and component filtering, all without model training."""
    out = (np.asarray(mask) > 0).astype(np.uint8)
    try:
        from scipy import ndimage
    except Exception:
        return out

    structure = np.ones((3, 3, 3), dtype=bool)
    if close_iters and close_iters > 0:
        out = ndimage.binary_closing(out, structure=structure, iterations=int(close_iters))
    if open_iters and open_iters > 0:
        out = ndimage.binary_opening(out, structure=structure, iterations=int(open_iters))
    if fill_holes:
        out = ndimage.binary_fill_holes(out)

    out = out.astype(np.uint8)
    if out.sum() == 0:
        return out

    labeled, n = ndimage.label(out)
    if n == 0:
        return out
    sizes = ndimage.sum(out, labeled, range(1, n + 1))
    order = np.argsort(sizes)[::-1]
    keep = []
    for idx in order:
        component_id = int(idx) + 1
        size = int(sizes[idx])
        if min_component_voxels and size < int(min_component_voxels):
            continue
        keep.append(component_id)
        if max_components is not None and len(keep) >= int(max_components):
            break
    if not keep:
        keep = [int(order[0]) + 1]
    return np.isin(labeled, keep).astype(np.uint8)


def mask_consensus(masks, vote_threshold: float = 0.5):
    """Voxel-wise vote over candidate binary masks."""
    masks = [(np.asarray(mask) > 0).astype(np.float32) for mask in masks]
    if not masks:
        raise ValueError("mask_consensus requires at least one mask")
    vote = np.mean(np.stack(masks, axis=0), axis=0)
    return (vote >= float(vote_threshold)).astype(np.uint8)


def mask_quality_score(prob, mask, min_volume: int = 8, max_volume_ratio: float = 0.20,
                       component_penalty: float = 0.03, volume_penalty: float = 0.10):
    """Score a candidate mask using only model probability and shape priors.

    Higher is better. This is intentionally heuristic: AFlow/LLM can search the
    surrounding candidate generation and post-processing parameters.
    """
    p = np.asarray(prob, dtype=np.float32)
    m = (np.asarray(mask) > 0)
    volume = int(m.sum())
    total = int(m.size)
    if volume <= 0:
        return -1e9

    fg = float(p[m].mean()) if volume else 0.0
    bg = float(p[~m].mean()) if volume < total else 0.0
    margin = fg - bg
    entropy = -p * np.log(np.clip(p, 1e-6, 1.0)) - (1.0 - p) * np.log(np.clip(1.0 - p, 1e-6, 1.0))
    confidence = 1.0 - float(entropy[m].mean() / math.log(2.0))

    component_count = 1
    try:
        from scipy import ndimage
        _, component_count = ndimage.label(m)
    except Exception:
        pass

    score = margin + 0.25 * fg + 0.10 * confidence
    if volume < int(min_volume):
        score -= volume_penalty * (int(min_volume) - volume) / max(int(min_volume), 1)
    if total > 0 and volume / total > float(max_volume_ratio):
        score -= volume_penalty * ((volume / total) - float(max_volume_ratio)) / max(float(max_volume_ratio), 1e-6)
    score -= component_penalty * max(component_count - 1, 0)
    return float(score)


def select_mask_candidate(masks, prob, min_volume: int = 8, max_volume_ratio: float = 0.20):
    """Pick the best candidate with a GT-free confidence/shape score."""
    if not masks:
        raise ValueError("select_mask_candidate requires at least one mask")
    scores = [
        mask_quality_score(prob, mask, min_volume=min_volume, max_volume_ratio=max_volume_ratio)
        for mask in masks
    ]
    best_idx = int(np.argmax(scores))
    return (np.asarray(masks[best_idx]) > 0).astype(np.uint8), float(scores[best_idx]), best_idx


def dice(gt_mask, pred_mask):
    """Soerensen-Dice. Both empty -> NaN (matches validation.py)."""
    gt = (np.asarray(gt_mask) > 0).astype(np.uint8)
    pred = (np.asarray(pred_mask) > 0).astype(np.uint8)
    vol = gt.sum() + pred.sum()
    if vol == 0:
        return float("nan")
    return float(2.0 * (gt & pred).sum() / vol)


@torch.no_grad()
def official_click_eval(
    image3D,
    gt3D,
    num_clicks: int = 5,
    click_method: str = "default",
    seed: int = 2023,
    threshold: float = 0.5,
    device: str = "cuda",
):
    """Match SAM-Med3D validation.py's 3D click metric for one case.

    The upstream script records Dice after every click and reports max(dice_list)
    for the case. This helper keeps that metric explicit so it is not confused
    with AFlow's normal "score the workflow-returned mask" objective.
    """
    probs = predict_prob_sequence(
        image3D,
        gt3D,
        num_clicks=num_clicks,
        click_method=click_method,
        seed=seed,
        device=device,
    )
    gt = gt3D.detach().cpu().numpy().squeeze().astype(np.uint8)
    masks = [prob_to_mask(prob, threshold=threshold) for prob in probs]
    per_click_dice = [dice(gt, mask) for mask in masks]
    if not per_click_dice:
        return {
            "best_dice": float("nan"),
            "best_click": -1,
            "final_dice": float("nan"),
            "per_click_dice": [],
            "masks": [],
        }
    finite = np.asarray([
        score if score == score else -np.inf for score in per_click_dice
    ])
    best_click = int(np.argmax(finite))
    return {
        "best_dice": float(per_click_dice[best_click]),
        "best_click": best_click,
        "final_dice": float(per_click_dice[-1]),
        "per_click_dice": [float(score) for score in per_click_dice],
        "masks": masks,
    }
