"""
CVAE 推理。

流程对齐 `inference_spatial_anomaly_prior.py`：
- 以 good 二值前景作为条件输入
- 从 latent 采样多个候选
- 做候选后处理与筛选
- 支持按类别批量推理与可视化输出
"""

import argparse
import glob
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from torch.utils.data import DataLoader

from code4maskgeneration.cvae import CondVAE
from code4maskgeneration.mvtec_mask_dataset import (
    Mask2MaskDataset,
    list_seg_mask_categories,
    list_training_defect_types,
)
from code4maskgeneration.train_mask_generation_cvae import (
    REL_HIST_BINS_DT,
    REL_HIST_BINS_WIDTH,
    UV_HIST_BINS,
    _cond_morphology_batch,
    _cond_uv_maps_batch,
    _rel_hist_loss,
    _rel_histogram,
    _uv_histogram,
    compute_dataset_rel_hist_prior,
    compute_dataset_uv_hist_prior,
)

DEFAULT_DATASET_ROOT = "/home/wyh/data/mvtec_ad/seg_mask"
DEFAULT_OUTPUT_ROOT = "/d242/wyh/M2M"
DEFAULT_INFER_MASK_ROOT = "/home/wyh/data/mvtec_ad"
WEIGHTS_SUBDIR = "genmask_cvae"
IMG_SIZE = 512
LATENT_DIM = 64

DROPOUT_P = 0.1
CANDIDATE_POOL = 5
MAX_RESAMPLE_ROUNDS = 20
TEMPERATURE = 1.0
THRESH = 0.5
MIN_CC_AREA_RATIO = 0.2
MAX_AREA_RATIO = 1.0
OVERLAY_ALPHA = 0.35
INFER_UV_BONUS = 2.5


def weights_dir_for_date(output_root: str, date_code: str) -> str:
    return str(Path(output_root) / WEIGHTS_SUBDIR / f"weights_{date_code}")


def infer_out_dir_for_date(infer_mask_root: str, date_code: str) -> str:
    return str(Path(infer_mask_root) / f"generated_mask_{date_code}")


def resolve_date_code(date_code: str | None) -> str:
    """脚本内只调用一次，避免 parse 默认值与后续逻辑各取各的日期。"""
    return date_code or datetime.now().strftime("%Y%m%d")


def build_cond_tensor(binary_fg_path, device):
    gray = cv2.imread(binary_fg_path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(binary_fg_path)
    pil = Image.fromarray(gray)
    fg = TF.to_tensor(pil)
    fg = TF.resize(fg, [IMG_SIZE, IMG_SIZE], interpolation=TF.InterpolationMode.NEAREST)
    fg = (fg > 0.5).float().unsqueeze(0).to(device)
    return fg


def sample_priors(model, cond, latent_dim, num_samples, temperature, device):
    model.eval()
    outs = []
    with torch.no_grad():
        for _ in range(num_samples):
            z = torch.randn(cond.size(0), latent_dim, device=device) * float(temperature)
            out = model.decode(z, cond)
            outs.append(out.squeeze().cpu().numpy())
    return outs


def _list_component_masks(bin_m, min_area_ratio=MIN_CC_AREA_RATIO, reject_border_components=True):
    """返回通过面积/边界筛选的各连通域二值 mask 列表。"""
    if bin_m.dtype != np.uint8:
        bin_m = (bin_m > 0).astype(np.uint8) * 255
    else:
        bin_m = ((bin_m > 0).astype(np.uint8)) * 255

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_m)
    if num_labels <= 1:
        return [bin_m] if int(bin_m.sum()) > 0 else []

    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return []

    max_area = int(areas.max())
    keep_thr = max(1, int(max_area * float(min_area_ratio)))
    h, w = bin_m.shape[:2]
    comps = []
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < keep_thr:
            continue
        if reject_border_components:
            x, y, bw, bh, _ = stats[i]
            touches_border = x <= 0 or y <= 0 or (x + bw) >= w or (y + bh) >= h
            if touches_border:
                continue
        comp = np.zeros_like(bin_m)
        comp[labels == i] = 255
        comps.append(comp)
    return comps


def select_best_connected_component(
    mask_u8,
    prob,
    cond_u8,
    cond_t,
    rel_hist_prior,
    uv_hist_prior,
    device,
    min_area=None,
    max_area=None,
    target_inside_cond=None,
    min_area_ratio=MIN_CC_AREA_RATIO,
    reject_border_components=True,
):
    """多连通域时按与训练一致的打分只保留最佳一块（减轻中心伪影+真位置双峰）。"""
    comps = _list_component_masks(
        mask_u8,
        min_area_ratio=min_area_ratio,
        reject_border_components=reject_border_components,
    )
    if not comps:
        return np.zeros_like(mask_u8, dtype=np.uint8)

    best_mask = comps[0]
    best_score = -1e9
    for comp in comps:
        score = _score_candidate(
            prob,
            comp,
            cond_u8,
            cond_t,
            rel_hist_prior,
            uv_hist_prior,
            device,
            min_area=min_area,
            max_area=max_area,
            target_inside_cond=target_inside_cond,
        )
        if score > best_score:
            best_score = score
            best_mask = comp
    return best_mask


def postprocess_binary(
    mask_u8,
    prob=None,
    cond_u8=None,
    cond_t=None,
    rel_hist_prior=None,
    uv_hist_prior=None,
    device=None,
    min_area=None,
    max_area=None,
    target_inside_cond=None,
    min_area_ratio=MIN_CC_AREA_RATIO,
    reject_border_components=True,
):
    if (
        prob is not None
        and cond_u8 is not None
        and cond_t is not None
        and rel_hist_prior is not None
        and uv_hist_prior is not None
        and device is not None
    ):
        return select_best_connected_component(
            mask_u8,
            prob,
            cond_u8,
            cond_t,
            rel_hist_prior,
            uv_hist_prior,
            device,
            min_area=min_area,
            max_area=max_area,
            target_inside_cond=target_inside_cond,
            min_area_ratio=min_area_ratio,
            reject_border_components=reject_border_components,
        )
    comps = _list_component_masks(mask_u8, min_area_ratio, reject_border_components)
    if not comps:
        return np.zeros_like(mask_u8, dtype=np.uint8)
    out = np.zeros_like(comps[0])
    for comp in comps:
        out = np.maximum(out, comp)
    return out


def is_valid_mask(mask_u8, min_area_ratio=0.3, max_area_ratio=1.0):
    bin_m = (mask_u8 > 0).astype(np.uint8) * 255
    area = int((bin_m > 0).sum())
    if area <= 0:
        return False, "empty"
    if _touches_image_border(bin_m):
        return False, "touches_border"
    h, w = bin_m.shape[:2]
    max_area = int(h * w * float(max_area_ratio))
    min_area = max(1, int(h * w * float(min_area_ratio)))
    if area < min_area:
        return False, f"area_too_small({area}<{min_area})"
    if area > max_area:
        return False, f"area_too_large({area}>{max_area})"
    return True, "ok"


def _mask_stats(mask_u8, ref_u8=None):
    mask = (mask_u8 > 0).astype(np.uint8)
    area = float(mask.sum())
    h, w = mask.shape[:2]
    if area <= 0:
        return {
            "area": 0.0,
            "centroid": (0.5, 0.5),
            "touches_border": False,
            "iou": 0.0,
            "overlap_ratio": 0.0,
        }
    ys, xs = np.nonzero(mask)
    cy = float(ys.mean() / max(h - 1, 1))
    cx = float(xs.mean() / max(w - 1, 1))
    touches_border = bool((mask[0, :] > 0).any() or (mask[-1, :] > 0).any() or (mask[:, 0] > 0).any() or (mask[:, -1] > 0).any())
    iou = 0.0
    overlap_ratio = 0.0
    if ref_u8 is not None:
        ref = (ref_u8 > 0).astype(np.uint8)
        inter = float((mask & ref).sum())
        union = float((mask | ref).sum()) if float((mask | ref).sum()) > 0 else 1.0
        ref_area = float(ref.sum()) if float(ref.sum()) > 0 else 1.0
        iou = inter / union
        overlap_ratio = inter / ref_area
    return {
        "area": area,
        "centroid": (cy, cx),
        "touches_border": touches_border,
        "iou": iou,
        "overlap_ratio": overlap_ratio,
    }


def _rel_hist_score(prob, cond_t, rel_hist_prior, device):
    recon = torch.from_numpy(prob).float().to(device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        dt, width = _cond_morphology_batch(cond_t)
        pred_hist = _rel_histogram(recon, dt, width)
        prior_t = torch.from_numpy(rel_hist_prior).to(device)
        loss = _rel_hist_loss(pred_hist, prior_t).item()
    return 1.0 - loss


def _uv_hist_score(prob, cond_t, uv_hist_prior, device):
    recon = torch.from_numpy(prob).float().to(device).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        u_map, v_map = _cond_uv_maps_batch(cond_t)
        pred_hist = _uv_histogram(recon, u_map, v_map)
        prior_t = torch.from_numpy(uv_hist_prior).to(device)
        loss = _rel_hist_loss(pred_hist, prior_t).item()
    return 1.0 - loss


def _score_candidate(
    prob,
    bin_pp,
    cond_u8,
    cond_t,
    rel_hist_prior,
    uv_hist_prior,
    device,
    min_area=None,
    max_area=None,
    target_inside_cond=None,
):
    bin_u8 = ((bin_pp > 0).astype(np.uint8))
    area = float(bin_u8.sum())
    if area <= 0:
        return -1e9

    cond = (cond_u8 > 0).astype(np.uint8)
    pred_mass = max(float(bin_u8.sum()), 1.0)
    inside_ratio = float((bin_u8 & cond).sum()) / pred_mass
    target_inside = 1.0 if target_inside_cond is None else float(target_inside_cond)
    match_bonus = 1.0 - abs(inside_ratio - target_inside)
    excess_outside = max(0.0, target_inside - inside_ratio)

    stats = _mask_stats(bin_u8 * 255)
    border_penalty = 0.5 if stats["touches_border"] else 0.0
    prob_bonus = float(prob[bin_u8 > 0].mean()) if (bin_u8 > 0).any() else float(prob.mean()) * 0.5

    area_pen = 0.0
    if min_area is not None and area < float(min_area):
        area_pen += (float(min_area) - area) / float(IMG_SIZE * IMG_SIZE)
    if max_area is not None and area > float(max_area):
        area_pen += (area - float(max_area)) / float(IMG_SIZE * IMG_SIZE)

    rel_bonus = _rel_hist_score(prob, cond_t, rel_hist_prior, device)
    uv_bonus = _uv_hist_score(prob, cond_t, uv_hist_prior, device)

    return float(
        prob_bonus
        + 3.0 * match_bonus
        + 2.5 * rel_bonus
        + INFER_UV_BONUS * uv_bonus
        - 2.0 * excess_outside
        - border_penalty
        - area_pen
    )


def blend_red_overlay(bgr, mask_u8, alpha):
    a = float(alpha)
    a = 0.0 if a < 0 else (1.0 if a > 1.0 else a)
    if a <= 0:
        return bgr
    m = (mask_u8.astype(np.float32) / 255.0)[..., None]
    blend = m * a
    out = bgr.astype(np.float32).copy()
    red = np.zeros_like(out)
    red[..., 2] = 255.0
    out = out * (1.0 - blend) + red * blend
    return np.clip(out, 0, 255).astype(np.uint8)


def find_image_for_stem(image_dir, stem):
    if not image_dir:
        return None
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
        p = os.path.join(image_dir, stem + ext)
        if os.path.isfile(p):
            return p
    return None


def _touches_image_border(mask_u8):
    if mask_u8.size == 0:
        return False
    return bool((mask_u8[0, :] > 0).any() or (mask_u8[-1, :] > 0).any() or (mask_u8[:, 0] > 0).any() or (mask_u8[:, -1] > 0).any())


def collect_gt_area_range(dataset_root, category, defect, img_size=IMG_SIZE):
    gt_dir = os.path.join(dataset_root, category, "Ground_truth", defect)
    if not os.path.isdir(gt_dir):
        gt_dir = os.path.join(dataset_root, category, "ground_truth", defect)
    if not os.path.isdir(gt_dir):
        return None

    areas = []
    for fp in sorted(glob.glob(os.path.join(gt_dir, "*.png"))):
        gray = cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        pil = Image.fromarray(gray)
        mask = TF.to_tensor(pil)
        mask = TF.resize(mask, [img_size, img_size], interpolation=TF.InterpolationMode.NEAREST)
        mask = (mask > 0.5).float().squeeze(0).cpu().numpy().astype(np.uint8)
        areas.append(int(mask.sum()))

    if not areas:
        return None

    return {
        "min": int(min(areas)),
        "max": int(max(areas)),
        "mean": float(np.mean(areas)),
        "count": len(areas),
    }


def _load_rel_hist_prior(dataset_root, category, defect, img_size=IMG_SIZE):
    dataset = Mask2MaskDataset(
        root_dir=dataset_root,
        category=category,
        defect_type=defect,
        img_size=img_size,
        augment=False,
    )
    if len(dataset) == 0:
        n = REL_HIST_BINS_DT * REL_HIST_BINS_WIDTH
        return np.ones(n, dtype=np.float32) / n
    return compute_dataset_rel_hist_prior(dataset)


def _load_uv_hist_prior(dataset_root, category, defect, img_size=IMG_SIZE):
    dataset = Mask2MaskDataset(
        root_dir=dataset_root,
        category=category,
        defect_type=defect,
        img_size=img_size,
        augment=False,
    )
    if len(dataset) == 0:
        n = UV_HIST_BINS * UV_HIST_BINS
        return np.ones(n, dtype=np.float32) / n
    return compute_dataset_uv_hist_prior(dataset)


def collect_gt_cond_inside_stats(dataset_root, category, defect, img_size=IMG_SIZE):
    """GT 异常落在 cond（test 前景）内的质量占比，与训练统计一致。"""
    dataset = Mask2MaskDataset(
        root_dir=dataset_root,
        category=category,
        defect_type=defect,
        img_size=img_size,
        augment=False,
    )
    if len(dataset) == 0:
        return None

    ratios = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for cond, gt in loader:
        cond_bin = (cond > 0.5).float()
        gt_bin = (gt > 0.5).float()
        gt_mass = float(gt_bin.sum().item())
        if gt_mass < 1.0:
            continue
        inside = float((gt_bin * cond_bin).sum().item())
        ratios.append(inside / gt_mass)

    if not ratios:
        return None

    return {
        "min": float(min(ratios)),
        "max": float(max(ratios)),
        "mean": float(np.mean(ratios)),
        "count": len(ratios),
    }


def pick_best_mask(
    prob_list,
    thresh,
    cond,
    rel_hist_prior,
    uv_hist_prior,
    device,
    min_area=None,
    max_area=None,
    target_inside_cond=None,
):
    cond_u8 = (cond.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
    best = None
    best_score = -1e9
    fallback = None
    fallback_score = -1e9
    for prob in prob_list:
        bin_m = ((prob > thresh).astype(np.uint8)) * 255
        bin_pp = select_best_connected_component(
            bin_m,
            prob,
            cond_u8,
            cond,
            rel_hist_prior,
            uv_hist_prior,
            device,
            min_area=min_area,
            max_area=max_area,
            target_inside_cond=target_inside_cond,
        )
        area = int((bin_pp > 0).sum())
        valid = area > 0 and not _touches_image_border(bin_pp)
        if min_area is not None:
            valid = valid and area >= int(min_area)
        if max_area is not None:
            valid = valid and area <= int(max_area)
        if valid:
            score = _score_candidate(
                prob,
                bin_pp,
                cond_u8,
                cond,
                rel_hist_prior,
                uv_hist_prior,
                device,
                min_area=min_area,
                max_area=max_area,
                target_inside_cond=target_inside_cond,
            )
            if score > best_score:
                best_score = score
                best = {"bin": bin_pp, "prob": prob, "score": score, "reason": "ok"}
            continue

        raw_area = int((bin_m > 0).sum())
        raw_score = float(prob[bin_m > 0].mean()) if raw_area > 0 else float(prob.mean()) * 0.5 + raw_area * 1e-6
        raw_score -= 0.25 if _touches_image_border(bin_m) else 0.0
        if raw_score > fallback_score:
            fallback_score = raw_score
            fallback = {"bin": bin_m, "prob": prob, "score": raw_score, "reason": "invalid_candidate"}

    return best if best is not None else fallback


def match_weight_to_category(weight_path, category):
    """严格匹配 `类别名_缺陷名.pth` 形式。

    例如：
    - zipper_squeezed_teeth.pth
    - metal_nut_bent.pth

    不接受：
    - mask_generation_cvae_zipper_squeezed_teeth.pth
    - zipper.pth
    - zipper_squeezed_teeth.pt
    """
    name = os.path.basename(weight_path)
    expected = f"{category}_"
    return name.startswith(expected) and name.endswith(".pth") and len(name) > len(expected) + 4


def defect_from_weight_filename(weight_filename, category):
    """从权重文件名 `{category}_{defect}.pth` 解析缺陷名。"""
    name = os.path.basename(weight_filename)
    expected = f"{category}_"
    if not name.startswith(expected) or not name.endswith(".pth"):
        return None
    defect = name[len(expected) : -len(".pth")]
    if not defect:
        return None
    for suffix in ("_best", "_final"):
        if defect.endswith(suffix):
            return defect[: -len(suffix)]
    return defect


def run_one_mask(model_paths, cond, category, stem, bg_path, out_root, device, dataset_root):
    for wp in model_paths:
        name = os.path.basename(wp)
        expected = f"{category}_"
        if not name.startswith(expected) or not name.endswith(".pth"):
            continue
        defect = defect_from_weight_filename(name, category)
        if not defect:
            continue
        output_tag = defect
        model = CondVAE(latent_dim=LATENT_DIM, cond_channels=1, dropout_p=DROPOUT_P).to(device)
        model.load_state_dict(torch.load(wp, map_location=device))

        area_range = collect_gt_area_range(dataset_root, category, defect, img_size=IMG_SIZE)
        cond_inside = collect_gt_cond_inside_stats(dataset_root, category, defect, img_size=IMG_SIZE)
        rel_hist_prior = _load_rel_hist_prior(dataset_root, category, defect)
        uv_hist_prior = _load_uv_hist_prior(dataset_root, category, defect)
        print(
            f"[*] {category}/{defect}: rel_hist peak={rel_hist_prior.max():.4f}, "
            f"uv_hist peak={uv_hist_prior.max():.4f} (bins={UV_HIST_BINS})"
        )
        if area_range is None:
            print(f"[!] {category}/{defect}: 未找到 gt 面积统计，跳过面积约束")
        else:
            print(f"[*] {category}/{defect}: gt area range = [{area_range['min']}, {area_range['max']}], count={area_range['count']}")
        if cond_inside is None:
            print(f"[!] {category}/{defect}: 未找到 gt/cond 重叠统计，cond 内占比目标默认 1.0")
        else:
            print(
                f"[*] {category}/{defect}: gt 在 cond 内质量占比 "
                f"min={cond_inside['min']:.3f}, max={cond_inside['max']:.3f}, mean={cond_inside['mean']:.3f}"
            )

        min_area = area_range["min"] if area_range else None
        max_area = area_range["max"] if area_range else None
        target_inside_cond = cond_inside["mean"] if cond_inside else None

        best = None
        for round_idx in range(MAX_RESAMPLE_ROUNDS):
            probs = sample_priors(model, cond, LATENT_DIM, CANDIDATE_POOL, TEMPERATURE, device)
            candidate = pick_best_mask(
                probs,
                THRESH,
                cond,
                rel_hist_prior,
                uv_hist_prior,
                device,
                min_area=min_area,
                max_area=max_area,
                target_inside_cond=target_inside_cond,
            )
            if candidate is None:
                continue
            area = int((candidate["bin"] > 0).sum())
            border_bad = _touches_image_border(candidate["bin"])
            area_bad = (min_area is not None and area < int(min_area)) or (max_area is not None and area > int(max_area))
            if not border_bad and not area_bad:
                candidate["reason"] = "ok"
                best = candidate
                break

        if best is None:
            cond_fg_u8 = (cond.squeeze().cpu().numpy() > 0.5).astype(np.uint8) * 255
            best = {"bin": np.zeros_like(cond_fg_u8, dtype=np.uint8), "prob": np.zeros_like(cond_fg_u8, dtype=np.float32), "score": -1e9, "reason": "no_valid_candidate"}

        sub = os.path.join(out_root, category, output_tag)
        os.makedirs(sub, exist_ok=True)

        img_bgr = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取 overlay 背景图: {bg_path}")

        final_bin = select_best_connected_component(
            best["bin"],
            best["prob"],
            (cond.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8) * 255,
            cond,
            rel_hist_prior,
            uv_hist_prior,
            device,
            min_area=min_area,
            max_area=max_area,
            target_inside_cond=target_inside_cond,
        )
        best["bin"] = final_bin

        out_mask = os.path.join(sub, f"{stem}_mask.png")
        cv2.imwrite(out_mask, best["bin"])
        overlay_path = os.path.join(sub, f"{stem}_overlay.png")
        cv2.imwrite(overlay_path, blend_red_overlay(img_bgr, best["bin"], OVERLAY_ALPHA))
        print(f"[+] {category}/{defect} -> {out_mask}, overlay={overlay_path}, score={best.get('score', 0.0):.4f}, reason={best.get('reason', 'ok')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--infer_mask_root", type=str, default=DEFAULT_INFER_MASK_ROOT)
    p.add_argument(
        "--date_code",
        type=str,
        default=None,
        help="YYYYMMDD；需与训练一致，run_genmask_train_infer.py 会在启动时固定并传入",
    )
    p.add_argument(
        "--weights_dir",
        type=str,
        default=None,
        help="显式指定时覆盖 date_code 推导结果",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="显式指定时覆盖 date_code 推导结果",
    )
    p.add_argument(
        "--good_dir",
        type=str,
        default=None,
        help="直接指定 good 掩码目录，目录内放 png",
    )
    p.add_argument(
        "--normal_mask",
        type=str,
        default=None,
        help="单张 good 掩码路径",
    )
    p.add_argument(
        "--category",
        type=str,
        default=None,
        help="对应类别名；当使用 --good_dir 或 --normal_mask 时建议同时指定",
    )
    p.add_argument(
        "--run_all_good",
        action="store_true",
        help="遍历 dataset_root 下所有类别的 good 目录",
    )
    args = p.parse_args()

    date_code = resolve_date_code(args.date_code)
    weights_dir = args.weights_dir or weights_dir_for_date(args.output_root, date_code)
    out_dir = args.out_dir or infer_out_dir_for_date(args.infer_mask_root, date_code)
    print(f"[*] date_code={date_code}, weights_dir={weights_dir}, out_dir={out_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(out_dir, exist_ok=True)

    def weights_for_category(cat):
        paths = sorted(glob.glob(os.path.join(weights_dir, "*.pth")))
        matched = [p for p in paths if os.path.isfile(p) and match_weight_to_category(p, cat)]
        current = [p for p in matched if not os.path.basename(p).endswith(("_best.pth", "_final.pth"))]
        if current:
            return current
        legacy = [p for p in matched if os.path.basename(p).endswith("_best.pth")]
        if legacy:
            print(f"[!] {cat}: 未找到新格式权重，回退使用旧版 *_best.pth")
            return legacy
        legacy_final = [p for p in matched if os.path.basename(p).endswith("_final.pth")]
        if legacy_final:
            print(f"[!] {cat}: 未找到新格式权重，回退使用旧版 *_final.pth")
            return legacy_final
        return matched

    def infer_category_good(cat, mask_paths):
        paths = weights_for_category(cat)
        if not paths:
            print(f"[!] {cat}: 未找到权重")
            return
        for mp in mask_paths:
            stem = os.path.splitext(os.path.basename(mp))[0]
            cond = build_cond_tensor(mp, device)
            run_one_mask(paths, cond, cat, stem, mp, out_dir, device, args.dataset_root)

    if args.run_all_good:
        cats = list_seg_mask_categories(args.dataset_root)
        print(f"[*] 检测到类别数: {len(cats)} -> {cats}")
        for cat in cats:
            gdir = os.path.join(args.dataset_root, cat, "good")
            if not os.path.isdir(gdir):
                print(f"[!] {cat}: 无 good 目录 ({gdir})，跳过")
                continue
            masks = sorted(f for f in glob.glob(os.path.join(gdir, "*.png")) if os.path.isfile(f))
            if not masks:
                print(f"[!] {cat}: good 目录下无 png 图像，跳过")
                continue
            print(f"[*] {cat}: 找到 {len(masks)} 张原始图像")
            infer_category_good(cat, masks)
        return

    if args.good_dir:
        gdir = os.path.abspath(args.good_dir)
        cat = args.category
        if cat is None:
            parts = gdir.split(os.sep)
            if len(parts) >= 2 and parts[-1] == "good":
                cat = parts[-3] if parts[-2] == "test" else parts[-2]
            else:
                raise SystemExit("使用 --good_dir 时请提供 --category，或使路径以 .../类名/good 结尾")
        masks = sorted(f for f in glob.glob(os.path.join(gdir, "*.png")) if os.path.isfile(f))
        if not masks:
            raise SystemExit(f"good 目录无 png: {gdir}")
        infer_category_good(cat, masks)
        return

    if args.normal_mask:
        if not args.category:
            parts = os.path.abspath(args.normal_mask).split(os.sep)
            if len(parts) >= 2:
                args.category = parts[-3] if parts[-2] == "good" and len(parts) >= 3 else parts[-2]
        if not args.category:
            raise SystemExit("使用 --normal_mask 时必须指定 --category，或让路径包含类别信息")
        infer_category_good(args.category, [args.normal_mask])
        return

    raise SystemExit("请指定其一: --run_all_good | --good_dir | --normal_mask")


if __name__ == "__main__":
    main()