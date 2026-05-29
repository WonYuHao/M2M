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
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from code4maskgeneration.cvae import CondVAE
from code4maskgeneration.mvtec_mask_dataset import list_seg_mask_categories, list_training_defect_types


# 这些通常不需要走命令行参数，直接在代码里改即可
IMG_SIZE = 512
LATENT_DIM = 64
DROPOUT_P = 0.1
CANDIDATE_POOL = 5
MAX_RESAMPLE_ROUNDS = 20
TEMPERATURE = 1.0
THRESH = 0.5
MIN_AREA_RATIO = 0.3
MAX_AREA_RATIO = 1.0
OVERLAY_ALPHA = 0.35


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


def postprocess_binary(mask_u8, min_area_ratio=0.3, reject_border_components=True):
    if mask_u8.dtype != np.uint8:
        mask_u8 = mask_u8.astype(np.uint8)

    bin_m = (mask_u8 > 0).astype(np.uint8) * 255
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bin_m)
    if num_labels <= 1:
        return bin_m

    areas = stats[1:, cv2.CC_STAT_AREA]
    if len(areas) == 0:
        return np.zeros_like(bin_m)

    max_area = int(areas.max())
    keep_thr = max(1, int(max_area * float(min_area_ratio)))

    out = np.zeros_like(bin_m)
    h, w = bin_m.shape[:2]
    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < keep_thr:
            continue
        if reject_border_components:
            x, y, bw, bh, _ = stats[i]
            touches_border = x <= 0 or y <= 0 or (x + bw) >= w or (y + bh) >= h
            if touches_border:
                continue
        out[labels == i] = 255
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


def _score_candidate(prob, bin_pp, cond_u8, cond_area):
    bin_u8 = ((bin_pp > 0).astype(np.uint8))
    area = float(bin_u8.sum())
    if area <= 0:
        return -1e9
    stats = _mask_stats(bin_u8 * 255, ref_u8=cond_u8)
    cond_stats = _mask_stats(cond_u8)
    center_dist = float(np.sqrt((stats["centroid"][0] - cond_stats["centroid"][0]) ** 2 + (stats["centroid"][1] - cond_stats["centroid"][1]) ** 2))
    area_ratio = area / max(cond_area, 1.0)
    area_prior = np.exp(-abs(np.log(max(area_ratio, 1e-6))))
    overlap_bonus = stats["iou"] * 2.5 + stats["overlap_ratio"] * 1.5
    center_bonus = np.exp(-3.0 * center_dist)
    border_penalty = 0.35 if stats["touches_border"] else 0.0
    prob_bonus = float(prob[bin_u8 > 0].mean()) if (bin_u8 > 0).any() else float(prob.mean()) * 0.5
    return float(prob_bonus + overlap_bonus + center_bonus + 0.8 * area_prior - border_penalty)


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


def pick_best_mask(prob_list, thresh, cond, min_area=None, max_area=None):
    cond_u8 = (cond.squeeze().detach().cpu().numpy() > 0.5).astype(np.uint8) * 255
    cond_area = float((cond_u8 > 0).sum())
    best = None
    best_score = -1e9
    fallback = None
    fallback_score = -1e9
    for prob in prob_list:
        bin_m = ((prob > thresh).astype(np.uint8)) * 255
        bin_pp = postprocess_binary(bin_m, min_area_ratio=0.0, reject_border_components=True)
        area = int((bin_pp > 0).sum())
        valid = area > 0 and not _touches_image_border(bin_pp)
        if min_area is not None:
            valid = valid and area >= int(min_area)
        if max_area is not None:
            valid = valid and area <= int(max_area)
        if valid:
            score = _score_candidate(prob, bin_pp, cond_u8, cond_area)
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


def run_one_mask(model_paths, cond, category, stem, bg_path, out_root, device, dataset_root):
    for wp in model_paths:
        name = os.path.basename(wp)
        expected = f"{category}_"
        if not name.startswith(expected) or not name.endswith(".pth"):
            continue
        defect = name[len(expected) : -len(".pth")]
        if not defect:
            continue
        model = CondVAE(latent_dim=LATENT_DIM, cond_channels=1, dropout_p=DROPOUT_P).to(device)
        model.load_state_dict(torch.load(wp, map_location=device))

        area_range = collect_gt_area_range(dataset_root, category, defect, img_size=IMG_SIZE)
        if area_range is None:
            print(f"[!] {category}/{defect}: 未找到 gt 面积统计，跳过面积约束")
        else:
            print(f"[*] {category}/{defect}: gt area range = [{area_range['min']}, {area_range['max']}], count={area_range['count']}")

        min_area = area_range["min"] if area_range else None
        max_area = area_range["max"] if area_range else None

        best = None
        for round_idx in range(MAX_RESAMPLE_ROUNDS):
            probs = sample_priors(model, cond, LATENT_DIM, CANDIDATE_POOL, TEMPERATURE, device)
            candidate = pick_best_mask(probs, THRESH, cond, min_area=min_area, max_area=max_area)
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

        sub = os.path.join(out_root, category, defect)
        os.makedirs(sub, exist_ok=True)

        img_bgr = cv2.imread(bg_path, cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(f"无法读取 overlay 背景图: {bg_path}")

        out_mask = os.path.join(sub, f"{stem}_mask.png")
        cv2.imwrite(out_mask, best["bin"])
        overlay_path = os.path.join(sub, f"{stem}_overlay.png")
        cv2.imwrite(overlay_path, blend_red_overlay(img_bgr, best["bin"], OVERLAY_ALPHA))
        print(f"[+] {category}/{defect} -> {out_mask}, overlay={overlay_path}, score={best.get('score', 0.0):.4f}, reason={best.get('reason', 'ok')}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=str, default="/home/wyh/data/mvtec_ad/seg_mask")
    p.add_argument(
        "--weights_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "weights"),
        help="权重目录",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="./outputs/cvae_prior",
        help="输出目录",
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    def weights_for_category(cat):
        paths = sorted(glob.glob(os.path.join(args.weights_dir, "*.pth")))
        best_paths = [p for p in paths if os.path.isfile(p) and os.path.basename(p).startswith(f"{cat}_") and os.path.basename(p).endswith("_best.pth")]
        if best_paths:
            return best_paths
        final_paths = [p for p in paths if os.path.isfile(p) and os.path.basename(p).startswith(f"{cat}_") and os.path.basename(p).endswith("_final.pth")]
        if final_paths:
            print(f"[!] {cat}: 未找到 *_best.pth，回退使用 *_final.pth")
            return final_paths
        return [p for p in paths if os.path.isfile(p) and match_weight_to_category(p, cat)]

    def infer_category_good(cat, mask_paths):
        paths = weights_for_category(cat)
        if not paths:
            print(f"[!] {cat}: 未找到权重")
            return
        for mp in mask_paths:
            stem = os.path.splitext(os.path.basename(mp))[0]
            cond = build_cond_tensor(mp, device)
            run_one_mask(paths, cond, cat, stem, mp, args.out_dir, device, args.dataset_root)

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