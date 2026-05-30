"""
训练 CVAE mask generation。

训练目标：推理时从随机 latent 采样，得到面积合理的掩码；
与 cond 的重叠程度对齐训练数据中 GT 的统计（并非所有异常都完全落在 cond 内）。

策略：
- 全量数据训练 + 生成式指标选 checkpoint
- 双路径损失：posterior 重建 + prior 随机 z 解码
- overlap：pred 在 cond 内占比 ≈ GT 在 cond 内占比
- rel_hist：GT 质量在 cond 形态空间 (深度 DT, 局部宽度) 的分布
- band：recon 路径在 cond 邻域带内与 GT 的软重叠（同图坐标）
"""

import argparse
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from code4maskgeneration.cvae import CondVAE
from code4maskgeneration.mvtec_mask_dataset import (
    Mask2MaskDataset,
    list_seg_mask_categories,
    list_training_defect_types,
    _is_training_defect,
)

DEFAULT_DATASET_ROOT = "/home/wyh/data/mvtec_ad/seg_mask"
DEFAULT_OUTPUT_ROOT = "/d242/wyh/M2M"
WEIGHTS_SUBDIR = "genmask_cvae"
IMG_SIZE = 512
LATENT_DIM = 64
IMG_AREA = float(IMG_SIZE * IMG_SIZE)

LEARNING_RATE = 1e-4
BATCH_SIZE = 8
NUM_WORKERS = 0
PATIENCE = 20
MIN_DELTA = 1e-4

KL_WEIGHT = 0.01
KL_ANNEAL_EPOCHS = 25
BCE_WEIGHT = 0.1
DICE_WEIGHT = 0.15
COND_OVERLAP_WEIGHT = 2.0
OUTSIDE_COND_WEIGHT = 2.0
AREA_WEIGHT = 1.0
EMPTY_WEIGHT = 2.0
REL_HIST_WEIGHT = 3.0
BAND_OVERLAP_WEIGHT = 1.5
BAND_RADIUS = 15
REL_HIST_BINS_DT = 10
REL_HIST_BINS_WIDTH = 10
LOCAL_WIDTH_KERNEL = 25
PRIOR_WEIGHT = 1.75
PRIOR_SAMPLES = 2
GEN_EVAL_Z = 3


def _dice_loss(recon_x: torch.Tensor, x: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred = recon_x.view(recon_x.size(0), -1)
    target = x.view(x.size(0), -1)
    inter = (pred * target).sum(dim=1)
    denom = pred.sum(dim=1) + target.sum(dim=1)
    dice = (2.0 * inter + smooth) / (denom + smooth)
    return (1.0 - dice).mean()


def _foreground_bce(recon_x: torch.Tensor, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    valid = ((cond > 0.5) | (x > 0.5)).float()
    per_pixel = F.binary_cross_entropy(recon_x, x, reduction="none")
    return (per_pixel * valid).sum() / valid.sum().clamp_min(1.0)


def _inside_cond_ratio(tensor: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """前景质量落在 cond 内的占比，shape [B, 1]。"""
    mass = tensor.clamp(0.0, 1.0).sum(dim=(2, 3)).clamp_min(1e-6)
    cond_bin = (cond > 0.5).float()
    inside = (tensor.clamp(0.0, 1.0) * cond_bin).sum(dim=(2, 3))
    return inside / mass


def _gt_inside_cond_ratio(gt: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    """当前样本 GT 异常落在 cond 内的质量占比，shape [B, 1]。"""
    gt_bin = (gt > 0.5).float()
    gt_mass = gt_bin.sum(dim=(2, 3)).clamp_min(1e-6)
    cond_bin = (cond > 0.5).float()
    inside = (gt_bin * cond_bin).sum(dim=(2, 3))
    return inside / gt_mass


def _cond_morphology_maps(cond_u8: np.ndarray):
    """cond 内蕴形态：DT 深度 + 局部宽度（邻域 DT 极大值，区分平端宽/尖端窄）。"""
    c = (cond_u8 > 0.5).astype(np.uint8)
    h, w = c.shape
    if c.max() == 0:
        z = np.zeros((h, w), dtype=np.float32)
        return z, z

    dt = cv2.distanceTransform(c, cv2.DIST_L2, 5).astype(np.float32)
    dt_max = float(dt.max())
    dt_norm = dt / dt_max if dt_max > 0 else dt

    k = int(LOCAL_WIDTH_KERNEL)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    local_w = cv2.dilate(dt, kernel).astype(np.float32)
    lw_max = float(local_w.max())
    if lw_max > 0:
        local_w = local_w / lw_max

    fg = c.astype(np.float32)
    return dt_norm * fg, local_w * fg


def _cond_morphology_batch(cond: torch.Tensor):
    """逐样本从 cond mask 计算形态图 [B,1,H,W]。"""
    device = cond.device
    b, _, h, w = cond.shape
    dt_list, width_list = [], []
    for i in range(b):
        c = (cond[i, 0].detach().cpu().numpy() > 0.5).astype(np.uint8)
        dt, width = _cond_morphology_maps(c)
        dt_list.append(torch.from_numpy(dt))
        width_list.append(torch.from_numpy(width))
    dt_t = torch.stack(dt_list, dim=0).unsqueeze(1).to(device)
    width_t = torch.stack(width_list, dim=0).unsqueeze(1).to(device)
    return dt_t, width_t


def _rel_histogram(
    mask: torch.Tensor,
    dt: torch.Tensor,
    width: torch.Tensor,
    bins_dt: int = REL_HIST_BINS_DT,
    bins_width: int = REL_HIST_BINS_WIDTH,
) -> torch.Tensor:
    """mask 质量在 (cond深度, cond局部宽度) 上的分布。"""
    m = mask.clamp(0.0, 1.0)
    b = m.size(0)
    out = []
    n_bins = bins_dt * bins_width
    for i in range(b):
        mi = m[i, 0]
        dti = dt[i, 0]
        wi = width[i, 0]
        active = mi > 0.05
        if int(active.sum()) < 1:
            out.append(torch.zeros(n_bins, device=m.device))
            continue
        d_vals = (dti[active] * float(bins_dt - 1e-3)).long().clamp(0, bins_dt - 1)
        w_vals = (wi[active] * float(bins_width - 1e-3)).long().clamp(0, bins_width - 1)
        weights = mi[active]
        hist = torch.zeros(n_bins, device=m.device)
        idx = d_vals * bins_width + w_vals
        hist.index_add_(0, idx, weights)
        hist = hist / hist.sum().clamp_min(1e-6)
        out.append(hist)
    return torch.stack(out, dim=0)


def compute_dataset_rel_hist_prior(dataset):
    """训练集 GT 在 cond 形态空间的平均直方图（prior / 推理选模）。"""
    hists = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for cond, gt in loader:
        dt, width = _cond_morphology_batch(cond)
        hists.append(_rel_histogram(gt, dt, width)[0].cpu().numpy())
    if not hists:
        n = REL_HIST_BINS_DT * REL_HIST_BINS_WIDTH
        return np.ones(n, dtype=np.float32) / n
    mean_hist = np.mean(np.stack(hists, axis=0), axis=0).astype(np.float32)
    mean_hist = mean_hist / max(float(mean_hist.sum()), 1e-6)
    return mean_hist


def _rel_hist_loss(pred_hist: torch.Tensor, target_hist: torch.Tensor) -> torch.Tensor:
    target = target_hist.to(pred_hist.device)
    if target.dim() == 1:
        target = target.unsqueeze(0).expand_as(pred_hist)
    return F.l1_loss(pred_hist, target)


def _cond_neighbor_band(cond: torch.Tensor, radius: int = BAND_RADIUS) -> torch.Tensor:
    c = (cond > 0.5).float()
    k = 2 * int(radius) + 1
    dilated = F.max_pool2d(c, kernel_size=k, stride=1, padding=int(radius))
    return (dilated > 0.5).float()


def _band_overlap_loss(pred: torch.Tensor, gt: torch.Tensor, band: torch.Tensor) -> torch.Tensor:
    """同图坐标下，cond 邻域带内 pred 与 GT 的软重叠（仅 recon 有 GT 时使用）。"""
    a = pred.clamp(0.0, 1.0) * band
    b = gt.clamp(0.0, 1.0) * band
    inter = (a * b).sum(dim=(2, 3))
    union = a.sum(dim=(2, 3)) + b.sum(dim=(2, 3))
    dice = (2.0 * inter + 1.0) / (union + 1.0)
    return (1.0 - dice).mean()


def compute_dataset_cond_stats(dataset):
    """统计训练集 GT 落在 cond 内的质量占比（每类缺陷一种分布）。"""
    ratios = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for cond, gt in loader:
        r = float(_gt_inside_cond_ratio(gt, cond).item())
        ratios.append(r)
    if not ratios:
        return 1.0, 1.0, 1.0
    arr = np.array(ratios, dtype=np.float64)
    return float(arr.min()), float(arr.max()), float(arr.mean())


def _overlap_to_target_loss(pred_inside: torch.Tensor, target_inside: torch.Tensor) -> torch.Tensor:
    """pred 在 cond 内占比接近目标占比（来自 GT 或数据集均值）。"""
    target = target_inside.view_as(pred_inside).to(pred_inside.device)
    return F.l1_loss(pred_inside, target)


def _excess_outside_cond_loss(pred_inside: torch.Tensor, target_inside: torch.Tensor) -> torch.Tensor:
    """仅当 pred 落在 cond 外的比例超过 GT 允许值时惩罚。"""
    target = target_inside.view_as(pred_inside).to(pred_inside.device)
    return F.relu(target - pred_inside).mean()


def _area_loss(recon_x: torch.Tensor, area_min: float, area_max: float) -> torch.Tensor:
    """仅约束面积落在数据集统计范围内（归一化到 [0,1] 量级）。"""
    pred_area = recon_x.sum(dim=(2, 3))
    low_pen = F.relu(float(area_min) - pred_area) / IMG_AREA
    high_pen = F.relu(pred_area - float(area_max)) / IMG_AREA
    return 0.5 * (low_pen.mean() + high_pen.mean())


def _empty_penalty(recon_x: torch.Tensor, area_min: float) -> torch.Tensor:
    pred_area = recon_x.sum(dim=(2, 3))
    return F.relu(float(area_min) * 0.25 - pred_area).mean() / IMG_AREA


def compute_dataset_area_bounds(dataset):
    areas = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    for _, gt in loader:
        areas.append(float(gt.sum().item()))
    if not areas:
        return 1.0, float(IMG_SIZE * IMG_SIZE), 0.0
    arr = np.array(areas, dtype=np.float64)
    return float(arr.min()), float(arr.max()), float(arr.mean())


def _base_spatial_losses(recon_x, cond, area_min, area_max, target_inside_cond):
    pred_inside = _inside_cond_ratio(recon_x, cond)
    return {
        "overlap": _overlap_to_target_loss(pred_inside, target_inside_cond),
        "outside": _excess_outside_cond_loss(pred_inside, target_inside_cond),
        "area": _area_loss(recon_x, area_min, area_max),
    }


def _add_spatial_weights(spatial, total, logs, rel_hist=None, band=None):
    total = (
        total
        + COND_OVERLAP_WEIGHT * spatial["overlap"]
        + OUTSIDE_COND_WEIGHT * spatial["outside"]
        + AREA_WEIGHT * spatial["area"]
    )
    logs["overlap"] = spatial["overlap"].detach()
    logs["outside"] = spatial["outside"].detach()
    logs["area"] = spatial["area"].detach()
    if rel_hist is not None:
        total = total + REL_HIST_WEIGHT * rel_hist
        logs["rel_hist"] = rel_hist.detach()
    if band is not None:
        total = total + BAND_OVERLAP_WEIGHT * band
        logs["band"] = band.detach()
    return total, logs


def loss_recon_path(recon_x, x, mu, logvar, cond, area_min, area_max, kl_weight):
    target_inside = _gt_inside_cond_ratio(x, cond)
    spatial = _base_spatial_losses(recon_x, cond, area_min, area_max, target_inside)
    dt, width = _cond_morphology_batch(cond)
    pred_hist = _rel_histogram(recon_x, dt, width)
    gt_hist = _rel_histogram(x, dt, width)
    rel_hist = _rel_hist_loss(pred_hist, gt_hist)
    band = _band_overlap_loss(recon_x, x, _cond_neighbor_band(cond))
    bce = _foreground_bce(recon_x, x, cond)
    dice = _dice_loss(recon_x, x)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

    total = BCE_WEIGHT * bce + DICE_WEIGHT * dice + kl_weight * kld
    logs = {"bce": bce.detach(), "dice": dice.detach(), "kld": kld.detach()}
    total, logs = _add_spatial_weights(spatial, total, logs, rel_hist=rel_hist, band=band)
    return total, logs


def loss_prior_path(recon_x, cond, area_min, area_max, target_inside_mean, rel_hist_prior):
    target_inside = torch.full(
        (recon_x.size(0), 1, 1, 1),
        float(target_inside_mean),
        device=recon_x.device,
        dtype=recon_x.dtype,
    )
    spatial = _base_spatial_losses(recon_x, cond, area_min, area_max, target_inside)
    dt, width = _cond_morphology_batch(cond)
    pred_hist = _rel_histogram(recon_x, dt, width)
    prior_t = torch.from_numpy(rel_hist_prior).to(recon_x.device)
    rel_hist = _rel_hist_loss(pred_hist, prior_t)
    empty_pen = _empty_penalty(recon_x, area_min)

    total = EMPTY_WEIGHT * empty_pen
    logs = {"empty": empty_pen.detach()}
    total, logs = _add_spatial_weights(spatial, total, logs, rel_hist=rel_hist)
    return total, logs


def _kl_weight_for_epoch(epoch):
    if KL_ANNEAL_EPOCHS <= 0:
        return KL_WEIGHT
    progress = min(1.0, float(epoch + 1) / float(KL_ANNEAL_EPOCHS))
    return KL_WEIGHT * progress


def _run_epoch(
    model, loader, device, area_bounds, cond_bounds, rel_hist_prior, optimizer=None, epoch=0
):
    training = optimizer is not None
    model.train(training)
    kl_weight = _kl_weight_for_epoch(epoch) if training else KL_WEIGHT
    area_min, area_max, _ = area_bounds
    _, _, inside_mean = cond_bounds

    keys = [
        "loss", "recon", "prior", "bce", "dice", "kld",
        "overlap", "outside", "area", "rel_hist", "band", "empty",
    ]
    running = {k: 0.0 for k in keys}

    with torch.set_grad_enabled(training):
        for cond_mask, gt_mask in loader:
            cond_mask = cond_mask.to(device)
            gt_mask = gt_mask.to(device)
            if training:
                optimizer.zero_grad()

            recon, mu, logvar = model(gt_mask, cond_mask)
            recon_loss, recon_logs = loss_recon_path(
                recon, gt_mask, mu, logvar, cond_mask, area_min, area_max, kl_weight
            )

            prior_losses = []
            prior_logs = {
                "overlap": 0.0,
                "outside": 0.0,
                "area": 0.0,
                "rel_hist": 0.0,
                "empty": 0.0,
            }
            for _ in range(PRIOR_SAMPLES):
                z = torch.randn(cond_mask.size(0), LATENT_DIM, device=device)
                prior_recon = model.decode(z, cond_mask)
                p_loss, p_logs = loss_prior_path(
                    prior_recon,
                    cond_mask,
                    area_min,
                    area_max,
                    inside_mean,
                    rel_hist_prior,
                )
                prior_losses.append(p_loss)
                for k in prior_logs:
                    prior_logs[k] += p_logs[k].item()
            prior_loss = torch.stack(prior_losses).mean()
            for k in prior_logs:
                prior_logs[k] /= max(PRIOR_SAMPLES, 1)

            loss = recon_loss + PRIOR_WEIGHT * prior_loss
            if training:
                loss.backward()
                optimizer.step()

            running["loss"] += loss.item()
            running["recon"] += recon_loss.item()
            running["prior"] += prior_loss.item()
            running["bce"] += recon_logs["bce"].item()
            running["dice"] += recon_logs["dice"].item()
            running["kld"] += recon_logs["kld"].item()
            running["overlap"] += recon_logs["overlap"].item() + prior_logs["overlap"]
            running["outside"] += recon_logs["outside"].item() + prior_logs["outside"]
            running["area"] += recon_logs["area"].item() + prior_logs["area"]
            running["rel_hist"] += recon_logs["rel_hist"].item() + prior_logs["rel_hist"]
            running["band"] += recon_logs["band"].item()
            running["empty"] += prior_logs["empty"]

    denom = max(len(loader), 1)
    return {k: running[k] / denom for k in keys}


@torch.no_grad()
def _evaluate_generative(
    model, dataset, device, area_bounds, cond_bounds, rel_hist_prior, num_z=GEN_EVAL_Z
):
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    area_min, area_max, _ = area_bounds
    _, _, inside_mean = cond_bounds
    target_inside = torch.tensor([[inside_mean]], device=device)
    prior_t = torch.from_numpy(rel_hist_prior)

    total_score = 0.0
    total_overlap = total_outside = total_rel_hist = total_area_pen = total_empty = 0.0
    n = 0

    for cond_mask, _gt_mask in loader:
        cond_mask = cond_mask.to(device)
        sample_score = 0.0
        dt, width = _cond_morphology_batch(cond_mask)

        for _ in range(num_z):
            z = torch.randn(1, LATENT_DIM, device=device)
            recon = model.decode(z, cond_mask)
            pred_inside = _inside_cond_ratio(recon, cond_mask)
            overlap = _overlap_to_target_loss(pred_inside, target_inside).item()
            outside = _excess_outside_cond_loss(pred_inside, target_inside).item()
            pred_hist = _rel_histogram(recon, dt, width)
            rel_hist = _rel_hist_loss(pred_hist, prior_t).item()
            pred_area = float(recon.sum().item())
            area_pen = (max(0.0, area_min - pred_area) + max(0.0, pred_area - area_max)) / IMG_AREA
            empty = 1.0 if pred_area < area_min * 0.2 else 0.0
            sample_score += overlap + outside + rel_hist + area_pen + 5.0 * empty

            total_overlap += overlap
            total_outside += outside
            total_rel_hist += rel_hist
            total_area_pen += area_pen
            total_empty += empty

        total_score += sample_score / float(num_z)
        n += 1

    if n == 0:
        return {
            "score": float("inf"),
            "overlap": 1.0,
            "outside": 1.0,
            "rel_hist": 1.0,
            "area_pen": 1.0,
            "empty_rate": 1.0,
        }

    z_total = n * num_z
    return {
        "score": total_score / n,
        "overlap": total_overlap / z_total,
        "outside": total_outside / z_total,
        "rel_hist": total_rel_hist / z_total,
        "area_pen": total_area_pen / z_total,
        "empty_rate": total_empty / z_total,
    }


def train_model(dataset, category, defect, weights_dir, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    area_bounds = compute_dataset_area_bounds(dataset)
    cond_bounds = compute_dataset_cond_stats(dataset)
    rel_hist_prior = compute_dataset_rel_hist_prior(dataset)
    area_min, area_max, area_mean = area_bounds
    inside_min, inside_max, inside_mean = cond_bounds
    print(
        f"[*] [{category}/{defect}] GT 面积统计: min={area_min:.0f}, max={area_max:.0f}, mean={area_mean:.0f}"
    )
    print(
        f"[*] [{category}/{defect}] GT 在 cond 内质量占比: min={inside_min:.3f}, "
        f"max={inside_max:.3f}, mean={inside_mean:.3f}"
    )
    print(
        f"[*] [{category}/{defect}] cond 形态直方图 (DT×宽度): bins={REL_HIST_BINS_DT}x{REL_HIST_BINS_WIDTH}, "
        f"peak={rel_hist_prior.max():.4f}"
    )

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    model = CondVAE(latent_dim=LATENT_DIM, cond_channels=1, dropout_p=0.1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    ckpt_path = weight_checkpoint_path(weights_dir, category, defect)
    best_metric = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        train_logs = _run_epoch(
            model,
            loader,
            device,
            area_bounds,
            cond_bounds,
            rel_hist_prior,
            optimizer=optimizer,
            epoch=epoch,
        )
        gen_logs = _evaluate_generative(
            model, dataset, device, area_bounds, cond_bounds, rel_hist_prior
        )

        current_metric = gen_logs["score"]
        if current_metric < (best_metric - MIN_DELTA):
            best_metric = current_metric
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1

        print(
            f"[{category}/{defect}] Epoch [{epoch + 1}/{epochs}] "
            f"train_loss: {train_logs['loss']:.4f} recon: {train_logs['recon']:.4f} prior: {train_logs['prior']:.4f} "
            f"dice: {train_logs['dice']:.4f} overlap: {train_logs['overlap']:.4f} outside: {train_logs['outside']:.4f} "
            f"rel_hist: {train_logs['rel_hist']:.4f} band: {train_logs['band']:.4f} "
            f"area: {train_logs['area']:.4f} empty: {train_logs['empty']:.4f} | "
            f"gen_score: {gen_logs['score']:.4f} gen_rel_hist: {gen_logs['rel_hist']:.4f} "
            f"gen_empty: {gen_logs['empty_rate']:.3f}"
        )

        if patience_counter >= PATIENCE:
            print(f"[!] Early stopping at epoch {epoch + 1}, best_gen_score={best_metric:.4f}")
            break

    print(f"[+] 保存权重: {ckpt_path} (best_gen_score={best_metric:.4f})")


def weights_dir_for_date(output_root: str, date_code: str) -> str:
    return str(Path(output_root) / WEIGHTS_SUBDIR / f"weights_{date_code}")


def resolve_date_code(date_code: str | None) -> str:
    """脚本内只调用一次，避免 parse 默认值与后续逻辑各取各的日期。"""
    return date_code or datetime.now().strftime("%Y%m%d")


def weight_checkpoint_path(weights_dir: str, category: str, defect: str) -> str:
    return str(Path(weights_dir) / f"{category}_{defect}.pth")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET_ROOT)
    p.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument(
        "--date_code",
        type=str,
        default=None,
        help="YYYYMMDD；与 infer 保持一致，run_genmask_train_infer.py 会在启动时固定并传入",
    )
    p.add_argument(
        "--weights_dir",
        type=str,
        default=None,
        help="显式指定时覆盖 date_code 推导结果",
    )
    p.add_argument("--category", type=str, default=None)
    p.add_argument("--defect_type", type=str, default=None)
    p.add_argument("--epochs", type=int, default=100)
    args = p.parse_args()

    date_code = resolve_date_code(args.date_code)
    weights_dir = args.weights_dir or weights_dir_for_date(args.output_root, date_code)
    print(f"[*] date_code={date_code}, weights_dir={weights_dir}")

    os.makedirs(weights_dir, exist_ok=True)
    categories = [args.category] if args.category else list_seg_mask_categories(args.dataset_root)
    if not categories:
        raise SystemExit(f"未找到类别目录: {args.dataset_root}")

    for category in categories:
        gt_dir = os.path.join(args.dataset_root, category, "ground_truth")
        if not os.path.isdir(gt_dir):
            print(f"[!] 跳过 {category}：无 ground_truth ({gt_dir})")
            continue

        if args.defect_type is not None:
            if not _is_training_defect(args.defect_type):
                print(f"[!] 跳过 {category}：defect_type={args.defect_type} 为 good，不参与训练")
                continue
            defect_types = [args.defect_type]
        else:
            defect_types = list_training_defect_types(gt_dir)

        for defect in defect_types:
            dataset = Mask2MaskDataset(root_dir=args.dataset_root, category=category, defect_type=defect, img_size=IMG_SIZE)
            num_samples = len(dataset)
            if num_samples == 0:
                print(f"[!] 跳过 {category}/{defect}：无样本")
                continue
            print(f"[*] [{category}/{defect}] 样本数: {num_samples}，全量训练 + prior 路径 + 生成式选模")
            train_model(dataset, category, defect, weights_dir, args.epochs)


if __name__ == "__main__":
    main()
