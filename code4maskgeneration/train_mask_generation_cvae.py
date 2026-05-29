"""
训练 CVAE mask generation。

训练策略：
- 小样本场景下先用 leave-one-out 做轮换验证，观察模型在不同样本上的稳定性；
- 再用全量数据重新训练一次最终模型；
- 最终只保存一个 best checkpoint 和一个 final checkpoint。

这样可以避免固定验证集带来的分布偏置，同时不需要为每个 fold 单独保存模型。
"""

import argparse
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from code4maskgeneration.cvae import CondVAE
from code4maskgeneration.mvtec_mask_dataset import (
    Mask2MaskDataset,
    list_seg_mask_categories,
    list_training_defect_types,
    _is_training_defect,
)


IMG_SIZE = 512
LEARNING_RATE = 1e-4
KL_WEIGHT = 0.001
DT_WEIGHT = 1.0
BATCH_SIZE = 8
NUM_WORKERS = 0
PATIENCE = 15
MIN_DELTA = 1e-4


def _distance_transform_batch(cond: torch.Tensor) -> torch.Tensor:
    cond_np = cond.detach().float().cpu().numpy()
    dists = []
    for i in range(cond_np.shape[0]):
        fg_u8 = (cond_np[i, 0] > 0.5).astype(np.uint8)
        if fg_u8.max() == 0:
            dists.append(np.zeros_like(fg_u8, dtype=np.float32))
            continue
        dist = cv2.distanceTransform(fg_u8, cv2.DIST_L2, 5)
        maxv = float(dist.max())
        if maxv > 0:
            dist = dist / maxv
        dists.append(dist.astype(np.float32))
    return torch.from_numpy(np.stack(dists, axis=0)).to(cond.device)


def _spatial_relation_stats(cond: torch.Tensor, x: torch.Tensor) -> dict:
    cond_bin = (cond > 0.5).float()
    x_bin = (x > 0.5).float()

    inter = (cond_bin * x_bin).sum(dim=(2, 3), keepdim=True)
    cond_area = cond_bin.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    x_area = x_bin.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    union = (cond_bin + x_bin - cond_bin * x_bin).sum(dim=(2, 3), keepdim=True).clamp_min(1.0)

    coords_y = torch.linspace(0.0, 1.0, cond_bin.shape[2], device=cond.device).view(1, 1, -1, 1)
    coords_x = torch.linspace(0.0, 1.0, cond_bin.shape[3], device=cond.device).view(1, 1, 1, -1)
    cond_mass = cond_bin.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    x_mass = x_bin.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    cond_cy = (cond_bin * coords_y).sum(dim=(2, 3), keepdim=True) / cond_mass
    cond_cx = (cond_bin * coords_x).sum(dim=(2, 3), keepdim=True) / cond_mass
    x_cy = (x_bin * coords_y).sum(dim=(2, 3), keepdim=True) / x_mass
    x_cx = (x_bin * coords_x).sum(dim=(2, 3), keepdim=True) / x_mass
    centroid_dist = torch.sqrt((cond_cy - x_cy).pow(2) + (cond_cx - x_cx).pow(2)).clamp_min(0.0)

    return {
        "iou": (inter / union).squeeze(-1).squeeze(-1),
        "centroid_dist": centroid_dist.squeeze(-1).squeeze(-1),
    }


def _relation_weights(cond: torch.Tensor, x: torch.Tensor) -> dict:
    stats = _spatial_relation_stats(cond, x)
    iou = stats["iou"]
    centroid_dist = stats["centroid_dist"]
    overlap_conf = torch.clamp((iou - 0.08) / 0.35, 0.0, 1.0)
    separation_conf = torch.clamp((centroid_dist - 0.18) / 0.35, 0.0, 1.0)
    aligned_conf = torch.clamp(1.0 - separation_conf, 0.0, 1.0)

    bce_scale = (1.0 + 1.5 * overlap_conf + 0.5 * (1.0 - separation_conf)).detach()
    dt_scale = (0.2 + 1.8 * overlap_conf * aligned_conf).detach()
    kl_scale = (0.5 + 0.5 * (1.0 - overlap_conf)).detach()

    weight_map = (x * (8.0 + 16.0 * overlap_conf.view(-1, 1, 1, 1)) + 1.0)
    weight_map = weight_map * (cond * (1.0 + 2.0 * aligned_conf.view(-1, 1, 1, 1)) + 1.0)

    return {
        "stats": stats,
        "bce_scale": bce_scale,
        "dt_scale": dt_scale,
        "kl_scale": kl_scale,
        "weight_map": weight_map,
    }


def _position_alignment_loss(recon_x: torch.Tensor, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
    dt = _distance_transform_batch(cond).unsqueeze(1)
    gt_mass = x.sum(dim=(2, 3), keepdim=True).clamp_min(1.0)
    pred_mass = recon_x.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    gt_expect = (x * dt).sum(dim=(2, 3), keepdim=True) / gt_mass
    pred_expect = (recon_x * dt).sum(dim=(2, 3), keepdim=True) / pred_mass
    return F.l1_loss(pred_expect, gt_expect)


def loss_function(recon_x, x, mu, logvar, cond, kl_weight=0.001, dt_weight=1.0):
    relation = _relation_weights(cond, x)
    bce = F.binary_cross_entropy(recon_x, x, weight=relation["weight_map"], reduction="mean")
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    dt_loss = _position_alignment_loss(recon_x, x, cond)

    total = relation["bce_scale"].mean() * bce + kl_weight * relation["kl_scale"].mean() * kld + dt_weight * relation["dt_scale"].mean() * dt_loss
    return total, {
        "bce": bce.detach(),
        "kld": kld.detach(),
        "dt": dt_loss.detach(),
        "iou": relation["stats"]["iou"].mean().detach(),
        "centroid_dist": relation["stats"]["centroid_dist"].mean().detach(),
    }


def build_leave_one_out(num_samples):
    if num_samples <= 0:
        return []
    if num_samples == 1:
        return [([0], [0])]
    return [([idx for idx in range(num_samples) if idx != val_idx], [val_idx]) for val_idx in range(num_samples)]


def _run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)

    running = running_bce = running_kld = running_dt = running_iou = running_centroid = 0.0
    with torch.set_grad_enabled(training):
        for cond_mask, gt_mask in loader:
            cond_mask = cond_mask.to(device)
            gt_mask = gt_mask.to(device)
            if training:
                optimizer.zero_grad()
            recon, mu, logvar = model(gt_mask, cond_mask)
            loss, logs = loss_function(recon, gt_mask, mu, logvar, cond_mask, kl_weight=KL_WEIGHT, dt_weight=DT_WEIGHT)
            if training:
                loss.backward()
                optimizer.step()

            running += loss.item()
            running_bce += logs["bce"].item()
            running_kld += logs["kld"].item()
            running_dt += logs["dt"].item()
            running_iou += logs["iou"].item()
            running_centroid += logs["centroid_dist"].item()

    denom = max(len(loader), 1)
    return {
        "loss": running / denom,
        "bce": running_bce / denom,
        "kld": running_kld / denom,
        "dt": running_dt / denom,
        "iou": running_iou / denom,
        "centroid": running_centroid / denom,
    }


def train_with_loo_then_full(dataset, category, defect, weights_dir, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    folds = build_leave_one_out(len(dataset))
    if not folds:
        print(f"[!] 跳过 {category}/{defect}：无样本")
        return

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=1, shuffle=False, num_workers=NUM_WORKERS)

        model = CondVAE(latent_dim=64, cond_channels=1, dropout_p=0.1).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        best_val = float("inf")
        patience_counter = 0
        for epoch in range(epochs):
            train_logs = _run_epoch(model, train_loader, device, optimizer=optimizer)
            val_logs = _run_epoch(model, val_loader, device)

            if val_logs["loss"] < (best_val - MIN_DELTA):
                best_val = val_logs["loss"]
                patience_counter = 0
            else:
                patience_counter += 1

            print(
                f"[{category}/{defect}] LOO Fold [{fold_idx + 1}/{len(folds)}] Epoch [{epoch + 1}/{epochs}] "
                f"train_loss: {train_logs['loss']:.4f} val_loss: {val_logs['loss']:.4f} "
                f"train_dt: {train_logs['dt']:.4f} val_dt: {val_logs['dt']:.4f} "
                f"train_iou: {train_logs['iou']:.4f} val_iou: {val_logs['iou']:.4f} "
                f"train_centroid: {train_logs['centroid']:.4f} val_centroid: {val_logs['centroid']:.4f}"
            )

            if patience_counter >= PATIENCE:
                break

    full_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    final_model = CondVAE(latent_dim=64, cond_channels=1, dropout_p=0.1).to(device)
    final_optimizer = optim.Adam(final_model.parameters(), lr=LEARNING_RATE)

    best_path = os.path.join(weights_dir, f"{category}_{defect}_best.pth")
    final_path = os.path.join(weights_dir, f"{category}_{defect}_final.pth")
    best_metric = float("inf")
    patience_counter = 0

    for epoch in range(epochs):
        train_logs = _run_epoch(final_model, full_loader, device, optimizer=final_optimizer)
        current_metric = train_logs["loss"] + 0.2 * train_logs["centroid"] + 0.2 * train_logs["dt"]

        if current_metric < (best_metric - MIN_DELTA):
            best_metric = current_metric
            patience_counter = 0
            torch.save(final_model.state_dict(), best_path)
        else:
            patience_counter += 1

        print(
            f"[{category}/{defect}] Full Train Epoch [{epoch + 1}/{epochs}] "
            f"loss: {train_logs['loss']:.4f} bce: {train_logs['bce']:.4f} kld: {train_logs['kld']:.4f} "
            f"dt: {train_logs['dt']:.4f} iou: {train_logs['iou']:.4f} centroid: {train_logs['centroid']:.4f}"
        )

        if patience_counter >= PATIENCE:
            print(f"[!] Full training early stopping at epoch {epoch + 1}, best_metric={best_metric:.4f}")
            break

    torch.save(final_model.state_dict(), final_path)
    print(f"[+] 保存最佳权重: {best_path} (best_metric={best_metric:.4f})")
    print(f"[+] 保存最终权重: {final_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset_root", type=str, default="/home/wyh/data/mvtec_ad/seg_mask")
    p.add_argument("--weights_dir", type=str, default=os.path.join(os.path.dirname(__file__), "weights"))
    p.add_argument("--category", type=str, default=None)
    p.add_argument("--defect_type", type=str, default=None)
    p.add_argument("--epochs", type=int, default=100)
    args = p.parse_args()

    os.makedirs(args.weights_dir, exist_ok=True)
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
            print(f"[*] [{category}/{defect}] 样本数: {num_samples}，采用 leave-one-out + full retrain")
            train_with_loo_then_full(dataset, category, defect, args.weights_dir, args.epochs)


if __name__ == "__main__":
    main()
