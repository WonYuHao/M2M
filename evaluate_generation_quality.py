import argparse
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import linalg
from torchvision import models, transforms

import lpips
from tqdm.auto import tqdm

IMG_EXTS = {".png"}


@dataclass
class ObjectMetrics:
    object_name: str
    num_images: int
    is_mean: float
    is_std: float
    icl_mean: float
    defects: Dict[str, int]


@dataclass
class CellFIDMetrics:
    category: str
    defect_type: str
    num_gen_images: int
    num_real_images: int
    fid: float

# ==========================================
# 1. 算法复用：图像加载逻辑 (保持像素精度)
# ==========================================
def load_image_tensor_for_inception(img_path: Path, image_size: int = 299) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert("RGB")
    return tf(img)

def load_image_tensor_for_lpips(img_path: Path, image_size: int = 256) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert("RGB")
    return (tf(img) * 2.0 - 1.0).unsqueeze(0)


class InceptionFeatureExtractor(torch.nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        self.model = models.inception_v3(weights=weights, transform_input=False, aux_logits=True)
        self.model.AuxLogits = None
        self.model.fc = torch.nn.Identity()
        self.model.eval().to(device)
        self.device = device

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def load_image_tensor(img_path: Path, image_size: int = 299) -> torch.Tensor:
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    img = Image.open(img_path).convert("RGB")
    return tf(img)


def compute_activations(image_paths: List[Path], device: torch.device, batch_size: int = 32, desc: str = "computing activations") -> np.ndarray:
    if len(image_paths) == 0:
        return np.empty((0, 2048), dtype=np.float64)

    model = InceptionFeatureExtractor(device)
    acts: List[np.ndarray] = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc=desc, leave=False):
            batch_paths = image_paths[i : i + batch_size]
            batch = torch.stack([load_image_tensor(p) for p in batch_paths], dim=0).to(device)
            feat = model(batch)
            acts.append(feat.detach().float().cpu().numpy())

    return np.concatenate(acts, axis=0)


def compute_statistics(activations: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if activations.shape[0] == 0:
        raise ValueError("空激活张量，无法计算 FID")
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    if sigma.ndim == 0:
        sigma = np.array([[sigma]], dtype=np.float64)
    return mu, sigma


def calculate_fid(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(fid)


def evaluate_cell(
    gen_paths: List[Path],
    real_paths: List[Path],
    device: torch.device,
    batch_size: int,
    category: str,
    defect_type: str,
) -> Tuple[int, int, float]:
    if len(gen_paths) == 0 or len(real_paths) == 0:
        return len(gen_paths), len(real_paths), float("nan")

    gen_acts = compute_activations(gen_paths, device=device, batch_size=batch_size, desc=f"{category}/{defect_type} gen")
    real_acts = compute_activations(real_paths, device=device, batch_size=batch_size, desc=f"{category}/{defect_type} real")
    mu_g, sigma_g = compute_statistics(gen_acts)
    mu_r, sigma_r = compute_statistics(real_acts)
    return len(gen_paths), len(real_paths), calculate_fid(mu_g, sigma_g, mu_r, sigma_r)


# ==========================================
# 2. 核心算法：Inception Score (物体级汇总计算)
# ==========================================
def compute_inception_score(image_paths: List[Path], model, device, batch_size=32, splits=10):
    if len(image_paths) < 2:
        return float("nan"), float("nan")
    
    probs_list = []
    with torch.no_grad():
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i : i + batch_size]
            batch = torch.stack([load_image_tensor_for_inception(p) for p in batch_paths], dim=0).to(device)
            logits = model(batch)
            probs = F.softmax(logits, dim=1)
            probs_list.append(probs.cpu())

    probs_all = torch.cat(probs_list, dim=0)
    n = probs_all.shape[0]
    num_splits = max(1, min(splits, n))
    split_scores = []

    for k in range(num_splits):
        part = probs_all[k * n // num_splits : (k + 1) * n // num_splits]
        if part.numel() == 0:
            continue
        py = torch.mean(part, dim=0, keepdim=True)
        kl = part * (torch.log(part + 1e-12) - torch.log(py + 1e-12))
        score = torch.exp(torch.mean(torch.sum(kl, dim=1))).item()
        split_scores.append(score)

    return float(np.mean(split_scores)), float(np.std(split_scores))

# ==========================================
# 3. 核心算法：IC-L (精准匹配参考图)
# ==========================================
def get_anchor_images(obj_name: str, defect_dir_name: str, dataset_root: Path) -> List[Path]:
    anchor_dir = dataset_root / obj_name / "test" / defect_dir_name
    if not anchor_dir.is_dir():
        return []
    return sorted([p for p in anchor_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

def compute_ic_l_clustered(image_paths, obj_name, defect_dir_name, dataset_root, lpips_model, device, cluster_size=50, seed=42):
    anchors = get_anchor_images(obj_name, defect_dir_name, Path(dataset_root))
    print(f"    [IC-L] {obj_name}/{defect_dir_name} anchors={len(anchors)} gen_images={len(image_paths)}")
    if not anchors or len(image_paths) < 2:
        return float("nan")

    with torch.no_grad():
        anchor_ts = [load_image_tensor_for_lpips(p).to(device) for p in anchors]
        clusters = [[] for _ in range(len(anchor_ts))]

        for p in image_paths:
            gen_t = load_image_tensor_for_lpips(p).to(device)
            dists = [lpips_model(a, gen_t).item() for a in anchor_ts]
            clusters[np.argmin(dists)].append(p)

        rng = random.Random(seed)
        means = []
        for files in clusters:
            if len(files) < 2:
                continue
            rng.shuffle(files)
            sel = files[:cluster_size]
            dists = []
            for i in range(len(sel)):
                t1 = load_image_tensor_for_lpips(sel[i]).to(device)
                for j in range(i + 1, len(sel)):
                    t2 = load_image_tensor_for_lpips(sel[j]).to(device)
                    dists.append(float(lpips_model(t1, t2).item()))
            if dists:
                means.append(float(np.mean(dists)))

    return float(np.mean(means)) if means else float("nan")


def discover_generated_objects(gen_root: Path) -> List[str]:
    return sorted([p.name for p in gen_root.iterdir() if p.is_dir()]) if gen_root.is_dir() else []


def discover_generated_defects(gen_root: Path, category: str) -> List[str]:
    pred_root = gen_root / category / "generated_datasets"
    if not pred_root.is_dir():
        return []
    return sorted([p.name for p in pred_root.iterdir() if p.is_dir()])


def discover_generated_images(gen_root: Path, category: str, defect_type: str) -> List[Path]:
    pred_root = gen_root / category / "generated_datasets" / defect_type
    if not pred_root.is_dir():
        return []
    return sorted([p for p in pred_root.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS])


def build_object_level_report(gen_root: Path, dataset_root: Path, batch_size: int, device: torch.device, metric: str) -> str:
    inception = None
    if metric in ["is", "all"]:
        inception = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1, transform_input=False).to(device).eval()

    lpips_model = None
    if metric in ["ic-l", "all"]:
        lpips_model = lpips.LPIPS(net='vgg').to(device).eval()

    all_object_results = []
    for obj in discover_generated_objects(gen_root):
        defects = discover_generated_defects(gen_root, obj)
        if not defects:
            continue

        obj_all_imgs: List[Path] = []
        defect_counts: Dict[str, int] = {}
        ic_l_list: List[float] = []
        print(f"\nProcessing Object: {obj}")
        for defect_type in defects:
            imgs = discover_generated_images(gen_root, obj, defect_type)
            if not imgs:
                continue
            obj_all_imgs.extend(imgs)
            defect_counts[defect_type] = len(imgs)
            if lpips_model is not None:
                val = compute_ic_l_clustered(imgs, obj, defect_type, dataset_root, lpips_model, device)
                if not math.isnan(val):
                    ic_l_list.append(val)
                print(f"  - {defect_type}: IC-L = {val:.4f}")

        if not obj_all_imgs:
            continue
        is_m, is_s = (float("nan"), float("nan"))
        if inception is not None:
            is_m, is_s = compute_inception_score(obj_all_imgs, inception, device, batch_size)
        obj_icl = np.mean(ic_l_list) if ic_l_list else float("nan")
        all_object_results.append(ObjectMetrics(obj, len(obj_all_imgs), is_m, is_s, obj_icl, defect_counts))
        print(f"  Summary -> IS: {is_m:.4f}±{is_s:.4f} | IC-L: {obj_icl:.4f}")

    md = "# Quality Evaluation Report\n\n"
    md += "| Object | Total Images | Inception Score | IC-L (Diversity) | Composition |\n"
    md += "| :--- | :---: | :---: | :---: | :--- |\n"
    for r in all_object_results:
        is_str = f"{r.is_mean:.4f}±{r.is_std:.4f}" if not math.isnan(r.is_mean) else "N/A"
        icl_str = f"{r.icl_mean:.4f}" if not math.isnan(r.icl_mean) else "N/A"
        composition = ", ".join([f"{k}({v})" for k, v in r.defects.items()])
        md += f"| {r.object_name} | {r.num_images} | {is_str} | {icl_str} | {composition} |\n"
    return md


def build_fid_report(gen_root: Path, dataset_root: Path, batch_size: int, device: torch.device) -> str:
    categories = discover_categories(dataset_root)
    if not categories:
        raise RuntimeError(f"未找到类别，请检查 dataset_root: {dataset_root}")

    defect_types_by_category: Dict[str, List[str]] = {category: discover_defect_types(dataset_root, category) for category in categories}
    metrics: Dict[Tuple[str, str], CellFIDMetrics] = {}
    for category in categories:
        for defect_type in defect_types_by_category[category]:
            gen_paths = discover_generated_images(gen_root, category, defect_type)
            real_paths = discover_real_images(dataset_root, category, defect_type)
            print(f"[FID] {category}/{defect_type} | gen={len(gen_paths)} real={len(real_paths)}")
            num_gen, num_real, fid_val = evaluate_cell(gen_paths, real_paths, device=device, batch_size=batch_size, category=category, defect_type=defect_type)
            metrics[(category, defect_type)] = CellFIDMetrics(category, defect_type, num_gen, num_real, fid_val)

    category_averages = print_results_table(categories, defect_types_by_category, metrics)
    return build_markdown_table(categories, defect_types_by_category, metrics, category_averages)


# ==========================================
# 4. 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gen_root", type=str, required=True)
    parser.add_argument("--dataset_root", type=str, required=True)
    parser.add_argument("--pred_subdir_name", type=str, default="generated_datasets")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--metric", type=str, default="all")
    parser.add_argument("--output_md", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    gen_path = Path(args.gen_root)

    if args.metric == "fid":
        md = build_fid_report(gen_path, Path(args.dataset_root), args.batch_size, device)
    else:
        md = build_object_level_report(gen_path, Path(args.dataset_root), args.batch_size, device, args.metric)

    if args.output_md:
        output_path = Path(args.output_md)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(md, encoding="utf-8")
        print(f"\n[DONE] Markdown report saved to: {args.output_md}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
