import argparse
import os
import shutil
from collections import defaultdict
from typing import Dict, Mapping, Optional, Sequence

import cv2
import numpy as np
from tqdm import tqdm

from seg_method.seg_SAM2 import build_custom_dataset as run_sam2
from seg_method.seg_SAM3 import build_custom_dataset as run_sam3
from seg_method.seg_geometric import build_custom_dataset as run_geometric

# 统一输入/输出配置（仅在本文件定义）
ORIGINAL_MVTEC_ROOT = "/home/wyh/data/mvtec_ad/MVTEC_AD_512"
SEG_MASK_ROOT = "/home/wyh/data/mvtec_ad/seg_mask"
SAM2_MODEL_TYPE = "/d242/wyh/model/SAM/sam2_b.pt"
SAM3_MODEL_TYPE = "/d242/wyh/model/SAM/sam3.pt"

# 类别到分割方法的路由表
CATEGORY_TO_METHOD = {
    "carpet": "texture",
    "grid": "geometric",
    "leather": "texture",
    "tile": "texture",
    "wood": "texture",
    "bottle": "sam2",
    "cable": "sam2",
    "capsule": "sam3",
    "hazelnut": "geometric",
    "metal_nut": "sam2",
    "pill": "sam3",
    "screw": "sam3",
    "toothbrush": "geometric",
    "transistor": "sam3",
    "zipper": "sam3",
}

def copy_ground_truth_if_needed(category, original_mvtec_root, new_dataset_root):
    """直接复制指定类别的 ground_truth 文件夹"""
    orig_cat_dir = os.path.join(original_mvtec_root, category)
    if not os.path.exists(orig_cat_dir):
        return

    new_cat_dir = os.path.join(new_dataset_root, category)
    new_gt_dir = os.path.join(new_cat_dir, "ground_truth")
    os.makedirs(new_gt_dir, exist_ok=True)

    orig_gt_dir = os.path.join(orig_cat_dir, "ground_truth")
    if not os.path.exists(orig_gt_dir):
        return

    for defect in [d for d in os.listdir(orig_gt_dir) if os.path.isdir(os.path.join(orig_gt_dir, d))]:
        orig_defect_dir = os.path.join(orig_gt_dir, defect)
        new_defect_dir = os.path.join(new_gt_dir, defect)
        os.makedirs(new_defect_dir, exist_ok=True)
        for file in [f for f in os.listdir(orig_defect_dir) if f.endswith(".png")]:
            src = os.path.join(orig_defect_dir, file)
            dst = os.path.join(new_defect_dir, file)
            if not os.path.exists(dst):
                shutil.copy2(src, dst)

def run_texture(original_mvtec_root, new_dataset_root, categories):
    """处理纹理类别，直接生成全白掩码，不进行模型推理"""
    for category in categories:
        print(f"\n[*] 正在处理类别: {category} (Texture - 生成全白 Mask)")
        orig_cat_dir = os.path.join(original_mvtec_root, category)
        new_cat_dir = os.path.join(new_dataset_root, category)

        # 拷贝 Ground Truth
        copy_ground_truth_if_needed(category, original_mvtec_root, new_dataset_root)

        # ---------------------------------------------------------
        # 模块 A: 处理 test 路径
        # ---------------------------------------------------------
        orig_test_dir = os.path.join(orig_cat_dir, 'test')
        new_test_dir = os.path.join(new_cat_dir, 'test')
        if os.path.exists(orig_test_dir):
            for defect_type in os.listdir(orig_test_dir):
                orig_defect_dir = os.path.join(orig_test_dir, defect_type)
                
                if not os.path.isdir(orig_defect_dir):
                    continue
                    
                new_defect_dir = os.path.join(new_test_dir, defect_type)
                os.makedirs(new_defect_dir, exist_ok=True)

                for img_name in tqdm(os.listdir(orig_defect_dir), desc=f"test - {defect_type}"):
                    img_path = os.path.join(orig_defect_dir, img_name)
                    out_path = os.path.join(new_defect_dir, img_name)
                    
                    img = cv2.imread(img_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        # 生成与原图同样大小的全白掩码 (255)
                        white_mask = np.ones((h, w), dtype=np.uint8) * 255
                        cv2.imwrite(out_path, white_mask)

        # ---------------------------------------------------------
        # 模块 B: 处理 train/good 路径
        # ---------------------------------------------------------
        orig_train_dir = os.path.join(orig_cat_dir, 'train', 'good')
        new_train_dir = os.path.join(new_cat_dir, 'good')
        if os.path.exists(orig_train_dir):
            os.makedirs(new_train_dir, exist_ok=True)
            for img_name in tqdm(os.listdir(orig_train_dir), desc="train/good"):
                img_path = os.path.join(orig_train_dir, img_name)
                out_path = os.path.join(new_train_dir, img_name)
                
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    white_mask = np.ones((h, w), dtype=np.uint8) * 255
                    cv2.imwrite(out_path, white_mask)


def _filter_route_by_categories(
    route: Mapping[str, str], only_categories: Optional[Sequence[str]]
) -> Dict[str, str]:
    """仅保留指定类别；only_categories 为 None 时返回完整路由。"""
    if only_categories is None:
        return dict(route)
    if not only_categories:
        raise ValueError("only_categories 为空列表时无法执行；请传入至少一个类别名，或省略该参数以处理全部类别。")
    
    requested = {c.strip() for c in only_categories if c and str(c).strip()}
    if not requested:
        raise ValueError("only_categories 未包含有效类别名；请传入至少一个类别名，或省略该参数以处理全部类别。")
        
    known = set(route.keys())
    unknown = requested - known
    if unknown:
        raise ValueError(f"未知类别: {sorted(unknown)}。有效类别: {sorted(known)}")
        
    return {k: v for k, v in route.items() if k in requested}


def dispatch_by_category(
    category_to_method: Optional[Mapping[str, str]] = None,
    only_categories: Optional[Sequence[str]] = None,
):
    """根据类别路由到不同分割脚本，子脚本内部已实现 test 与 train/good 的全量处理。"""
    base_route = category_to_method or CATEGORY_TO_METHOD
    route = _filter_route_by_categories(base_route, only_categories)
    
    # 将类别按方法进行分组
    method_to_categories = defaultdict(list)
    for category, method in route.items():
        method_to_categories[method].append(category)

    # 1. 调度 Texture 纹理类别方法（全白掩码输出）
    texture_categories = method_to_categories.get("texture", [])
    if texture_categories:
        print(f"\n{'=' * 50}")
        print(f"[*] 方法 [texture] 处理类别: {texture_categories}")
        run_texture(
            original_mvtec_root=ORIGINAL_MVTEC_ROOT,
            new_dataset_root=SEG_MASK_ROOT,
            categories=texture_categories,
        )

    # 2. 调度 Geometric 几何分割方法
    geometric_categories = method_to_categories.get("geometric", [])
    if geometric_categories:
        print(f"\n{'=' * 50}")
        print(f"[*] 方法 [geometric] 推理类别: {geometric_categories}")
        for category in geometric_categories:
            copy_ground_truth_if_needed(category, ORIGINAL_MVTEC_ROOT, SEG_MASK_ROOT)
            
        run_geometric(
            original_mvtec_root=ORIGINAL_MVTEC_ROOT,
            new_dataset_root=SEG_MASK_ROOT,
            categories=geometric_categories,
        )

    # 3. 调度 SAM 2 分割方法
    sam2_categories = method_to_categories.get("sam2", [])
    if sam2_categories:
        print(f"\n{'=' * 50}")
        print(f"[*] 方法 [sam2] 推理类别: {sam2_categories}")
        # SAM2/3 代码内如有 copy ground truth 的逻辑，这里保持其原有机制即可；
        # 若 SAM 没有，也可在 run_sam2 调用前补充 copy_ground_truth_if_needed
        for category in sam2_categories:
            copy_ground_truth_if_needed(category, ORIGINAL_MVTEC_ROOT, SEG_MASK_ROOT)
        run_sam2(
            original_mvtec_root=ORIGINAL_MVTEC_ROOT,
            new_dataset_root=SEG_MASK_ROOT,
            model_type=SAM2_MODEL_TYPE,
            categories=sam2_categories,
        )

    # 4. 调度 SAM 3 分割方法
    sam3_categories = method_to_categories.get("sam3", [])
    if sam3_categories:
        print(f"\n{'=' * 50}")
        print(f"[*] 方法 [sam3] 推理类别: {sam3_categories}")
        for category in sam3_categories:
            copy_ground_truth_if_needed(category, ORIGINAL_MVTEC_ROOT, SEG_MASK_ROOT)
        run_sam3(
            original_mvtec_root=ORIGINAL_MVTEC_ROOT,
            new_dataset_root=SEG_MASK_ROOT,
            model_type=SAM3_MODEL_TYPE,
            categories=sam3_categories,
        )

    # 兜底校验
    unknown_methods = set(method_to_categories.keys()) - {"texture", "geometric", "sam2", "sam3"}
    if unknown_methods:
        raise ValueError(
            f"存在不支持的方法: {sorted(unknown_methods)}。可选方法仅支持: ['texture', 'geometric', 'sam2', 'sam3']"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="按 MVTec 类别生成分割 mask。子模块已自动处理 test 与 train/good 路径。"
    )
    parser.add_argument(
        "-c",
        "--category",
        action="append",
        dest="categories",
        metavar="NAME",
        help="只处理指定类别（可多次传入，例如 -c bottle -c carpet）；缺省则处理全部",
    )

    args = parser.parse_args()
    dispatch_by_category(only_categories=args.categories)
    print("\n[+] 全部分割任务已执行完毕。")