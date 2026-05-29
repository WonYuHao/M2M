import os
import cv2
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def list_seg_mask_categories(root_dir):
    """seg_mask 根目录下，含有 Ground_truth 的类别名列表（有序）。"""
    if not os.path.isdir(root_dir):
        return []
    names = []
    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if os.path.isdir(path) and os.path.isdir(
            os.path.join(path, "Ground_truth")
        ):
            names.append(name)
    return names


def _is_training_defect(defect_name):
    """训练不使用 good；good 仅作推理时的正常前景来源。"""
    return defect_name.lower() != "good"


def list_training_defect_types(gt_dir):
    """Ground_truth 下用于训练的缺陷子文件夹（排除 good）。"""
    if not os.path.isdir(gt_dir):
        return []
    return sorted(
        d
        for d in os.listdir(gt_dir)
        if os.path.isdir(os.path.join(gt_dir, d)) and _is_training_defect(d)
    )


class Mask2MaskDataset(Dataset):
    """
    缺陷样本对均为二值 mask：
    - 前景 mask：已对异常测试图做完前景分割（与正常图分割流程一致）
    - 异常真值 mask：Ground_truth 中的标注

    训练学习「二值前景 ↔ 二值异常区域」的空间关系；推理时仅输入正常图的二值前景，得到二值异常位置预测（由训练脚本外的阈值化产生）。
    """

    def __init__(
        self,
        root_dir,
        category="screw",
        defect_type=None,
        img_size=512,
    ):
        self.root_dir = root_dir
        self.category = category
        self.img_size = img_size
        self.samples = []

        gt_dir = os.path.join(root_dir, category, "Ground_truth")
        test_dir = os.path.join(root_dir, category, "test")

        if os.path.exists(gt_dir):
            if defect_type is not None:
                defect_types = [defect_type]
            else:
                defect_types = [
                    d
                    for d in os.listdir(gt_dir)
                    if os.path.isdir(os.path.join(gt_dir, d))
                ]

            defect_types = [d for d in defect_types if _is_training_defect(d)]

            for defect in defect_types:
                defect_gt_dir = os.path.join(gt_dir, defect)
                defect_test_dir = os.path.join(test_dir, defect)

                if not os.path.exists(defect_gt_dir):
                    continue

                gt_files = [f for f in os.listdir(defect_gt_dir) if f.endswith(".png")]
                for gt_file in gt_files:
                    base_name = gt_file.replace("_mask", "")
                    sam_mask_path = os.path.join(defect_test_dir, base_name)
                    gt_mask_path = os.path.join(defect_gt_dir, gt_file)

                    if os.path.exists(sam_mask_path) and os.path.exists(gt_mask_path):
                        self.samples.append((sam_mask_path, gt_mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sam_path, gt_path = self.samples[idx]

        sam_mask = cv2.imread(sam_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        sam_pil = Image.fromarray(sam_mask)
        gt_pil = Image.fromarray(gt_mask)

        if random.random() > 0.5:
            sam_pil = TF.hflip(sam_pil)
            gt_pil = TF.hflip(gt_pil)

        if random.random() > 0.5:
            sam_pil = TF.vflip(sam_pil)
            gt_pil = TF.vflip(gt_pil)

        sam_tensor = TF.to_tensor(sam_pil)
        gt_tensor = TF.to_tensor(gt_pil)

        sam_tensor = TF.resize(
            sam_tensor,
            [self.img_size, self.img_size],
            interpolation=TF.InterpolationMode.NEAREST,
        )
        gt_tensor = TF.resize(
            gt_tensor,
            [self.img_size, self.img_size],
            interpolation=TF.InterpolationMode.NEAREST,
        )

        sam_tensor = (sam_tensor > 0.5).float()
        gt_tensor = (gt_tensor > 0.5).float()

        return sam_tensor, gt_tensor
