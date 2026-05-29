import os
import cv2
import torch
import random
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF


def list_seg_mask_categories(root_dir):
    if not os.path.isdir(root_dir):
        return []
    names = []
    for name in sorted(os.listdir(root_dir)):
        path = os.path.join(root_dir, name)
        if not os.path.isdir(path):
            continue
        has_good = os.path.isdir(os.path.join(path, "good"))
        has_gt = os.path.isdir(os.path.join(path, "Ground_truth")) or os.path.isdir(os.path.join(path, "ground_truth"))
        if has_good or has_gt:
            names.append(name)
    return names


def _is_training_defect(defect_name):
    return defect_name.lower() != "good"


def list_training_defect_types(gt_dir):
    if not os.path.isdir(gt_dir):
        return []
    return sorted(
        d
        for d in os.listdir(gt_dir)
        if os.path.isdir(os.path.join(gt_dir, d)) and _is_training_defect(d)
    )


class Mask2MaskDataset(Dataset):
    def __init__(
        self,
        root_dir,
        category="screw",
        defect_type=None,
        img_size=512,
        augment=True,
    ):
        self.root_dir = root_dir
        self.category = category
        self.img_size = img_size
        self.augment = augment
        self.samples = []

        gt_dir = os.path.join(root_dir, category, "Ground_truth")
        if not os.path.exists(gt_dir):
            gt_dir = os.path.join(root_dir, category, "ground_truth")
        test_dir = os.path.join(root_dir, category, "test")

        if os.path.exists(gt_dir):
            if defect_type is not None:
                defect_types = [defect_type]
            else:
                defect_types = [
                    d for d in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, d))
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
                    fg_mask_path = os.path.join(defect_test_dir, base_name)
                    if not os.path.exists(fg_mask_path):
                        fg_mask_path = os.path.join(defect_test_dir, gt_file)
                    gt_mask_path = os.path.join(defect_gt_dir, gt_file)

                    if os.path.exists(fg_mask_path) and os.path.exists(gt_mask_path):
                        self.samples.append((fg_mask_path, gt_mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fg_path, gt_path = self.samples[idx]

        fg_mask = cv2.imread(fg_path, cv2.IMREAD_GRAYSCALE)
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

        fg_pil = Image.fromarray(fg_mask)
        gt_pil = Image.fromarray(gt_mask)

        if self.augment:
            if random.random() > 0.5:
                fg_pil = TF.hflip(fg_pil)
                gt_pil = TF.hflip(gt_pil)

            if random.random() > 0.5:
                fg_pil = TF.vflip(fg_pil)
                gt_pil = TF.vflip(gt_pil)

        fg_tensor = TF.to_tensor(fg_pil)
        gt_tensor = TF.to_tensor(gt_pil)

        fg_tensor = TF.resize(
            fg_tensor,
            [self.img_size, self.img_size],
            interpolation=TF.InterpolationMode.NEAREST,
        )
        gt_tensor = TF.resize(
            gt_tensor,
            [self.img_size, self.img_size],
            interpolation=TF.InterpolationMode.NEAREST,
        )

        fg_tensor = (fg_tensor > 0.5).float()
        gt_tensor = (gt_tensor > 0.5).float()

        return fg_tensor, gt_tensor
