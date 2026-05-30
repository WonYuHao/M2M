#!/usr/bin/env python3
"""Copy prior mask images into a cleaned directory structure.

Source layout:
  /d242/wyh/M2M/genmask_cvae/inference_out/{category}/{defect_type}_ep200/{000}_prior.png

Target layout:
  /home/wyh/data/mvtec_ad/genmask_mask_good/{category}/{defect_type}/{000}.png

This script:
- copies only files ending with `_prior.png`
- removes the `_ep200` suffix from the defect folder name
- removes the `_prior` suffix from the image filename
"""

from __future__ import annotations

import shutil
from pathlib import Path

SOURCE_ROOT = Path("/d242/wyh/M2M/genmask_cvae/inference_out")
TARGET_ROOT = Path("/home/wyh/data/mvtec_ad/genmask_mask_good")


def clean_defect_name(name: str) -> str:
    return name[:-6] if name.endswith("_ep200") else name


def clean_image_name(name: str) -> str:
    if name.endswith("_prior.png"):
        return name[: -len("_prior.png")] + ".png"
    return name


def main() -> None:
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"Source root not found: {SOURCE_ROOT}")

    copied = 0
    skipped = 0

    for category_dir in sorted(p for p in SOURCE_ROOT.iterdir() if p.is_dir()):
        category = category_dir.name

        for defect_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
            defect_type = clean_defect_name(defect_dir.name)
            target_dir = TARGET_ROOT / category / defect_type
            target_dir.mkdir(parents=True, exist_ok=True)

            for img_path in sorted(defect_dir.glob("*_prior.png")):
                if not img_path.is_file():
                    skipped += 1
                    continue

                target_name = clean_image_name(img_path.name)
                target_path = target_dir / target_name
                shutil.copy2(img_path, target_path)
                copied += 1
                print(f"[copied] {img_path} -> {target_path}")

    print(f"Done. copied={copied}, skipped={skipped}, target_root={TARGET_ROOT}")


if __name__ == "__main__":
    main()
