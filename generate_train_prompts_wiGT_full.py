import argparse
import json
import os
from glob import glob
from pathlib import Path


IMAGE_EXTS = ("png")


def masked_to_ground_truth_path(masked_path: Path):
    """
    Map a "masked rgb" image path to the corresponding binary mask path.

    Strict dataset convention:
      .../<class>/masked/<defect>/<name>.png
      .../<class>/ground_truth/<defect>/<name>_mask.png
    """
    parts = masked_path.parts
    try:
        masked_idx = parts.index("masked")
    except ValueError:
        raise ValueError(f"Expected 'masked' in path: {masked_path}")
    stem = masked_path.stem
    ext = masked_path.suffix  # includes the dot, e.g. ".png"
    if ext.lower() != ".png":
        # This script is intended for the strict 512x512 .png dataset.
        raise ValueError(f"Expected .png under masked/, got: {masked_path}")

    # .../<class>/masked/<defect>/<name>.png -> .../<class>/ground_truth/<defect>/<name>_mask.png
    return Path(
        *parts[:masked_idx],
        "ground_truth",
        *parts[masked_idx + 1 : -1],
        stem + "_mask" + ext,
    )


def masked_to_test_target_path(masked_path: Path):
    """
    Map a "masked rgb" image path to the corresponding anomalous RGB image path.

    Expected:
      .../<class>/masked/<defect>/<name>.<ext>
      .../<class>/test/<defect>/<name>.<ext>
    """
    parts = masked_path.parts
    try:
        masked_idx = parts.index("masked")
    except ValueError:
        raise ValueError(f"Expected 'masked' in path: {masked_path}")

    return Path(*parts[:masked_idx], "test", *parts[masked_idx + 1 :])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default="/home/wyh/data/mvtec_ad/MVTEC_AD_512",
        help="MVTec 数据根目录（包含 <class>/masked/<defect>/...）",
    )
    parser.add_argument(
        "--prompt_dirname",
        default="prompt",
        help="prompt 文本所在目录名（默认: prompt）",
    )
    parser.add_argument("--out", default="train_prompts_wiGT_full.json", help="输出 JSON 路径")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if not data_root.is_dir():
        raise FileNotFoundError(str(data_root))

    all_entries = []

    # Strictly traverse actual dataset:
    # data_root/<class>/masked/<defect>/*.png
    # prompt text: data_root/<class>/<prompt_dirname>/<defect>/<name>.txt
    # ground truth mask: .../ground_truth/<defect>/<name>_mask.png
    # target image: .../test/<defect>/<name>.png
    for cls_dir in sorted(data_root.iterdir()):
        if not cls_dir.is_dir():
            continue
        cls = cls_dir.name
        masked_root = cls_dir / "masked"
        if not masked_root.is_dir():
            continue

        for defect_dir in sorted(masked_root.iterdir()):
            if not defect_dir.is_dir():
                continue
            defect = defect_dir.name
            if defect.lower() == "good":
                continue

            masked_images = sorted(defect_dir.glob("*.png"))
            if not masked_images:
                continue

            for img_path in masked_images:
                name = img_path.stem
                prompt_txt_path = cls_dir / args.prompt_dirname / defect / f"{name}.txt"
                if not prompt_txt_path.exists():
                    raise FileNotFoundError(f"Missing prompt txt for {img_path} -> {prompt_txt_path}")

                with prompt_txt_path.open("r", encoding="utf-8") as f:
                    prompt = f.read().strip()
                if not prompt:
                    raise ValueError(f"Empty prompt txt for {prompt_txt_path}")

                mask_path = masked_to_ground_truth_path(img_path)
                if not mask_path.exists():
                    raise FileNotFoundError(f"Missing ground_truth(mask) for {img_path} -> {mask_path}")

                target_path = masked_to_test_target_path(img_path)
                if not target_path.exists():
                    raise FileNotFoundError(f"Missing target image for {img_path} -> {target_path}")

                all_entries.append(
                    {
                        "image": str(img_path),
                        "ground_truth": str(mask_path),
                        "target": str(target_path),
                        "prompt": prompt,
                        "_info": f"{cls}/{defect}",
                    }
                )

    # Deterministic output ordering
    all_entries = sorted(all_entries, key=lambda x: (x["_info"], x["image"]))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(all_entries, f, ensure_ascii=False, indent=4)

    print(f"Generated {len(all_entries)} entries -> {out_path}")


if __name__ == "__main__":
    main()

