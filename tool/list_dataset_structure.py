import argparse
import json
from pathlib import Path


def list_categories_defects(dataset_root: Path):
    """
    读取数据集目录结构，输出 categories/defect 列表。

    默认结构：
      dataset_root/<category>/Ground_truth/<defect>

    仅收集 Ground_truth 下的缺陷子目录，跳过 good。
    """
    pairs = []
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"dataset_root 不存在: {dataset_root}")

    for category_dir in sorted([p for p in dataset_root.iterdir() if p.is_dir()]):
        gt_dir = category_dir / "Ground_truth"
        if not gt_dir.is_dir():
            continue

        for defect_dir in sorted([p for p in gt_dir.iterdir() if p.is_dir()]):
            if defect_dir.name.lower() == "good":
                continue
            pairs.append(f"{category_dir.name}/{defect_dir.name}")

    return pairs


def parse_args():
    parser = argparse.ArgumentParser(description="输出数据集目录中的 categories/defect 列表")
    parser.add_argument("--dataset_root", type=str, required=True, help="数据集根目录")
    parser.add_argument("--output_json", type=str, default=None, help="可选：保存为 JSON 文件")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_root = Path(args.dataset_root)
    pairs = list_categories_defects(dataset_root)

    for item in pairs:
        print(item)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n已保存: {out_path}")


if __name__ == "__main__":
    main()
