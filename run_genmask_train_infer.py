import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


DEFAULT_DATASET_ROOT = "/home/wyh/data/mvtec_ad/seg_mask"
DEFAULT_OUTPUT_ROOT = "/d242/wyh/M2M"


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def run_command(cmd):
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, check=True)


def has_weights(weights_dir: str) -> bool:
    return any(Path(weights_dir).glob("*_*.pth"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--date_code",
        type=str,
        default=None,
        help="日期编码，默认使用当天 YYYYMMDD；用于固定训练和推理使用同一批权重目录",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train_infer",
        choices=["train", "infer", "train_infer"],
        help="运行模式: train(仅训练), infer(仅推理), train_infer(先后进行训练和推理)",
    )
    parser.add_argument(
        "--train_script",
        type=str,
        default=str(Path(__file__).parent / "code4maskgeneration" / "train_mask_generation_cvae.py"),
    )
    parser.add_argument(
        "--infer_script",
        type=str,
        default=str(Path(__file__).parent / "code4maskgeneration" / "infer_mask_generation_cvae.py"),
    )
    args = parser.parse_args()

    out_root = Path(args.output_root)
    ensure_dir(out_root)

    date_code = args.date_code or datetime.now().strftime("%Y%m%d")
    python = sys.executable
    weights_dir = str(out_root / "genmask_cvae" / f"weights_{date_code}")
    infer_out_dir = str(Path("/home/wyh/data/mvtec_ad") / f"generated_mask_{date_code}")

    do_train = args.mode in ["train", "train_infer"]
    do_infer = args.mode in ["infer", "train_infer"]

    if do_train:
        train_cmd = [
            python,
            args.train_script,
            "--dataset_root",
            args.dataset_root,
            "--weights_dir",
            weights_dir,
        ]
        print("\n[*] ======= 开始训练 CVAE =======")
        run_command(train_cmd)
    else:
        print("\n[*] ======= 跳过训练阶段 =======")

    if do_infer:
        print("\n[*] ======= 开始推理 CVAE =======")
        if not has_weights(weights_dir):
            print(f"[!] 推理中止：在 {weights_dir} 目录下未找到训练权重")
        else:
            ensure_dir(Path(infer_out_dir))
            infer_cmd = [
                python,
                args.infer_script,
                "--dataset_root",
                args.dataset_root,
                "--weights_dir",
                weights_dir,
                "--out_dir",
                infer_out_dir,
                "--run_all_good",
            ]
            run_command(infer_cmd)

    print(f"\n[*] 任务全部完成。权重保存在: {weights_dir}")
    print(f"[*] 推理结果保存在: {infer_out_dir}")


if __name__ == "__main__":
    main()