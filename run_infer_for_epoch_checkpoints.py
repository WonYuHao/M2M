import argparse
import os
import subprocess
from pathlib import Path


DEFAULT_TRAIN_OUTPUT_DIR = "/d242/wyh/model/M2M_controlnet"
DEFAULT_BASE_MODEL = "/d242/wyh/model/ldm/inpainting/sd-v1-5-inpainting.ckpt"
DEFAULT_DATASET_ROOT = "/home/wyh/data/mvtec_ad/MVTEC_AD_512"
DEFAULT_MASK_ROOT = Path(__file__).resolve().parent / "outputs" / "spatial_prior"
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "post_train_infer"
DEFAULT_EPOCHS = "20,30,40,50"
DEFAULT_TRAIN_JSON_PATH = "/home/wyh/data/mvtec_ad/train_prompts_wiGT_full.json"


def _append_optional_arg(command: list[str], flag: str, value):
    if value is not None:
        command.extend([flag, str(value)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_device", type=str, default="3", help="整个流程使用的 CUDA 设备编号")
    parser.add_argument("--python_executable", type=str, default="python")
    parser.add_argument("--train_script", type=str, default=str(Path(__file__).resolve().parent / "train_controlnet_inpaint_prompt.py"))
    parser.add_argument("--train_json_path", type=str, default=DEFAULT_TRAIN_JSON_PATH)
    parser.add_argument("--train_output_dir", type=str, default=DEFAULT_TRAIN_OUTPUT_DIR)
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--lr_scheduler", type=str, default="constant")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--mixed_precision", type=str, default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_epochs", type=int, default=10)
    parser.add_argument("--checkpoints_total_limit", type=int, default=None)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--prediction_type", type=str, default=None, choices=[None, "epsilon", "v_prediction"])
    parser.add_argument("--allow_tf32", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--set_grads_to_none", action="store_true")
    parser.add_argument("--validation_steps", type=int, default=0)
    parser.add_argument("--num_validation_samples", type=int, default=4)
    parser.add_argument("--validation_num_inference_steps", type=int, default=20)
    parser.add_argument("--infer_epochs", type=str, default=DEFAULT_EPOCHS, help="训练结束后自动推理的 epoch 列表")
    parser.add_argument("--infer_output_root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--infer_dataset_root", type=str, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--infer_mask_root", type=str, default=str(DEFAULT_MASK_ROOT))
    parser.add_argument("--infer_prompt_dirname", type=str, default="prompt")
    parser.add_argument("--infer_num_inference_steps", type=int, default=30)
    parser.add_argument("--infer_guidance_scale", type=float, default=7.5)
    parser.add_argument("--infer_controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--infer_negative_prompt", type=str, default=None)
    args = parser.parse_args()

    train_script = Path(args.train_script)
    if not train_script.is_file():
        raise FileNotFoundError(f"Training script not found: {train_script}")

    output_root = Path(args.infer_output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device

    command = [
        args.python_executable,
        str(train_script),
        "--base_model",
        args.base_model,
        "--output_dir",
        args.train_output_dir,
        "--image_size",
        str(args.image_size),
        "--batch_size",
        str(args.batch_size),
        "--num_train_epochs",
        str(args.num_train_epochs),
        "--learning_rate",
        str(args.learning_rate),
        "--lr_scheduler",
        args.lr_scheduler,
        "--lr_warmup_steps",
        str(args.lr_warmup_steps),
        "--gradient_accumulation_steps",
        str(args.gradient_accumulation_steps),
        "--max_grad_norm",
        str(args.max_grad_norm),
        "--mixed_precision",
        args.mixed_precision,
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(args.seed),
        "--save_epochs",
        str(args.save_epochs),
        "--validation_steps",
        str(args.validation_steps),
        "--num_validation_samples",
        str(args.num_validation_samples),
        "--validation_num_inference_steps",
        str(args.validation_num_inference_steps),
        "--run_inference_after_training",
        "--infer_epochs",
        args.infer_epochs,
        "--infer_output_root",
        args.infer_output_root,
        "--infer_dataset_root",
        args.infer_dataset_root,
        "--infer_mask_root",
        args.infer_mask_root,
        "--infer_prompt_dirname",
        args.infer_prompt_dirname,
        "--infer_num_inference_steps",
        str(args.infer_num_inference_steps),
        "--infer_guidance_scale",
        str(args.infer_guidance_scale),
        "--infer_controlnet_conditioning_scale",
        str(args.infer_controlnet_conditioning_scale),
    ]

    _append_optional_arg(command, "--train_json_path", args.train_json_path)
    _append_optional_arg(command, "--checkpoints_total_limit", args.checkpoints_total_limit)
    _append_optional_arg(command, "--resume_from_checkpoint", args.resume_from_checkpoint)
    _append_optional_arg(command, "--prediction_type", args.prediction_type)
    _append_optional_arg(command, "--infer_negative_prompt", args.infer_negative_prompt)

    if args.allow_tf32:
        command.append("--allow_tf32")
    if args.gradient_checkpointing:
        command.append("--gradient_checkpointing")
    if args.set_grads_to_none:
        command.append("--set_grads_to_none")

    print(f"[*] CUDA_VISIBLE_DEVICES={args.cuda_device}")
    print(f"[*] running full pipeline command: {' '.join(command)}")
    subprocess.run(command, check=True, env=env)


if __name__ == "__main__":
    main()
