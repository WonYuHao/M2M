import os, subprocess, sys
from pathlib import Path

# ========================= 配置区 =========================
CUDA_VISIBLE_DEVICES = "3"
# 模式选择: infer_eval (跳过训练直接推理+评价) / full (全流程) / train_only / eval_only
RUN_MODE = "eval_only" 

PROJECT_ROOT = Path(__file__).resolve().parent
BASE_MODEL = "/d242/wyh/model/ldm/inpainting/sd-v1-5-inpainting.ckpt"
TRAIN_JSON = "/home/wyh/data/mvtec_ad/train_prompts_wiGT_full_with_visual_match.json"
DATASET_ROOT = "/home/wyh/data/mvtec_ad/MVTEC_AD_512"

# 1. 训练输出根目录 (当含有训练步骤时使用)
OUTPUT_DIR = "/d242/wyh/model/M2M_controlnet"

# 2. 外部预训练权重根目录 (当跳过训练时使用，请确保该目录下有 controlnet/ 和 ip_adapter_visual_prompt/ 两个文件夹)
EXTERNAL_WEIGHTS_PATH = "/d242/wyh/model/M2M_controlnet" # 外部指定权重路径 (当没有训练时使用)

# 3. 推理输出与评价配置
INFER_OUT = "/home/wyh/data/mvtec_ad/generated_datasets"
EVAL_METRIC = "ic-l"  # is / ic-l
EVAL_MD = str(PROJECT_ROOT / "outputs" / f"{EVAL_METRIC}_report.md")

# ========================= 自动逻辑计算 =========================

def main():
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": CUDA_VISIBLE_DEVICES}
    
    # 判定路径来源
    has_training = "train" in RUN_MODE or RUN_MODE == "full"
    weight_root = Path(OUTPUT_DIR) if has_training else Path(EXTERNAL_WEIGHTS_PATH)
    
    # 组装 ControlNet 和 IP-Adapter 的具体路径
    # 注意：根据你的 ls 结果，controlnet 是文件夹，ip_adapter 权重是文件夹下的 pytorch_model.bin
    effective_controlnet = str(weight_root / "controlnet")
    effective_ip_adapter = str(weight_root / "ip_adapter_visual_prompt" / "pytorch_model.bin")

    print(f"\n[INFO] 模式: {RUN_MODE}")
    print(f"[INFO] 权重来源: {'训练输出' if has_training else '外部指定'}")
    print(f"[INFO] ControlNet 路径: {effective_controlnet}")
    print(f"[INFO] IP-Adapter 路径: {effective_ip_adapter}\n")

    # 1. 训练参数
    train_args = {
        "base_model": BASE_MODEL,
        "output_dir": OUTPUT_DIR,
        "train_json_path": TRAIN_JSON,
        "enable_ip_adapter_visual_prompt": True,
        "ip_adapter_vision_model": "/d242/wyh/model/clip-vit-large-patch14",
        "num_train_epochs": 50,
        "save_epochs": 10,
    }

    # 2. 推理参数 (infer_controlnet_batch_same_class.py)
    infer_args = {
        "base_model": BASE_MODEL,
        "controlnet_path": effective_controlnet,
        "ip_adapter_weights_path": effective_ip_adapter,
        "output_dir": INFER_OUT,
        "dataset_root": DATASET_ROOT,
        "match_json_path": TRAIN_JSON,
        "mask_root": "/home/wyh/data/mvtec_ad/generated_mask_good",
        "enable_ip_adapter_visual_prompt": True,
        "enable_symmetric_latent_fusion": True,
        "ip_adapter_vision_model": "/d242/wyh/model/clip-vit-large-patch14",
        "num_inference_steps": 30
    }

    # 3. 评价参数 (evaluate_generation_quality.py)
    eval_args = {
        "gen_root": INFER_OUT,
        "dataset_root": DATASET_ROOT,
        "metric": EVAL_METRIC,
        "output_md": EVAL_MD,
        "batch_size": 32
    }

    # 任务流映射
    task_map = {
        "train_only": [(PROJECT_ROOT/"train_controlnet_inpaint_prompt.py", train_args)],
        "infer_only": [(PROJECT_ROOT/"infer_controlnet_batch_same_class.py", infer_args)],
        "eval_only":  [(PROJECT_ROOT/"evaluate_generation_quality.py", eval_args)],
        "infer_eval": [(PROJECT_ROOT/"infer_controlnet_batch_same_class.py", infer_args),
                        (PROJECT_ROOT/"evaluate_generation_quality.py", eval_args)],
        "full":        [(PROJECT_ROOT/"train_controlnet_inpaint_prompt.py", train_args),
                        (PROJECT_ROOT/"infer_controlnet_batch_same_class.py", infer_args),
                        (PROJECT_ROOT/"evaluate_generation_quality.py", eval_args)]
    }

    if RUN_MODE in task_map:
        for script, args in task_map[RUN_MODE]:
            run_cmd(script, args, env)
    else:
        print(f"ERROR: RUN_MODE '{RUN_MODE}' is not defined.")

def run_cmd(script, args_dict, env):
    if not script.exists():
        print(f"ERROR: 脚本 {script} 不存在")
        return
    
    cmd = [sys.executable, str(script)]
    for k, v in args_dict.items():
        if v is True:
            cmd.append(f"--{k}")
        elif v is not False and v is not None:
            cmd.extend([f"--{k}", str(v)])
            
    print(f"\n[RUNNING] {script.name}")
    print("-" * 50)
    subprocess.run(cmd, check=True, env=env)

if __name__ == "__main__":
    main()