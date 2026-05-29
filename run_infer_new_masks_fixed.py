import os
import subprocess
import sys
from pathlib import Path


# =========================
# 硬编码配置（全部写死）
# =========================
PYTHON_EXECUTABLE = sys.executable
ROOT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = ROOT_DIR / "train_controlnet_inpaint_prompt.py"
INFER_SCRIPT = ROOT_DIR / "infer_controlnet_batch_same_class.py"

# 通用数据与模型
BASE_MODEL = "/d242/wyh/model/ldm/inpainting/sd-v1-5-inpainting.ckpt"
TRAIN_JSON_PATH = "/home/wyh/data/mvtec_ad/train_prompts_wiGT_full_with_visual_match.json"
DATASET_ROOT = "/home/wyh/data/mvtec_ad/MVTEC_AD_512"
MASK_ROOT = "/d242/wyh/M2M/pre_mask/spatial_prior"
PROMPT_DIRNAME = "prompt"
PROMPT4GEN_ROOT = "/d242/wyh/M2M/prompt4generation"
RESULT_ROOT = "/d242/wyh/M2M/result"

# 新训练（关闭 IPAdapterVisualPrompt）
NO_IP_TRAIN_OUTPUT_DIR = "/d242/wyh/model/M2M_controlnet_no_ipadapter"
NO_IP_FINAL_CONTROLNET_PATH = "/d242/wyh/model/M2M_controlnet_no_ipadapter/controlnet"

# 已有训练（开启 IPAdapterVisualPrompt，50 epoch 最终结果）
WITH_IP_FINAL_CONTROLNET_PATH = "/d242/wyh/model/M2M_controlnet/controlnet"
WITH_IP_WEIGHTS_PATH = "/d242/wyh/model/M2M_controlnet/ip_adapter_visual_prompt/pytorch_model.bin"

# 设备分配（固定）
# CUDA_FOR_TRAIN_AND_NO_IP_INFER = "0"
# CUDA_FOR_WITH_IP_INFER = "1"
CUDA_DEVICE = "1"

# 训练超参数（固定）
NUM_TRAIN_EPOCHS = 50
BATCH_SIZE = 1
LEARNING_RATE = 1e-5
MIXED_PRECISION = "fp16"
SAVE_EPOCHS = 10
SEED = 42

# 推理超参数（固定）
IMAGE_SIZE = 512
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
CONTROLNET_CONDITIONING_SCALE = 1.0


def _ensure_exists(path: Path, path_desc: str):
    if not path.exists():
        raise FileNotFoundError(f"{path_desc} not found: {path}")


def _build_env_for_cuda(cuda_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_id
    return env


def _run_cmd(cmd: list[str], title: str, env: dict[str, str] | None = None):
    print(f"\n=== {title} ===")
    print(" ".join(cmd))
    if env is not None:
        print(f"[env] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}")
    subprocess.run(cmd, check=True, env=env)


def _spawn_cmd(cmd: list[str], title: str, env: dict[str, str] | None = None) -> subprocess.Popen:
    print(f"\n=== Spawn {title} ===")
    print(" ".join(cmd))
    if env is not None:
        print(f"[env] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}")
    return subprocess.Popen(cmd, env=env)


def _train_without_ip_adapter(cuda_id: str):
    out_dir = Path(NO_IP_TRAIN_OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_EXECUTABLE,
        str(TRAIN_SCRIPT),
        "--base_model",
        BASE_MODEL,
        "--train_json_path",
        TRAIN_JSON_PATH,
        "--output_dir",
        str(out_dir),
        "--num_train_epochs",
        str(NUM_TRAIN_EPOCHS),
        "--batch_size",
        str(BATCH_SIZE),
        "--learning_rate",
        str(LEARNING_RATE),
        "--mixed_precision",
        MIXED_PRECISION,
        "--save_epochs",
        str(SAVE_EPOCHS),
        "--seed",
        str(SEED),
        # 关键：不传 --enable_ip_adapter_visual_prompt，即关闭 IPAdapterVisualPrompt
    ]
    _run_cmd(
        cmd,
        "Train 50 epochs WITHOUT IPAdapterVisualPrompt",
        env=_build_env_for_cuda(cuda_id),
    )


def _infer_one(
    *,
    name: str,
    controlnet_path: str,
    enable_ip_adapter: bool,
    cuda_id: str,
    ip_weights_path: str | None = None,
):
    out_dir = Path(RESULT_ROOT) / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_EXECUTABLE,
        str(INFER_SCRIPT),
        "--controlnet_path",
        controlnet_path,
        "--base_model",
        BASE_MODEL,
        "--dataset_root",
        DATASET_ROOT,
        "--mask_root",
        MASK_ROOT,
        "--prompt_dirname",
        PROMPT_DIRNAME,
        "--prompt4generation_root",
        PROMPT4GEN_ROOT,
        "--output_dir",
        str(out_dir),
        "--image_size",
        str(IMAGE_SIZE),
        "--num_inference_steps",
        str(NUM_INFERENCE_STEPS),
        "--guidance_scale",
        str(GUIDANCE_SCALE),
        "--controlnet_conditioning_scale",
        str(CONTROLNET_CONDITIONING_SCALE),
        "--seed",
        str(SEED),
    ]

    if enable_ip_adapter:
        cmd.append("--enable_ip_adapter_visual_prompt")
        if ip_weights_path is not None:
            cmd.extend(["--ip_adapter_weights_path", ip_weights_path])

    _run_cmd(cmd, f"Inference: {name}", env=_build_env_for_cuda(cuda_id))


def _spawn_with_ip_infer_on_cuda1() -> subprocess.Popen:
    out_dir = Path(RESULT_ROOT) / "with_ipadapter_epoch50_final"
    out_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        PYTHON_EXECUTABLE,
        str(INFER_SCRIPT),
        "--controlnet_path",
        WITH_IP_FINAL_CONTROLNET_PATH,
        "--base_model",
        BASE_MODEL,
        "--dataset_root",
        DATASET_ROOT,
        "--mask_root",
        MASK_ROOT,
        "--prompt_dirname",
        PROMPT_DIRNAME,
        "--prompt4generation_root",
        PROMPT4GEN_ROOT,
        "--output_dir",
        str(out_dir),
        "--image_size",
        str(IMAGE_SIZE),
        "--num_inference_steps",
        str(NUM_INFERENCE_STEPS),
        "--guidance_scale",
        str(GUIDANCE_SCALE),
        "--controlnet_conditioning_scale",
        str(CONTROLNET_CONDITIONING_SCALE),
        "--seed",
        str(SEED),
        "--enable_ip_adapter_visual_prompt",
        "--ip_adapter_weights_path",
        WITH_IP_WEIGHTS_PATH,
    ]
    return _spawn_cmd(
        cmd,
        "Inference: with_ipadapter_epoch50_final",
        env=_build_env_for_cuda(CUDA_FOR_WITH_IP_INFER),
    )


def _wait_process(proc: subprocess.Popen, name: str):
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"{name} failed with exit code {ret}")


def main():
    # 1) 基础检查
    _ensure_exists(TRAIN_SCRIPT, "train script")
    _ensure_exists(INFER_SCRIPT, "infer script")
    _ensure_exists(Path(BASE_MODEL), "base model")
    _ensure_exists(Path(TRAIN_JSON_PATH), "train json")
    _ensure_exists(Path(DATASET_ROOT), "dataset root")
    _ensure_exists(Path(MASK_ROOT), "mask root")
    _ensure_exists(Path(WITH_IP_FINAL_CONTROLNET_PATH), "existing with-IP final controlnet")
    _ensure_exists(Path(WITH_IP_WEIGHTS_PATH), "existing with-IP adapter weights")

    with_ip_proc = None
    try:
        # 任务 1: 执行已有“开启 IP-Adapter”模型的推理
        print("\n>>> [Step 1/3] Running Inference with existing IP-Adapter model...")
        _infer_one(
            name="with_ipadapter_epoch50_final",
            controlnet_path=WITH_IP_FINAL_CONTROLNET_PATH,
            enable_ip_adapter=True,
            cuda_id=CUDA_DEVICE,
            ip_weights_path=WITH_IP_WEIGHTS_PATH,
        )

        # 任务 2: 训练“关闭 IP-Adapter”的新模型
        print("\n>>> [Step 2/3] Starting Training WITHOUT IP-Adapter...")
        _train_without_ip_adapter(cuda_id=CUDA_DEVICE)

        # 任务 3: 对刚训练好的“关闭 IP-Adapter”模型进行推理
        print("\n>>> [Step 3/3] Running Inference for the newly trained model...")
        _ensure_exists(Path(NO_IP_FINAL_CONTROLNET_PATH), "trained no-IP final controlnet")
        _infer_one(
            name="without_ipadapter_epoch50_final",
            controlnet_path=NO_IP_FINAL_CONTROLNET_PATH,
            enable_ip_adapter=False,
            cuda_id=CUDA_DEVICE,
        )

    except Exception as e:
        print(f"\n[!] An error occurred during the sequence: {e}")
        sys.exit(1)

    print("\n[*] Done: All tasks completed sequentially on CUDA 1.")

if __name__ == "__main__":
    main()
