import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PYTHON_EXECUTABLE = sys.executable
ROOT_DIR = Path(__file__).resolve().parent
INFER_SCRIPT = ROOT_DIR / "infer_controlnet_batch_same_class.py"
EVAL_SCRIPT = ROOT_DIR / "evaluate_generation_quality.py"

# =========================
# 固定路径（参考 run_infer_new_masks_fixed.py）
# =========================
BASE_MODEL = "/d242/wyh/model/ldm/inpainting/sd-v1-5-inpainting.ckpt"
DATASET_ROOT = "/home/wyh/data/mvtec_ad/MVTEC_AD_512"
MASK_ROOT = "/d242/wyh/M2M/pre_mask/spatial_prior"
PROMPT_DIRNAME = "prompt"
PROMPT4GEN_ROOT = "/d242/wyh/M2M/prompt4generation"
RESULT_ROOT = Path("/d242/wyh/M2M/result")

# 已有两组权重
WITH_IP_FINAL_CONTROLNET_PATH = "/d242/wyh/model/M2M_controlnet/controlnet"
WITH_IP_WEIGHTS_PATH = "/d242/wyh/model/M2M_controlnet/ip_adapter_visual_prompt/pytorch_model.bin"
WITHOUT_IP_FINAL_CONTROLNET_PATH = "/d242/wyh/model/M2M_controlnet_no_ipadapter/controlnet"

# =========================
# 运行配置（固定）
# =========================
CUDA_DEVICE = "1"
SEED = 42
IMAGE_SIZE = 512
NUM_INFERENCE_STEPS = 30
GUIDANCE_SCALE = 7.5
CONTROLNET_CONDITIONING_SCALE = 1.0

# 对称 latent 融合模块开关
ENABLE_SYMMETRIC_FUSION = True
SYMMETRIC_FUSION_STRENGTH = 1.0
SYMMETRIC_FUSION_ALL_STEPS = False
SYMMETRIC_FUSION_LAST_K_STEPS = 9  # 30 steps 的后 30%

# 评估配置（固定为异常类别粒度：object/defect）
EVAL_CLASS_LEVEL = "object_defect"
EVAL_BATCH_SIZE = 32
EVAL_IS_SPLITS = 10
EVAL_LPIPS_BACKBONE = "vgg"
EVAL_ICL_CLUSTER_SIZE = 50


@dataclass
class ExpConfig:
    name: str
    controlnet_path: str
    enable_ip_adapter: bool
    ip_weights_path: str | None = None


def _ensure_exists(path: Path, desc: str):
    if not path.exists():
        raise FileNotFoundError(f"{desc} not found: {path}")


def _build_env(cuda_id: str) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_id
    return env


def _run_cmd(cmd: list[str], title: str, env: dict[str, str] | None = None):
    print(f"\n=== {title} ===")
    print(" ".join(cmd))
    if env is not None:
        print(f"[env] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES')}")
    subprocess.run(cmd, check=True, env=env)


def _build_infer_cmd(exp: ExpConfig, out_dir: Path) -> list[str]:
    cmd = [
        PYTHON_EXECUTABLE,
        str(INFER_SCRIPT),
        "--controlnet_path",
        exp.controlnet_path,
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

    if exp.enable_ip_adapter:
        cmd.append("--enable_ip_adapter_visual_prompt")
        if exp.ip_weights_path is not None:
            cmd.extend(["--ip_adapter_weights_path", exp.ip_weights_path])

    if ENABLE_SYMMETRIC_FUSION:
        cmd.append("--enable_symmetric_latent_fusion")
        cmd.extend(["--symmetric_latent_fusion_strength", str(SYMMETRIC_FUSION_STRENGTH)])
        if SYMMETRIC_FUSION_ALL_STEPS:
            cmd.append("--symmetric_latent_fusion_all_steps")
        else:
            cmd.extend(["--symmetric_latent_fusion_last_k_steps", str(SYMMETRIC_FUSION_LAST_K_STEPS)])

    return cmd


def _build_eval_cmd(gen_root: Path, output_json: Path) -> list[str]:
    return [
        PYTHON_EXECUTABLE,
        str(EVAL_SCRIPT),
        "--gen_root",
        str(gen_root),
        "--dataset_root",
        DATASET_ROOT,
        "--class_level",
        EVAL_CLASS_LEVEL,
        "--batch_size",
        str(EVAL_BATCH_SIZE),
        "--is_splits",
        str(EVAL_IS_SPLITS),
        "--lpips_backbone",
        EVAL_LPIPS_BACKBONE,
        "--ic_l_cluster_size",
        str(EVAL_ICL_CLUSTER_SIZE),
        "--seed",
        str(SEED),
        "--output_json",
        str(output_json),
    ]


def _load_eval_json(path: Path) -> dict[str, dict[str, float | int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, float | int]] = {}
    for row in payload:
        result[row["class"]] = {
            "num_images": int(row["num_images"]),
            "is_mean": float(row["inception_score_mean"]),
            "is_std": float(row["inception_score_std"]),
            "ic_l": float(row["ic_l"]),
        }
    return result


def _fmt_float(x: float) -> str:
    if x != x:  # NaN
        return "nan"
    return f"{x:.4f}"


def _safe_mean(values: list[float]) -> float:
    vals = [v for v in values if v == v]
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _print_side_by_side_table(
    with_ip: dict[str, dict[str, float | int]],
    without_ip: dict[str, dict[str, float | int]],
):
    # 输入 key 固定为 object/defect
    all_classes = sorted(set(with_ip.keys()) | set(without_ip.keys()))

    object_to_classes: dict[str, list[str]] = {}
    for cls in all_classes:
        obj = cls.split("/", 1)[0]
        object_to_classes.setdefault(obj, []).append(cls)

    print("\n=== Comparison Summary (object mean first, then each defect) ===")

    for obj in sorted(object_to_classes.keys()):
        obj_classes = sorted(object_to_classes[obj])

        obj_with_is = [float(with_ip[c]["is_mean"]) for c in obj_classes if c in with_ip]
        obj_without_is = [float(without_ip[c]["is_mean"]) for c in obj_classes if c in without_ip]
        obj_with_icl = [float(with_ip[c]["ic_l"]) for c in obj_classes if c in with_ip]
        obj_without_icl = [float(without_ip[c]["ic_l"]) for c in obj_classes if c in without_ip]

        m_with_is = _safe_mean(obj_with_is)
        m_without_is = _safe_mean(obj_without_is)
        m_with_icl = _safe_mean(obj_with_icl)
        m_without_icl = _safe_mean(obj_without_icl)

        d_is = m_with_is - m_without_is if (m_with_is == m_with_is and m_without_is == m_without_is) else float("nan")
        d_icl = m_with_icl - m_without_icl if (m_with_icl == m_with_icl and m_without_icl == m_without_icl) else float("nan")

        print("\n" + "=" * 120)
        print(f"Object: {obj}")
        print("[Object Mean over Defects]")
        print(
            f"  IS(with)={_fmt_float(m_with_is)}  IS(w/o)={_fmt_float(m_without_is)}  ΔIS={_fmt_float(d_is)}   "
            f"IC-L(with)={_fmt_float(m_with_icl)}  IC-L(w/o)={_fmt_float(m_without_icl)}  ΔIC-L={_fmt_float(d_icl)}"
        )

        print("\n  [Defect Details]")
        print(
            f"  {'defect':<28} {'#img':>6} "
            f"{'IS(with)':>10} {'IS(w/o)':>10} {'ΔIS':>10} "
            f"{'IC-L(with)':>12} {'IC-L(w/o)':>12} {'ΔIC-L':>12}"
        )
        print("  " + "-" * 108)

        for cls in obj_classes:
            defect = cls.split("/", 1)[1] if "/" in cls else cls
            a = with_ip.get(cls)
            b = without_ip.get(cls)

            n = int(a["num_images"] if a is not None else (b["num_images"] if b is not None else 0))
            a_is = float(a["is_mean"]) if a is not None else float("nan")
            b_is = float(b["is_mean"]) if b is not None else float("nan")
            a_icl = float(a["ic_l"]) if a is not None else float("nan")
            b_icl = float(b["ic_l"]) if b is not None else float("nan")

            dd_is = a_is - b_is if (a_is == a_is and b_is == b_is) else float("nan")
            dd_icl = a_icl - b_icl if (a_icl == a_icl and b_icl == b_icl) else float("nan")

            print(
                f"  {defect:<28} {n:>6d} "
                f"{_fmt_float(a_is):>10} {_fmt_float(b_is):>10} {_fmt_float(dd_is):>10} "
                f"{_fmt_float(a_icl):>12} {_fmt_float(b_icl):>12} {_fmt_float(dd_icl):>12}"
            )


def main():
    # 基础检查
    _ensure_exists(INFER_SCRIPT, "infer script")
    _ensure_exists(EVAL_SCRIPT, "evaluation script")
    _ensure_exists(Path(BASE_MODEL), "base model")
    _ensure_exists(Path(DATASET_ROOT), "dataset root")
    _ensure_exists(Path(MASK_ROOT), "mask root")
    _ensure_exists(Path(WITH_IP_FINAL_CONTROLNET_PATH), "with-IP controlnet")
    _ensure_exists(Path(WITH_IP_WEIGHTS_PATH), "with-IP adapter weights")
    _ensure_exists(Path(WITHOUT_IP_FINAL_CONTROLNET_PATH), "without-IP controlnet")

    with_ip_name = "with_ipadapter_epoch50_final_fusion_last30pct"
    without_ip_name = "without_ipadapter_epoch50_final_fusion_last30pct"

    exps = [
        ExpConfig(
            name=with_ip_name,
            controlnet_path=WITH_IP_FINAL_CONTROLNET_PATH,
            enable_ip_adapter=True,
            ip_weights_path=WITH_IP_WEIGHTS_PATH,
        ),
        ExpConfig(
            name=without_ip_name,
            controlnet_path=WITHOUT_IP_FINAL_CONTROLNET_PATH,
            enable_ip_adapter=False,
            ip_weights_path=None,
        ),
    ]

    env = _build_env(CUDA_DEVICE)
    eval_jsons: dict[str, Path] = {}

    for idx, exp in enumerate(exps, start=1):
        out_dir = RESULT_ROOT / exp.name
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n>>> [Stage {idx}/4] Inference: {exp.name}")
        infer_cmd = _build_infer_cmd(exp, out_dir)
        _run_cmd(infer_cmd, f"Inference {exp.name}", env=env)

        print(f"\n>>> [Stage {idx + 2}/4] Evaluation: {exp.name}")
        eval_json = out_dir / f"eval_{EVAL_CLASS_LEVEL}.json"
        eval_cmd = _build_eval_cmd(out_dir, eval_json)
        _run_cmd(eval_cmd, f"Evaluate {exp.name}", env=env)
        eval_jsons[exp.name] = eval_json

    # 汇总比较表
    with_ip_metrics = _load_eval_json(eval_jsons[with_ip_name])
    without_ip_metrics = _load_eval_json(eval_jsons[without_ip_name])
    _print_side_by_side_table(with_ip_metrics, without_ip_metrics)

    print("\n[*] Done: inference + evaluation completed sequentially on CUDA 0.")
    print(f"[*] With-IP result dir: {RESULT_ROOT / with_ip_name}")
    print(f"[*] Without-IP result dir: {RESULT_ROOT / without_ip_name}")


if __name__ == "__main__":
    main()
