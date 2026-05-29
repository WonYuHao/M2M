import argparse
import json
import random
import re
import shutil
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from diffusers import ControlNetModel, DDPMScheduler, StableDiffusionInpaintPipeline
from PIL import Image
from torchvision import transforms

from ip_adapter_visual_prompt import IPAdapterVisualPrompt
from latent_symmetric_fusion import SymmetricLatentFusion, SymmetricLatentFusionConfig


DEFAULT_BASE_MODEL = "/d242/wyh/model/ldm/inpainting/sd-v1-5-inpainting.ckpt"
DEFAULT_OUTPUT_DIR = "./logout"

# =========================
# 自动测试模式参数（你只需要改这里）
# =========================
# 对应 predictd_anomaly_mask/<object_category>/<defect_type>/ 下的层级
OBJECT_CATEGORY = None  # 例如 "screw"；或 "none" 表示遍历 mask_root 下所有对象类别
ANOMALY_DEFECT_TYPE = None  # 例如 "scratch_head"；或 "none" 表示遍历该对象下所有异常缺陷子类

DEFAULT_DATASET_ROOT = "/home/wyh/data/mvtec_ad/MVTEC_AD_512"
DEFAULT_MASK_ROOT = Path("/home/wyh/data/mvtec_ad/genmask_mask_good")
DEFAULT_PROMPT_DIRNAME = "prompt"
DEFAULT_PROMPT4GEN_ROOT = "/d242/wyh/M2M/result"
DEFAULT_MATCH_JSON = "/home/wyh/data/mvtec_ad/train_prompts_wiGT_full_with_visual_match.json"

# 仅用于避免一次跑太多：None=不限制。按“每个对象类别/缺陷子类”限制 stem 数量。
MAX_STEMS_PER_DEFECT = None


def _iter_image_paths_for_stem(image_dir: Path, stem: str) -> Iterable[Path]:
    p = image_dir / f"{stem}.png"
    if p.is_file():
        yield p


def _find_first_image_for_stem(image_dir: Path, stem: str) -> Optional[Path]:
    for p in _iter_image_paths_for_stem(image_dir, stem):
        return p
    return None


def _group_masks_by_stem(mask_defect_dir: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[tuple[str, Path]]] = {}
    for p in sorted(mask_defect_dir.glob("*.png")):
        stem = p.stem
        key = stem[:3]
        if len(key) < 3 or not key.isdigit():
            continue
        grouped.setdefault(key, []).append((stem, p))

    return {
        key: [path for _, path in sorted(items, key=lambda item: item[0])]
        for key, items in sorted(grouped.items())
    }


def _load_match_records(match_json_path: str) -> list[dict]:
    p = Path(match_json_path)
    if not p.is_file():
        raise FileNotFoundError(f"match json not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("match json must be list[dict]")
    return data


def _build_category_prompt_to_refs(match_records: list[dict]) -> dict[tuple[str, str], list[dict]]:
    mapping: dict[tuple[str, str], list[dict]] = {}
    for item in match_records:
        if not isinstance(item, dict):
            continue
        info = str(item.get("_info", "")).strip()
        prompt = str(item.get("prompt", "")).strip()
        if not info or not prompt or "/" not in info:
            continue
        obj, defect_type = info.split("/", 1)
        key = (obj, defect_type)
        cands = item.get("visual_match_candidates") or []
        mapping.setdefault(key, [])
        for c in cands:
            if not isinstance(c, dict) or not c.get("image"):
                continue
            mapping[key].append({"image": str(c["image"]), "prompt": prompt, "type": str(c.get("type", "unknown")), "is_self": bool(c.get("is_self", False))})
    for key, vals in list(mapping.items()):
        seen = set()
        uniq = []
        for v in vals:
            k = (v["image"], v["prompt"], v["type"], v["is_self"])
            if k in seen:
                continue
            seen.add(k)
            uniq.append(v)
        mapping[key] = uniq
    return mapping


def _load_runtime_modules(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    inpaint_pipe = StableDiffusionInpaintPipeline.from_single_file(
        args.base_model,
        safety_checker=None,
        requires_safety_checker=False,
        torch_dtype=dtype,
    )
    tokenizer = inpaint_pipe.tokenizer
    text_encoder = inpaint_pipe.text_encoder.to(device=device, dtype=dtype).eval()
    vae = inpaint_pipe.vae.to(device=device, dtype=dtype).eval()
    unet = inpaint_pipe.unet.to(device=device, dtype=dtype).eval()

    noise_scheduler = DDPMScheduler.from_config(inpaint_pipe.scheduler.config)
    noise_scheduler.set_timesteps(args.num_inference_steps, device=device)

    controlnet = ControlNetModel.from_pretrained(args.controlnet_path, torch_dtype=dtype)
    controlnet = controlnet.to(device=device).eval()

    ip_adapter_module = None
    if args.enable_ip_adapter_visual_prompt:
        ip_adapter_module = IPAdapterVisualPrompt(
            vision_model_name_or_path=args.ip_adapter_vision_model,
            cross_attention_dim=text_encoder.config.hidden_size,
            num_queries=args.ip_adapter_num_queries,
            perceiver_depth=args.ip_adapter_perceiver_depth,
            perceiver_heads=args.ip_adapter_perceiver_heads,
            cross_attn_heads=args.ip_adapter_cross_attn_heads,
            scale=args.ip_adapter_scale,
        ).to(device=device, dtype=dtype)

        if args.ip_adapter_weights_path is not None:
            ip_weights = Path(args.ip_adapter_weights_path)
        else:
            ip_weights = Path(args.controlnet_path).resolve().parent / "ip_adapter_visual_prompt" / "pytorch_model.bin"

        if not ip_weights.is_file():
            raise FileNotFoundError(f"IP-Adapter enabled but weights not found: {ip_weights}")

        state_dict = torch.load(ip_weights, map_location="cpu")
        ip_adapter_module.load_state_dict(state_dict, strict=True)
        ip_adapter_module.visual_encoder.requires_grad_(False)
        ip_adapter_module.eval()

    image_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    mask_tf = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
    ])

    return {
        "device": device,
        "dtype": dtype,
        "tokenizer": tokenizer,
        "text_encoder": text_encoder,
        "vae": vae,
        "unet": unet,
        "noise_scheduler": noise_scheduler,
        "timesteps": noise_scheduler.timesteps,
        "controlnet": controlnet,
        "ip_adapter_module": ip_adapter_module,
        "image_tf": image_tf,
        "mask_tf": mask_tf,
    }


def _to_tensor_rgb(image: Image.Image, image_tf, device: str, dtype: torch.dtype) -> torch.Tensor:
    return image_tf(image).unsqueeze(0).to(device=device, dtype=dtype)


def _encode_prompt_with_optional_visual(*, args, runtime, cond_prompt: str, ref_image_cond_t: Optional[torch.Tensor], ref_image_uncond_t: Optional[torch.Tensor]) -> torch.Tensor:
    tokenizer = runtime["tokenizer"]
    text_encoder = runtime["text_encoder"]
    ip_adapter_module = runtime["ip_adapter_module"]
    device = runtime["device"]
    dtype = runtime["dtype"]

    do_cfg = args.guidance_scale is not None and float(args.guidance_scale) > 1.0
    neg_prompt = args.negative_prompt if args.negative_prompt is not None else ""

    tokens = tokenizer([neg_prompt, cond_prompt] if do_cfg else [cond_prompt], max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
    input_ids = tokens.input_ids.to(device)
    encoder_hidden_states = text_encoder(input_ids)[0].to(dtype=dtype)

    if ip_adapter_module is None:
        return encoder_hidden_states

    if do_cfg:
        uncond_hidden = encoder_hidden_states[0:1]
        cond_hidden = encoder_hidden_states[1:2]
        if ref_image_uncond_t is None:
            ref_image_uncond_t = ref_image_cond_t
        if ref_image_uncond_t is None or ref_image_cond_t is None:
            raise ValueError("IP-Adapter enabled but no reference image provided.")

        uncond_out = ip_adapter_module(ref_images=ref_image_uncond_t, text_hidden_states=uncond_hidden, ref_exists=torch.tensor([0.0 if args.disable_ip_adapter_on_uncond else 1.0], device=device))
        cond_out = ip_adapter_module(ref_images=ref_image_cond_t, text_hidden_states=cond_hidden, ref_exists=torch.tensor([1.0], device=device))
        return torch.cat([uncond_out.encoder_hidden_states, cond_out.encoder_hidden_states], dim=0)

    if ref_image_cond_t is None:
        raise ValueError("IP-Adapter enabled but no cond reference image provided.")
    out = ip_adapter_module(ref_images=ref_image_cond_t, text_hidden_states=encoder_hidden_states, ref_exists=torch.tensor([1.0], device=device))
    return out.encoder_hidden_states


def _save_single_outputs(*, out_root: Path, category: str, defect_type: str, stem: str, good_img: Image.Image, mask_img: Image.Image, pred_img: Image.Image, prompt_text: str):
    gen_dir = out_root / category / "generated_datasets" / defect_type
    mask_dir = out_root / category / "mask" / defect_type
    prompt_dir = out_root / category / "text_prompt" / defect_type
    gen_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    prompt_dir.mkdir(parents=True, exist_ok=True)

    pred_img.save(gen_dir / f"{stem}.png")
    mask_img.save(mask_dir / f"{stem}_mask.png")
    (prompt_dir / f"{stem}.txt").write_text(prompt_text, encoding="utf-8")


def _sample_once(*, args, runtime, good_img: Image.Image, mask_img: Image.Image, prompt_text: str, seed: int, ref_img_cond: Optional[Image.Image] = None, ref_img_uncond: Optional[Image.Image] = None) -> Image.Image:
    device = runtime["device"]
    dtype = runtime["dtype"]
    image_tf = runtime["image_tf"]
    mask_tf = runtime["mask_tf"]
    vae = runtime["vae"]
    unet = runtime["unet"]
    controlnet = runtime["controlnet"]
    noise_scheduler = runtime["noise_scheduler"]
    timesteps = runtime["timesteps"]

    generator = torch.Generator(device=device).manual_seed(seed)
    base_t = _to_tensor_rgb(good_img, image_tf, device, dtype)
    mask_t = (mask_tf(mask_img).unsqueeze(0).to(device=device, dtype=dtype) > 0.5).to(dtype=dtype)
    control_image_t = mask_t.repeat(1, 3, 1, 1)
    noise_px = torch.randn(base_t.shape, device=device, dtype=dtype, generator=generator)
    masked_image_t = base_t * (1.0 - mask_t) + noise_px * mask_t
    ref_cond_t = _to_tensor_rgb(ref_img_cond, image_tf, device, dtype) if ref_img_cond is not None else None
    ref_uncond_t = _to_tensor_rgb(ref_img_uncond, image_tf, device, dtype) if ref_img_uncond is not None else None

    with torch.no_grad():
        clean_latents = vae.encode(base_t).latent_dist.sample() * vae.config.scaling_factor
        masked_latents = vae.encode(masked_image_t).latent_dist.sample() * vae.config.scaling_factor
        latents = torch.randn((1, vae.config.latent_channels, args.image_size // 8, args.image_size // 8), device=device, dtype=dtype, generator=generator)
        latent_mask = F.interpolate(mask_t, size=latents.shape[-2:], mode="nearest")
        latent_mask_for_blend = latent_mask.repeat(1, latents.shape[1], 1, 1)
        noise_lat = torch.randn(masked_latents.shape, device=device, dtype=dtype, generator=generator)
        masked_latents = masked_latents * (1.0 - latent_mask_for_blend) + noise_lat * latent_mask_for_blend
        encoder_hidden_states = _encode_prompt_with_optional_visual(args=args, runtime=runtime, cond_prompt=prompt_text, ref_image_cond_t=ref_cond_t, ref_image_uncond_t=ref_uncond_t)
        do_cfg = args.guidance_scale is not None and float(args.guidance_scale) > 1.0
        fusion = SymmetricLatentFusion(SymmetricLatentFusionConfig(enabled=args.enable_symmetric_latent_fusion, strength=float(args.symmetric_latent_fusion_strength), all_steps=bool(args.symmetric_latent_fusion_all_steps), last_k_steps=int(args.symmetric_latent_fusion_last_k_steps)))
        reference_noise = latents.detach().clone()

        for step_index, t in enumerate(timesteps):
            if do_cfg:
                latents_in = torch.cat([latents] * 2, dim=0)
                latent_mask_in = torch.cat([latent_mask] * 2, dim=0)
                masked_latents_in = torch.cat([masked_latents] * 2, dim=0)
                control_image_in = torch.cat([control_image_t] * 2, dim=0)
                encoder_hidden_states_in = encoder_hidden_states
            else:
                latents_in = latents
                latent_mask_in = latent_mask
                masked_latents_in = masked_latents
                control_image_in = control_image_t
                encoder_hidden_states_in = encoder_hidden_states

            latent_model_input = torch.cat([latents_in, latent_mask_in, masked_latents_in], dim=1)
            latent_model_input = noise_scheduler.scale_model_input(latent_model_input, t)
            down_block_res_samples, mid_block_res_sample = controlnet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states_in, controlnet_cond=control_image_in, conditioning_scale=args.controlnet_conditioning_scale, return_dict=False)
            model_pred = unet(latent_model_input, t, encoder_hidden_states=encoder_hidden_states_in, down_block_additional_residuals=down_block_res_samples, mid_block_additional_residual=mid_block_res_sample, return_dict=False)[0]
            if do_cfg:
                model_pred_uncond, model_pred_text = model_pred.chunk(2)
                model_pred = model_pred_uncond + float(args.guidance_scale) * (model_pred_text - model_pred_uncond)
            latents = noise_scheduler.step(model_pred, t, latents, return_dict=False)[0]
            latents = fusion.apply(pred_latents=latents, ref_clean_latents=clean_latents, latent_mask=latent_mask, scheduler=noise_scheduler, timesteps=timesteps, step_index=step_index, reference_noise=reference_noise)

        image = vae.decode(latents / vae.config.scaling_factor).sample
        image = (image / 2.0 + 0.5).clamp(0, 1)
        image_u8 = (image[0].permute(1, 2, 0).detach().cpu().numpy() * 255.0).round().astype("uint8")
        return Image.fromarray(image_u8)


def run_batch_infer_auto(args):
    mask_root = Path(args.mask_root)
    dataset_root = Path(args.dataset_root)
    if not mask_root.is_dir():
        raise FileNotFoundError(str(mask_root))
    if not dataset_root.is_dir():
        raise FileNotFoundError(str(dataset_root))

    object_dirs = [p for p in mask_root.iterdir() if p.is_dir()] if str(OBJECT_CATEGORY).lower() == "none" else [mask_root / str(OBJECT_CATEGORY)]
    object_dirs = sorted([p for p in object_dirs if p.is_dir()])
    if not object_dirs:
        raise RuntimeError(f"No object dirs found under {mask_root}")

    if str(ANOMALY_DEFECT_TYPE).lower() == "none":
        defect_types_by_object = {od.name: sorted([p.name for p in od.iterdir() if p.is_dir()]) for od in object_dirs}
    else:
        defect_types_by_object = {od.name: [str(ANOMALY_DEFECT_TYPE)] for od in object_dirs}

    runtime = _load_runtime_modules(args)
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    match_records = _load_match_records(args.match_json_path)
    prompt_to_ref = _build_category_prompt_to_refs(match_records)

    prompt_cache: dict[tuple[str, str], list[str]] = {}
    prompt_to_ref_cache: dict[tuple[str, str], list[dict]] = {}

    for obj_dir in object_dirs:
        obj = obj_dir.name
        good_dir = dataset_root / obj / "train" / "good"
        if not good_dir.is_dir():
            print(f"[skip] {obj}: missing {good_dir}")
            continue

        for defect_type in defect_types_by_object.get(obj, []):
            mask_defect_dir = obj_dir / defect_type
            if not mask_defect_dir.is_dir():
                print(f"[skip] {obj}/{defect_type}: missing mask dir {mask_defect_dir}")
                continue

            mask_paths_by_stem = _group_masks_by_stem(mask_defect_dir)
            stems = list(mask_paths_by_stem.keys())
            if not stems:
                print(f"[skip] {obj}/{defect_type}: no stems found")
                continue
            if MAX_STEMS_PER_DEFECT is not None:
                stems = stems[:MAX_STEMS_PER_DEFECT]

            prompt_key = (obj, defect_type)
            if prompt_key not in prompt_cache:
                prompt_candidates = [
                    dataset_root / obj / args.prompt_dirname / defect_type,
                    Path(args.prompt4generation_root) / obj / defect_type,
                ]
                prompt_dir = next((p for p in prompt_candidates if p.is_dir()), None)
                if prompt_dir is None:
                    raise FileNotFoundError(
                        f"Prompt dir not found. Tried: {', '.join(str(p) for p in prompt_candidates)}"
                    )
                prompts = [tp.read_text(encoding="utf-8").strip() for tp in sorted(prompt_dir.glob("*.txt")) if tp.is_file()]
                prompts = [p for p in prompts if p]
                if not prompts:
                    raise RuntimeError(f"No prompts found under: {prompt_dir}")
                prompt_cache[prompt_key] = prompts

            for stem in stems:
                good_path = _find_first_image_for_stem(good_dir, stem)
                if good_path is None:
                    print(f"[skip] missing good image for stem={stem} in {good_dir}")
                    continue
                good_img = Image.open(good_path).convert("RGB").resize((args.image_size, args.image_size))
                stem_mask_paths = mask_paths_by_stem.get(stem, [])
                if not stem_mask_paths:
                    print(f"[skip] missing mask for stem={stem}")
                    continue

                prompt_candidates = prompt_cache[prompt_key]
                prompt_for_image = prompt_candidates[0]

                ref_img_cond = None
                if args.enable_ip_adapter_visual_prompt:
                    p2r_key = (obj, defect_type)
                    if p2r_key not in prompt_to_ref_cache:
                        prompt_to_ref_cache[p2r_key] = prompt_to_ref.get(p2r_key, [])
                    ref_candidates = prompt_to_ref_cache[p2r_key]
                    if ref_candidates:
                        ref_choice = rng.choice(ref_candidates)
                        ref_path = Path(ref_choice["image"])
                        prompt_for_image = str(ref_choice.get("prompt", prompt_for_image)).strip() or prompt_for_image
                        if ref_path.is_file():
                            ref_img_cond = Image.open(ref_path).convert("RGB").resize((args.image_size, args.image_size))
                    if ref_img_cond is None:
                        defect_img_dir = dataset_root / obj / "train" / defect_type
                        ref_pool = sorted([p for p in defect_img_dir.glob("*.png") if p.is_file()]) if defect_img_dir.is_dir() else []
                        ref_img_cond = Image.open(rng.choice(ref_pool)).convert("RGB").resize((args.image_size, args.image_size)) if ref_pool else good_img

                for mask_idx, mask_path in enumerate(stem_mask_paths):
                    mask_img = Image.open(mask_path).convert("L").resize((args.image_size, args.image_size))
                    out_stem = mask_path.stem
                    prompt_save_dir = out_root / obj / "text_prompt" / defect_type
                    prompt_save_dir.mkdir(parents=True, exist_ok=True)
                    (prompt_save_dir / f"{out_stem}.txt").write_text(prompt_for_image, encoding="utf-8")

                    pred_img = _sample_once(
                        args=args,
                        runtime=runtime,
                        good_img=good_img,
                        mask_img=mask_img,
                        prompt_text=prompt_for_image,
                        seed=args.seed + int(stem) * 1000 + mask_idx,
                        ref_img_cond=ref_img_cond,
                        ref_img_uncond=ref_img_cond,
                    )
                    _save_single_outputs(
                        out_root=out_root,
                        category=obj,
                        defect_type=defect_type,
                        stem=out_stem,
                        good_img=good_img,
                        mask_img=mask_img,
                        pred_img=pred_img,
                        prompt_text=prompt_for_image,
                    )
                    print(f"[saved] {obj}/{defect_type}/{out_stem}.png")


def run_batch_infer_manual(args):
    raise NotImplementedError("manual mode is not used in this pipeline")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controlnet_path", required=True)
    parser.add_argument("--base_model", type=str, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--controlnet_conditioning_scale", type=float, default=1.0)
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_root", type=str, default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--mask_root", type=str, default=str(DEFAULT_MASK_ROOT))
    parser.add_argument("--prompt_dirname", type=str, default=DEFAULT_PROMPT_DIRNAME)
    parser.add_argument("--prompt4generation_root", type=str, default=DEFAULT_PROMPT4GEN_ROOT)
    parser.add_argument("--match_json_path", type=str, default=DEFAULT_MATCH_JSON)
    parser.add_argument("--enable_ip_adapter_visual_prompt", action="store_true")
    parser.add_argument("--ip_adapter_weights_path", type=str, default=None)
    parser.add_argument("--ip_adapter_vision_model", type=str, default="/d242/wyh/model/clip-vit-large-patch14")
    parser.add_argument("--ip_adapter_num_queries", type=int, default=16)
    parser.add_argument("--ip_adapter_perceiver_depth", type=int, default=2)
    parser.add_argument("--ip_adapter_perceiver_heads", type=int, default=8)
    parser.add_argument("--ip_adapter_cross_attn_heads", type=int, default=8)
    parser.add_argument("--ip_adapter_scale", type=float, default=1.0)
    parser.add_argument("--disable_ip_adapter_on_uncond", action="store_true")
    parser.add_argument("--enable_symmetric_latent_fusion", action="store_true")
    parser.add_argument("--symmetric_latent_fusion_strength", type=float, default=1.0)
    parser.add_argument("--symmetric_latent_fusion_all_steps", action="store_true")
    parser.add_argument("--symmetric_latent_fusion_last_k_steps", type=int, default=0)
    args = parser.parse_args()

    run_batch_infer_auto(args)


if __name__ == "__main__":
    main()
