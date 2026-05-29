import argparse
import os
from typing import Optional

import cv2
import numpy as np
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
from diffusers.utils import load_image
from PIL import Image


DEFAULT_BASE_MODEL = "/d242/wyh/model/ldm/inpainting/sd-v1-5-inpainting.ckpt"
# 若你手头有更贴近 SD1.5 inpaint 的 ControlNet 权重，可通过 --controlnet_path 指定本地/远程模型
DEFAULT_CONTROLNET = None


def _to_pil_image(path_or_img):
    if isinstance(path_or_img, Image.Image):
        return path_or_img.convert("RGB")
    if isinstance(path_or_img, str):
        return load_image(path_or_img).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(path_or_img)}")


def load_binary_mask(mask_path: str, size: Optional[tuple[int, int]] = None) -> Image.Image:
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    if size is not None:
        mask = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
    mask = (mask > 127).astype(np.uint8) * 255
    return Image.fromarray(mask, mode="L")


def prepare_pipeline(
    base_model: str,
    controlnet_path: Optional[str],
    device: str,
    dtype: torch.dtype,
):
    if controlnet_path:
        controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=dtype)
    else:
        controlnet = None

    pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
        base_model,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe.to(device)
    return pipe


@torch.inference_mode()
def run_inpaint(
    pipe,
    image_path: str,
    mask_path: str,
    prompt: str,
    out_path: str,
    negative_prompt: Optional[str] = None,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    controlnet_conditioning_scale: float = 1.0,
    strength: float = 1.0,
    seed: int = 0,
):
    image = _to_pil_image(image_path)
    mask = load_binary_mask(mask_path, size=image.size)

    generator = None
    if seed and seed > 0:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)

    # mask 作为 control image，严格提供空间先验；inpaint mask 负责真正的局部编辑
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        mask_image=mask,
        control_image=mask.convert("RGB"),
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        strength=strength,
        generator=generator,
    ).images[0]

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    result.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="输入正常图像")
    parser.add_argument("--mask", required=True, help="二值 mask，白色为待编辑区域")
    parser.add_argument("--prompt", required=True, help="文本 prompt")
    parser.add_argument("--out", required=True, help="输出图片路径")
    parser.add_argument(
        "--base_model",
        type=str,
        default=DEFAULT_BASE_MODEL,
        help="Stable Diffusion inpainting 模型名或本地 checkpoint 路径",
    )
    parser.add_argument(
        "--controlnet_path",
        type=str,
        default=DEFAULT_CONTROLNET,
        help="ControlNet 模型名或本地路径；暂时可为空（则只用 inpainting）",
    )
    parser.add_argument("--negative_prompt", type=str, default=None)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--control_scale", type=float, default=1.0)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = prepare_pipeline(args.base_model, args.controlnet_path, device, dtype)
    run_inpaint(
        pipe=pipe,
        image_path=args.image,
        mask_path=args.mask,
        prompt=args.prompt,
        out_path=args.out,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance_scale,
        controlnet_conditioning_scale=args.control_scale,
        strength=args.strength,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
