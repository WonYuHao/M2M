from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPVisionConfig


class PerceiverAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int = 8, ff_mult: int = 4):
        super().__init__()
        self.norm_latents = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

    def forward(self, latents: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        latents_norm = self.norm_latents(latents)
        context_norm = self.norm_context(context)
        attended, _ = self.attn(latents_norm, context_norm, context_norm, need_weights=False)
        latents = latents + attended
        latents = latents + self.ff(latents)
        return latents


class PerceiverResampler(nn.Module):
    """
    将视觉 encoder 输出的 patch token 压缩成固定数量的感知 token。
    """

    def __init__(self, dim: int, num_queries: int = 16, depth: int = 2, heads: int = 8):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.blocks = nn.ModuleList(
            [PerceiverAttentionBlock(dim=dim, heads=heads) for _ in range(depth)]
        )
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, context_tokens: torch.Tensor) -> torch.Tensor:
        bsz = context_tokens.shape[0]
        latents = self.latents.expand(bsz, -1, -1)
        for block in self.blocks:
            latents = block(latents, context_tokens)
        return self.out_norm(latents)


@dataclass
class VisualPromptOutput:
    encoder_hidden_states: torch.Tensor
    visual_tokens: torch.Tensor


class IPAdapterVisualPrompt(nn.Module):
    """
    IP-Adapter 风格视觉 prompt 模块：
    - 冻结 CLIP Vision Encoder
    - 可训练 Perceiver Resampler
    - 可训练跨注意力，将视觉异常局部特征注入文本条件
    """

    def __init__(
        self,
        *,
        vision_model_name_or_path: str,
        cross_attention_dim: int,
        num_queries: int = 16,
        perceiver_depth: int = 2,
        perceiver_heads: int = 8,
        cross_attn_heads: int = 8,
        scale: float = 1.0,
    ):
        super().__init__()
        self.scale = scale

        self.visual_encoder = CLIPVisionModel.from_pretrained(vision_model_name_or_path)
        self.visual_encoder.requires_grad_(False)
        self.visual_encoder.eval()

        vision_hidden = self.visual_encoder.config.hidden_size
        self.vision_to_cross = nn.Linear(vision_hidden, cross_attention_dim)
        self.resampler = PerceiverResampler(
            dim=cross_attention_dim,
            num_queries=num_queries,
            depth=perceiver_depth,
            heads=perceiver_heads,
        )

        self.text_norm = nn.LayerNorm(cross_attention_dim)
        self.visual_norm = nn.LayerNorm(cross_attention_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=cross_attention_dim,
            num_heads=cross_attn_heads,
            batch_first=True,
        )
        self.out_proj = nn.Linear(cross_attention_dim, cross_attention_dim)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @property
    def device(self):
        return next(self.parameters()).device

    def _prepare_vision_inputs(self, image_tensor: torch.Tensor) -> torch.Tensor:
        # 输入默认来自训练图像 transform: [-1, 1]，转换回 [0, 1] 供 CLIP Vision 使用
        image_tensor = (image_tensor + 1.0) / 2.0
        image_tensor = image_tensor.clamp(0.0, 1.0)
        target_size = int(self.visual_encoder.config.image_size)
        if image_tensor.shape[-1] != target_size or image_tensor.shape[-2] != target_size:
            image_tensor = torch.nn.functional.interpolate(
                image_tensor,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )
        return image_tensor

    def forward(
        self,
        *,
        ref_images: torch.Tensor,
        text_hidden_states: torch.Tensor,
        ref_exists: Optional[torch.Tensor] = None,
    ) -> VisualPromptOutput:
        vision_inputs = self._prepare_vision_inputs(ref_images)

        with torch.no_grad():
            visual_features = self.visual_encoder(pixel_values=vision_inputs).last_hidden_state

        visual_tokens = self.vision_to_cross(visual_features)
        visual_tokens = self.resampler(visual_tokens)

        q = self.text_norm(text_hidden_states)
        kv = self.visual_norm(visual_tokens)
        fused, _ = self.cross_attn(q, kv, kv, need_weights=False)
        fused = self.out_proj(fused)

        if ref_exists is not None:
            # ref_exists: [B], 1 有视觉 prompt，0 无视觉 prompt
            mask = ref_exists.view(-1, 1, 1).to(fused.dtype)
            fused = fused * mask
            visual_tokens = visual_tokens * mask

        enhanced_hidden = text_hidden_states + self.scale * fused
        return VisualPromptOutput(encoder_hidden_states=enhanced_hidden, visual_tokens=visual_tokens)


def infer_vision_hidden_size(vision_model_name_or_path: str) -> int:
    cfg = CLIPVisionConfig.from_pretrained(vision_model_name_or_path)
    return int(cfg.hidden_size)
