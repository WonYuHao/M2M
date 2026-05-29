from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class SymmetricLatentFusionConfig:
    enabled: bool = False
    strength: float = 1.0
    all_steps: bool = True
    last_k_steps: int = 0


class SymmetricLatentFusion:
    """Blend latents in a Blended-Latent-Diffusion-like manner.

    Mask convention:
    - mask == 1: anomaly region (allow generation, keep current prediction)
    - mask == 0: background region (keep reference/noised background)

    Fusion rule:
        fused = (1 - m) * q(x0_ref, t_ref) + m * pred

    where q(x0_ref, t_ref) is built by scheduler.add_noise.

    Step scheduling:
    - all_steps=True: apply fusion at every denoising step.
    - all_steps=False and last_k_steps>0: apply only on the last k denoising steps.
    """

    def __init__(self, config: SymmetricLatentFusionConfig):
        self.config = config

    def _should_apply_at_step(self, *, step_index: int, total_steps: int) -> bool:
        if self.config.all_steps:
            return True

        k = int(self.config.last_k_steps)
        if k <= 0:
            # Backward-compatible fallback: if no valid k is provided,
            # still apply on all steps.
            return True

        start_index = max(0, total_steps - k)
        return step_index >= start_index

    def apply(
        self,
        *,
        pred_latents: torch.Tensor,
        ref_clean_latents: torch.Tensor,
        latent_mask: torch.Tensor,
        scheduler,
        timesteps: torch.Tensor,
        step_index: int,
        reference_noise: torch.Tensor,
    ) -> torch.Tensor:
        if not self.config.enabled:
            return pred_latents

        total_steps = len(timesteps)
        if not (0 <= step_index < total_steps):
            return pred_latents

        if not self._should_apply_at_step(step_index=step_index, total_steps=total_steps):
            return pred_latents

        # IMPORTANT:
        # Use the *current* timestep for reference noising so the blended background
        # stays on the same noise level as the predicted latents at this step.
        #
        # Using a mirrored/symmetric timestep here can inject very high noise during
        # late denoising steps (especially when only applying on last_k_steps), which
        # makes the protected background collapse into noise.
        t_ref = timesteps[step_index]

        # Build reference latent at current noising step.
        ref_sym = scheduler.add_noise(ref_clean_latents, reference_noise, t_ref)

        mask = latent_mask
        if mask.shape[1] == 1 and pred_latents.shape[1] != 1:
            mask = mask.repeat(1, pred_latents.shape[1], 1, 1)

        fused = (1.0 - mask) * ref_sym + mask * pred_latents

        if self.config.strength >= 1.0:
            return fused
        if self.config.strength <= 0.0:
            return pred_latents
        return pred_latents + self.config.strength * (fused - pred_latents)
