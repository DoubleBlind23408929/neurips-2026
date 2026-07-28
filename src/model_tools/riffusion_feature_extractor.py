import typing as tp
import torch
import torch.nn as nn

from diffusers import DiffusionPipeline, DDPMScheduler
from diffusers.models import UNet2DConditionModel


class RiffusionLatentFeatureExtractor(nn.Module):
    """
    Input x: (B, N, 4, h, w)  -- SD latents, already scaled by vae_scaling_factor (0.18215)
    Output: dict { "<basename>_layer_<id>": (B, N, C, h', w') }

    Noise schedule:
      Use DDPMScheduler built from pipeline scheduler config
      and call add_noise(latents, noise, timesteps). This matches SD training forward diffusion.
    """

    def __init__(
        self,
        model_id: str = "riffusion/riffusion-model-v1",
        prompt: str = "",
        torch_dtype: torch.dtype = torch.float16,
        default_timestep: int | None = None
    ) -> None:
        super().__init__()

        pipe = DiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )

        self.unet: UNet2DConditionModel = pipe.unet
        self.tokenizer = getattr(pipe, "tokenizer", None)
        self.text_encoder = getattr(pipe, "text_encoder", None)

        # training-style forward diffusion scheduler
        self.noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

        self._prompt = prompt
        self._text_embeds: torch.Tensor | None = None
        self.default_timestep = default_timestep

        # hook points
        self._hook_modules: list[nn.Module] = self._build_hook_modules(self.unet)
        self._hooks: list[tp.Any] = []
        self._acts: list[torch.Tensor | None] = [None for _ in range(len(self._hook_modules))]

        for i, m in enumerate(self._hook_modules):
            self._hooks.append(m.register_forward_hook(self._make_hook(i)))

        self.eval()

    def _make_hook(self, idx: int):
        def _hook(module, inputs, output):
            out0 = output[0] if isinstance(output, (tuple, list)) else output
            self._acts[idx] = out0
        return _hook

    @staticmethod
    def _build_hook_modules(unet: UNet2DConditionModel) -> list[nn.Module]:
        mods: list[nn.Module] = []

        for db in getattr(unet, "down_blocks", []):
            for r in getattr(db, "resnets", []):
                mods.append(r)
            for a in getattr(db, "attentions", []):
                mods.append(a)

        mb = getattr(unet, "mid_block", None)
        if mb is not None:
            for r in getattr(mb, "resnets", []):
                mods.append(r)
            for a in getattr(mb, "attentions", []):
                mods.append(a)

        for ub in getattr(unet, "up_blocks", []):
            for r in getattr(ub, "resnets", []):
                mods.append(r)
            for a in getattr(ub, "attentions", []):
                mods.append(a)

        if not mods:
            raise RuntimeError("No hook modules found in UNet (unexpected).")
        return mods

    def _get_text_embeds(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._text_embeds is not None and self._text_embeds.device == device and self._text_embeds.dtype == dtype:
            return self._text_embeds

        if self.tokenizer is None or self.text_encoder is None:
            # Safe fallback. SD1.5 uses 77 tokens, 768 hidden.
            self._text_embeds = torch.zeros((1, 77, 768), device=device, dtype=dtype)
            return self._text_embeds

        tokens = self.tokenizer(
            [self._prompt],
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        input_ids = tokens.input_ids.to(device)
        attn_mask = getattr(tokens, "attention_mask", None)
        if attn_mask is not None:
            attn_mask = attn_mask.to(device)

        with torch.no_grad():
            out = self.text_encoder(input_ids=input_ids, attention_mask=attn_mask)
            embeds = out[0]  # (1, seq, hidden)

        self._text_embeds = embeds.to(dtype=dtype)
        return self._text_embeds

    @torch.no_grad()
    def sample_feature(
        self,
        basename: str,
        x: torch.Tensor,
        layers: list[int],
        t: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        x: (B,N,4,h,w) scaled latents (already multiplied by scaling_factor).
        """
        if x.ndim != 5:
            raise ValueError(f"riffusion_latent expects x (B,N,4,h,w), got {tuple(x.shape)}")
        if x.shape[2] != 4:
            raise ValueError(f"riffusion_latent expects latent channels=4, got {x.shape[2]}")

        device = next(self.unet.parameters()).device
        dtype = next(self.unet.parameters()).dtype

        B, N, C, H, W = x.shape
        latents = x.reshape(B * N, C, H, W).to(device=device, dtype=dtype)

        # choose timestep
        T = self.noise_scheduler.config.num_train_timesteps
        if t is None:
            timesteps = torch.randint(0, T, size=(B, 1), device=device).repeat(1, N).flatten(0, 1)
        else:
            tt = int(t)
            timesteps = torch.full((B * N,), tt, device=device, dtype=torch.long)

        # forward diffusion (training-style)
        noise = torch.randn_like(latents[:B]).unsqueeze(1).repeat(1, N, 1, 1, 1).flatten(0, 1)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # text condition
        text_embeds = self._get_text_embeds(device=device, dtype=dtype)
        encoder_hidden_states = text_embeds.expand(B * N, -1, -1)

        # clear activations
        for i in range(len(self._acts)):
            self._acts[i] = None

        _ = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=True,
        )

        # normalize layer ids
        Ltot = len(self._acts)
        norm_layers: list[int] = []
        for lid in layers:
            lid2 = (Ltot + lid) if lid < 0 else lid
            if not (0 <= lid2 < Ltot):
                raise ValueError(f"layer id {lid} out of range [-(Ltot), {Ltot-1}] (total={Ltot})")
            norm_layers.append(lid2)

        out: dict[str, torch.Tensor] = {}
        for lid in norm_layers:
            act = self._acts[lid]
            if act is None:
                raise RuntimeError(f"Activation for layer {lid} is None (hook did not fire).")
            act_b = act.reshape(B, N, *act.shape[1:]).contiguous()
            out[f"{basename}_layer_{lid}"] = act_b
        return out

    def close(self):
        for h in self._hooks:
            try:
                h.remove()
            except Exception:
                pass
        self._hooks.clear()