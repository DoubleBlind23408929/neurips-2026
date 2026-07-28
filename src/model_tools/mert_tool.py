from __future__ import annotations
import torch
from torch import nn
from transformers import AutoModel


class MERTEncoderTool(nn.Module):
    """
    Take precomputed features, run MERT feature_projection -> (mask) -> encoder, and return hidden states.

    Input x:
      - (B, C, T) or (B, N, C, T)
      - also tolerates (B, T, C) or (B, N, T, C) via heuristics

    Masking (officially placed AFTER feature_projection):
      feature_projection -> _mask_hidden_states -> encoder
    """

    def __init__(
        self,
        model_id: str = "m-a-p/MERT-v1-330M",
        trust_remote_code: bool = True,

        # masking
        enable_mask: bool = True,
        mask_ratio: float = 0.0,                         # used in eval or if range disabled
        mask_ratio_range: tuple[float, float] | None = None,  # used when self.training
        mask_span: int = 10,
        deterministic: bool = False,
        seed: int = 1234,
        share_mask_across_n: bool = True,

        # output layout
        out_layout: str = "BNCT",   # BNCT/BNTC/BCT/BTC
        restore_bn: bool = True,
    ) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.model.eval()

        self.enable_mask = bool(enable_mask)
        self.mask_ratio = float(mask_ratio)
        self.mask_ratio_range = mask_ratio_range
        self.mask_span = int(mask_span)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.share_mask_across_n = bool(share_mask_across_n)

        self.out_layout = str(out_layout)
        self.restore_bn = bool(restore_bn)

        if self.mask_span < 1:
            raise ValueError("mask_span must be >= 1")
        if not (0.0 <= self.mask_ratio <= 1.0):
            raise ValueError("mask_ratio must be in [0,1]")
        if self.mask_ratio_range is not None:
            lo, hi = self.mask_ratio_range
            if not (0.0 <= lo <= hi <= 1.0):
                raise ValueError("mask_ratio_range must satisfy 0<=lo<=hi<=1")
        if self.out_layout not in ("BNCT", "BNTC", "BCT", "BTC"):
            raise ValueError("out_layout must be BNCT/BNTC/BCT/BTC")
        
    # ---- ModelFactory compat ----
    @classmethod
    def load_from_checkpoint(cls, ckpt_path: str | None = None, strict: bool = False, **kwargs):
        return cls(**kwargs)

    def on_fit_start(self) -> None:
        return

    # ---- shape helpers ----
    def _looks_like_channel(self, d: int) -> bool:
        return d in (64, 128, 256, 512, 768, 1024, 1536, 2048)

    def _to_bnct(self, x: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """
        Normalize to (B,N,C,T). N=1 if input is 3D.
        """
        meta = {"was_4d": False, "B": None, "N": None, "input_layout": None}

        if x.dim() == 3:
            B, d1, d2 = x.shape
            meta["B"], meta["N"] = B, 1
            if self._looks_like_channel(d1):
                meta["input_layout"] = "BCT"
                return x.unsqueeze(1).contiguous(), meta
            if self._looks_like_channel(d2):
                meta["input_layout"] = "BTC"
                return x.transpose(1, 2).unsqueeze(1).contiguous(), meta
            # fallback: smaller is C
            if d1 <= d2:
                meta["input_layout"] = "BCT"
                return x.unsqueeze(1).contiguous(), meta
            meta["input_layout"] = "BTC"
            return x.transpose(1, 2).unsqueeze(1).contiguous(), meta

        if x.dim() == 4:
            meta["was_4d"] = True
            B, N, d2, d3 = x.shape
            meta["B"], meta["N"] = B, N
            if self._looks_like_channel(d2) and not self._looks_like_channel(d3):
                meta["input_layout"] = "BNCT"
                return x.contiguous(), meta
            if self._looks_like_channel(d3) and not self._looks_like_channel(d2):
                meta["input_layout"] = "BNTC"
                return x.permute(0, 1, 3, 2).contiguous(), meta
            # fallback: smaller is C
            if d2 <= d3:
                meta["input_layout"] = "BNCT"
                return x.contiguous(), meta
            meta["input_layout"] = "BNTC"
            return x.permute(0, 1, 3, 2).contiguous(), meta

        raise ValueError(f"Expected 3D/4D tensor, got {x.dim()}D")

    def _restore(self, x_bnct: torch.Tensor, meta: dict) -> torch.Tensor:
        """
        x_bnct: (B,N,C,T) -> out layout.
        """
        B, N, C, T = x_bnct.shape

        if meta.get("was_4d", False) and self.restore_bn:
            if self.out_layout == "BNCT":
                return x_bnct.contiguous()
            if self.out_layout == "BNTC":
                return x_bnct.permute(0, 1, 3, 2).contiguous()
            # asked 3D but restoring BN: default BNCT
            return x_bnct.contiguous()

        # flatten BN -> Bflat
        x_bct = x_bnct.reshape(B * N, C, T)
        if self.out_layout == "BCT":
            return x_bct.contiguous()
        if self.out_layout == "BTC":
            return x_bct.transpose(1, 2).contiguous()
        return x_bct.contiguous()

    # ---- masking ----
    def _effective_ratio(self, device: torch.device) -> float:
        if self.enable_mask and self.mask_ratio_range is not None:
            lo, hi = self.mask_ratio_range
            if lo == hi:
                return float(lo)
            g = None
            if self.deterministic:
                g = torch.Generator(device=device)
                g.manual_seed(self.seed)
            r = torch.rand((), device=device, generator=g)
            return float(lo + (hi - lo) * r.item())
        return float(self.mask_ratio)

    def _make_mask_indices(self, B: int, T: int, ratio: float, device: torch.device) -> torch.Tensor:
        """
        Return mask_time_indices: (B,T) bool, span masking.
        """
        if (not self.enable_mask) or ratio <= 0.0 or T == 0:
            return torch.zeros((B, T), device=device, dtype=torch.bool)

        span = min(self.mask_span, T)
        n_spans = int(round((ratio * T) / span))
        n_spans = max(0, min(n_spans, T))
        if n_spans == 0:
            return torch.zeros((B, T), device=device, dtype=torch.bool)

        g = None
        if self.deterministic:
            g = torch.Generator(device=device)
            g.manual_seed(self.seed)

        mask = torch.zeros((B, T), device=device, dtype=torch.bool)
        max_start = T - span
        for b in range(B):
            if max_start <= 0:
                mask[b, :span] = True
                continue
            starts = torch.randint(0, max_start + 1, (n_spans,), generator=g, device=device)
            for s in starts.tolist():
                mask[b, s : s + span] = True
        return mask

    def _apply_official_mask(self, hidden_states: torch.Tensor, mask_time_indices: torch.Tensor) -> torch.Tensor:
        """
        hidden_states: (BN, T, H)
        mask_time_indices: (BN, T) bool

        Prefer calling model._mask_hidden_states (matches official code path),
        else fall back to manual replacement using model.masked_spec_embed if present.
        """
        if mask_time_indices is None or not mask_time_indices.any():
            return hidden_states

        if hasattr(self.model, "_mask_hidden_states") and callable(getattr(self.model, "_mask_hidden_states")):
            return self.model._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        # fallback: manual replace
        if not hasattr(self.model, "masked_spec_embed"):
            # if no embed, just zero out
            hs = hidden_states.clone()
            hs[mask_time_indices.unsqueeze(-1).expand_as(hs)] = 0.0
            return hs

        embed = self.model.masked_spec_embed  # (H,)
        hs = hidden_states.clone()
        hs[mask_time_indices] = embed.to(hs.dtype).to(hs.device)
        return hs

    # ---- core: projection -> mask -> encoder ----
    def _project_mask_encode(self, feats_bnct: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        feats_bnct: (B,N,C,T)
        Returns tuple hidden_states, each (B,N,T,H)
        """
        B, N, C, T = feats_bnct.shape
        device = next(self.model.parameters()).device
        feats_bnct = feats_bnct.to(device=device, dtype=torch.float32)

        # flatten BN and switch to (BN,T,C)
        x_bct = feats_bnct.reshape(B * N, C, T)              # (BN,C,T)
        x_btc = x_bct.transpose(1, 2).contiguous()           # (BN,T,C)

        hidden_size = int(getattr(self.model.config, "hidden_size", x_btc.size(-1)))

        # ---- IMPORTANT FIX: feature_projection returns TWO values ----
        if x_btc.size(-1) != hidden_size:
            if not hasattr(self.model, "feature_projection"):
                raise ValueError(
                    f"Input C={x_btc.size(-1)} != hidden_size={hidden_size}, "
                    "and model has no feature_projection."
                )
            proj_hidden = self.model.feature_projection(x_btc)  # (BN,T,H), (BN,T,feat_dim)
        else:
            proj_hidden = x_btc

        # ---- IMPORTANT FIX: mask is applied AFTER projection ----
        ratio = self._effective_ratio(device)
        self.enable_mask = False
        print(self.enable_mask, ratio, "===============")

        if self.enable_mask and ratio > 0.0:
            if self.share_mask_across_n:
                base = self._make_mask_indices(B, proj_hidden.size(1), ratio, device)     # (B,T)
                mask_time_indices = base.repeat_interleave(N, dim=0)                      # (BN,T)
            else:
                mask_time_indices = self._make_mask_indices(B * N, proj_hidden.size(1), ratio, device)

            proj_hidden = self._apply_official_mask(proj_hidden, mask_time_indices)

        # encoder
        enc_out = self.model.encoder(
            hidden_states=proj_hidden,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hs = enc_out.hidden_states  # tuple of (BN,T,H)

        # reshape back to (B,N,T,H)
        return tuple(h.view(B, N, h.size(1), h.size(2)) for h in hs)

    @torch.no_grad()
    def sample_feature(
        self,
        basename: str,
        x: torch.Tensor,
        layers: list[int],
        t: int | None = None,
    ) -> dict[str, torch.Tensor]:
        feats_bnct, meta = self._to_bnct(x.float())
        hs_bn = self._project_mask_encode(feats_bnct)  # tuple[(B,N,T,H)]

        out: dict[str, torch.Tensor] = {}
        for layer_id in layers:
            if layer_id < 0 or layer_id >= len(hs_bn):
                raise ValueError(f"Requested layer {layer_id}, but hidden_states has {len(hs_bn)} entries.")
            h = hs_bn[layer_id]                        # (B,N,T,H)
            out[f"{basename}_layer_{layer_id}"] = h
        return out