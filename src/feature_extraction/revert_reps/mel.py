from __future__ import annotations

import os
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import torch

from src.audioldm_2_utilities.feature_pipeline import AudioLDM2Pipeline

from .registry import register_revert_handler


class MelRevertHandler:
    name = "mel"

    def can_handle(self, split_group: h5py.Group) -> bool:
        return "mel" in split_group

    def num_items(self, split_group: h5py.Group) -> int:
        return int(split_group["mel"].shape[0])

    def init_state(self, split_group: h5py.Group, device: torch.device, args: Any) -> Any:
        ds = split_group["mel"]
        sr = int(ds.attrs.get("sr", getattr(args, "default_sr", 16000)))
        n_mels = int(ds.shape[1])
        local = getattr(args, "audioldm2_dir", None) or os.environ.get("AUDIO_LDM2_DIR")
        if not local:
            raise ValueError(
                "Mel revert needs AudioLDM2 weights. Set --audioldm2-dir or AUDIO_LDM2_DIR."
            )
        pipe = AudioLDM2Pipeline(localDir=local, sampleRate=sr, device=device, nMels=n_mels)
        pipe.eval()
        return {"pipe": pipe, "sr": sr}

    def decode_index(
        self,
        split_group: h5py.Group,
        index: int,
        state: Any,
        device: torch.device,
        args: Any,
    ) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        ds = split_group["mel"]
        x = ds[index]  # (n_mels, T)
        mel = torch.from_numpy(np.asarray(x)).to(torch.float32)
        with torch.no_grad():
            wav = state["pipe"].from_rep(mel, rep="mel")  # expected (1,N) or (B,N)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        elif wav.dim() == 2 and wav.shape[0] > 1:
            wav = wav[:1]
        wav = wav.detach().cpu().to(torch.float32)
        return wav, int(state["sr"]), {"feature_shape": tuple(mel.shape)}


register_revert_handler(MelRevertHandler())

