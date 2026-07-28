"""MERT reps: mert_inputs, mert_conv, mert_layer."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import h5py
import numpy as np
import torch
import torchaudio as ta
from transformers import AutoModel, AutoProcessor

from ..feature_utils import canonicalize_feat_2d
from ..registry import RepHandler, register_rep_handler


def _mert_state(device: torch.device, config: Optional[Dict[str, Any]] = None) -> Tuple[Any, Any, int]:
    cfg = config or {}
    model_id = cfg.get("mert_model_id", "m-a-p/MERT-v1-330M")
    target_sr = int(cfg.get("target_sr", 16000))
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device)
    model.eval()
    mert_sr = getattr(processor, "sampling_rate", target_sr)
    return processor, model, mert_sr


class _MertRepHandlerBase:
    def init_models(self, device: torch.device, config: Optional[Dict[str, Any]] = None) -> Any:
        return _mert_state(device, config)

    @staticmethod
    def _prepare_input(
        seg: torch.Tensor,
        state: Any,
        config: Optional[Dict[str, Any]],
        device: torch.device,
    ) -> Tuple[Any, Any, torch.Tensor]:
        cfg = config or {}
        target_sr = int(cfg.get("target_sr", 16000))
        proc, model, mert_sr = state
        if target_sr != mert_sr:
            seg_rs = ta.functional.resample(seg.unsqueeze(0), target_sr, mert_sr).squeeze(0)
        else:
            seg_rs = seg
        out = proc(seg_rs.detach().cpu().numpy(), sampling_rate=mert_sr, return_tensors="pt")
        input_values = out["input_values"].to(device)
        return proc, model, input_values

    @staticmethod
    def _create_2d_dataset(
        grp: h5py.Group,
        ds_name: str,
        shape: Tuple[int, ...],
        meta: Dict[str, Any],
        np_dtype,
        compression: Optional[str],
        extra_attrs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, h5py.Dataset]:
        comp = None if (compression in (None, "none")) else compression
        c, t = shape[0], shape[1]
        main = grp.create_dataset(
            ds_name,
            shape=(0, c, t),
            maxshape=(None, c, t),
            dtype=np_dtype,
            chunks=True,
            compression=comp,
        )
        main.attrs["mert_model_id"] = meta.get("mert_model_id", "m-a-p/MERT-v1-330M")
        main.attrs["mert_rep"] = ds_name
        for k, v in (extra_attrs or {}).items():
            main.attrs[k] = v
        return {"main": main}


class MertInputsRepHandler(_MertRepHandlerBase):
    name = "mert_inputs"

    def probe_shape(
        self,
        win_samples: int,
        device: torch.device,
        state: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tuple[int, ...], Dict[str, Any]]:
        cfg = config or {}
        target_sr = int(cfg.get("target_sr", 16000))
        proc, model, mert_sr = state
        probe = torch.zeros(int(round(win_samples / target_sr * mert_sr)), device=device)
        out = proc(probe.detach().cpu().numpy(), sampling_rate=mert_sr, return_tensors="pt")
        x0 = out["input_values"].detach().cpu().squeeze(0).unsqueeze(0)
        return tuple(x0.shape), {"mert_model_id": cfg.get("mert_model_id", "m-a-p/MERT-v1-330M")}

    def create_datasets(
        self,
        grp: h5py.Group,
        shape: Tuple[int, ...],
        meta: Dict[str, Any],
        *,
        np_dtype,
        compression: Optional[str],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, h5py.Dataset]:
        return self._create_2d_dataset(grp, "mert_inputs", shape, meta, np_dtype, compression)

    def extract(
        self,
        seg: torch.Tensor,
        *,
        device: torch.device,
        state: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        _, _, input_values = self._prepare_input(seg, state, config, device)
        feat = input_values.detach().cpu().squeeze(0).unsqueeze(0)
        return feat, {}


class MertConvRepHandler(_MertRepHandlerBase):
    name = "mert_conv"

    def probe_shape(
        self,
        win_samples: int,
        device: torch.device,
        state: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tuple[int, ...], Dict[str, Any]]:
        cfg = config or {}
        target_sr = int(cfg.get("target_sr", 16000))
        proc, model, mert_sr = state
        probe = torch.zeros(int(round(win_samples / target_sr * mert_sr)), device=device)
        out = proc(probe.detach().cpu().numpy(), sampling_rate=mert_sr, return_tensors="pt")
        conv = model.feature_extractor(out["input_values"].to(device))
        x0 = canonicalize_feat_2d(conv[0]).detach().cpu()
        return tuple(x0.shape), {"mert_model_id": cfg.get("mert_model_id", "m-a-p/MERT-v1-330M")}

    def create_datasets(
        self,
        grp: h5py.Group,
        shape: Tuple[int, ...],
        meta: Dict[str, Any],
        *,
        np_dtype,
        compression: Optional[str],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, h5py.Dataset]:
        return self._create_2d_dataset(grp, "mert_conv", shape, meta, np_dtype, compression)

    def extract(
        self,
        seg: torch.Tensor,
        *,
        device: torch.device,
        state: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        _, model, input_values = self._prepare_input(seg, state, config, device)
        conv = model.feature_extractor(input_values)
        feat = canonicalize_feat_2d(conv[0]).detach().cpu()
        return feat, {}


class MertLayerRepHandler(_MertRepHandlerBase):
    name = "mert_layer"

    @staticmethod
    def _resolve_layer(hs, mert_layer: int) -> int:
        L = mert_layer if mert_layer >= 0 else len(hs) + mert_layer
        if not (0 <= L < len(hs)):
            raise ValueError(f"Invalid mert_layer={mert_layer}. hidden_states has len={len(hs)}.")
        return L

    def probe_shape(
        self,
        win_samples: int,
        device: torch.device,
        state: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Tuple[int, ...], Dict[str, Any]]:
        cfg = config or {}
        target_sr = int(cfg.get("target_sr", 16000))
        mert_layer = int(cfg.get("mert_layer", -1))
        proc, model, mert_sr = state
        probe = torch.zeros(int(round(win_samples / target_sr * mert_sr)), device=device)
        inp = proc(probe.detach().cpu().numpy(), sampling_rate=mert_sr, return_tensors="pt")
        with torch.no_grad():
            out = model(inp["input_values"].to(device), output_hidden_states=True)
        hs = out.hidden_states
        L = self._resolve_layer(hs, mert_layer)
        x0 = canonicalize_feat_2d(hs[L][0]).detach().cpu()
        return tuple(x0.shape), {
            "mert_model_id": cfg.get("mert_model_id", "m-a-p/MERT-v1-330M"),
            "mert_layer": L,
        }

    def create_datasets(
        self,
        grp: h5py.Group,
        shape: Tuple[int, ...],
        meta: Dict[str, Any],
        *,
        np_dtype,
        compression: Optional[str],
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, h5py.Dataset]:
        return self._create_2d_dataset(
            grp, "mert_layer", shape, meta, np_dtype, compression,
            extra_attrs={"mert_layer": int(meta.get("mert_layer", -1))},
        )

    def extract(
        self,
        seg: torch.Tensor,
        *,
        device: torch.device,
        state: Any,
        config: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        cfg = config or {}
        mert_layer = int(cfg.get("mert_layer", -1))
        _, model, input_values = self._prepare_input(seg, state, config, device)
        with torch.no_grad():
            out = model(input_values, output_hidden_states=True)
        hs = out.hidden_states
        L = self._resolve_layer(hs, mert_layer)
        feat = canonicalize_feat_2d(hs[L][0]).detach().cpu()
        return feat, {}


register_rep_handler(MertInputsRepHandler())
register_rep_handler(MertConvRepHandler())
register_rep_handler(MertLayerRepHandler())
