from __future__ import annotations

from typing import Any, Dict, Optional, Protocol, Tuple

import h5py
import torch


class RevertHandler(Protocol):
    """
    Interface for reversible feature handlers.
    """

    name: str

    def can_handle(self, split_group: h5py.Group) -> bool:
        ...

    def num_items(self, split_group: h5py.Group) -> int:
        ...

    def init_state(self, split_group: h5py.Group, device: torch.device, args: Any) -> Any:
        ...

    def decode_index(
        self,
        split_group: h5py.Group,
        index: int,
        state: Any,
        device: torch.device,
        args: Any,
    ) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        """
        Returns:
          wav: torch.Tensor shaped (1, N), float32 on CPU
          sr:  sample rate
          meta: extra debug information
        """
        ...


_REGISTRY: Dict[str, RevertHandler] = {}


def register_revert_handler(handler: RevertHandler) -> None:
    if handler.name in _REGISTRY:
        raise ValueError(f"Revert handler '{handler.name}' already registered")
    _REGISTRY[handler.name] = handler


def get_revert_handler(name: str) -> RevertHandler:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown revert handler '{name}'. Available: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_revert_handlers() -> Dict[str, RevertHandler]:
    return dict(_REGISTRY)

