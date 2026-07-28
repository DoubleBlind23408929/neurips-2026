"""
Rep (feature representation) handlers for `pack_split_dataset`.

All rep modules under this package are auto-imported, and each module registers
its handler(s) with the central registry on import — so adding a new rep only
needs one new file under feature_extraction/reps/ (no edits here).

Built-ins: mel, mert_inputs, mert_conv, mert_layer, muq_layer, musicfm_layer,
cqt, riffusion_img, riffusion_latent.
"""
from __future__ import annotations

from importlib import import_module
from pkgutil import iter_modules


def _auto_import_rep_modules() -> None:
    for mod in iter_modules(__path__):
        name = str(mod.name)
        if name.startswith("_"):
            continue
        import_module(f"{__name__}.{name}")


_auto_import_rep_modules()
