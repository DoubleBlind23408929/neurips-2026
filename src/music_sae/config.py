from __future__ import annotations

import typing as tp


def _make_model_config(data_tag: str, layer_id: int, d_model: int) -> dict[str, tp.Any]:
    return {
        "data_tag": f"{data_tag}_{layer_id}",
        "feature_dims": {f"{data_tag}_{layer_id}": d_model},
        "models": {"feature": None},  # features pre-extracted to H5, passthrough
    }


aug_order_1 = ["ps-2", "ps-1", "orig", "ps+1", "ps+2"]
aug_order_7 = ["ps-14", "ps-7", "orig", "ps+7", "ps+14"]


def _make_exp(model_config: dict[str, tp.Any], **kwargs: tp.Any) -> dict[str, tp.Any]:
    base = {
        "aug": True,
        "share_mask": True,
        "n_saes": 3,
        "mlp_ratio": 4,
        "aug_order": aug_order_1,
        "norm_every": 1,
        "model_config": model_config,
        "norm_stats_path": None,  # path to .npz with mean/std; None → compute from batches
    }
    base.update(kwargs)
    return base


exp_config: dict[str, dict[str, tp.Any]] = {
    "musicfm-10_1-1": _make_exp(_make_model_config("musicfm_layer", 10, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "musicfm-8_1-1": _make_exp(_make_model_config("musicfm_layer", 8, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "musicfm-6_1-1": _make_exp(_make_model_config("musicfm_layer", 6, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "musicfm-5_1-1": _make_exp(_make_model_config("musicfm_layer", 5, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "musicfm-4_1-1": _make_exp(_make_model_config("musicfm_layer", 4, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "musicfm-3_1-1": _make_exp(_make_model_config("musicfm_layer", 3, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "musicfm-2_1-1": _make_exp(_make_model_config("musicfm_layer", 2, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "musicfm-0_1-1": _make_exp(_make_model_config("musicfm_layer", 0, 1024),
                            n_saes=1, aug_order=aug_order_1),


    "musicfm-10_3-1": _make_exp(_make_model_config("musicfm_layer", 10, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "musicfm-8_3-1": _make_exp(_make_model_config("musicfm_layer", 8, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "musicfm-6_3-1": _make_exp(_make_model_config("musicfm_layer", 6, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "musicfm-5_3-1": _make_exp(_make_model_config("musicfm_layer", 5, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "musicfm-4_3-1": _make_exp(_make_model_config("musicfm_layer", 4, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "musicfm-3_3-1": _make_exp(_make_model_config("musicfm_layer", 3, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "musicfm-2_3-1": _make_exp(_make_model_config("musicfm_layer", 2, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "musicfm-0_3-1": _make_exp(_make_model_config("musicfm_layer", 0, 1024),
                            n_saes=3, aug_order=aug_order_1),


    "muq-10_3-1": _make_exp(_make_model_config("muq_layer", 10, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "muq-8_3-1": _make_exp(_make_model_config("muq_layer", 8, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "muq-6_3-1": _make_exp(_make_model_config("muq_layer", 6, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "muq-5_3-1": _make_exp(_make_model_config("muq_layer", 5, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "muq-4_3-1": _make_exp(_make_model_config("muq_layer", 4, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "muq-3_3-1": _make_exp(_make_model_config("muq_layer", 3, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "muq-2_3-1": _make_exp(_make_model_config("muq_layer", 2, 1024),
                            n_saes=3, aug_order=aug_order_1),
    "muq-0_3-1": _make_exp(_make_model_config("muq_layer", 0, 1024),
                            n_saes=3, aug_order=aug_order_1),


    "muq-10_1-1": _make_exp(_make_model_config("muq_layer", 10, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "muq-8_1-1": _make_exp(_make_model_config("muq_layer", 8, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "muq-6_1-1": _make_exp(_make_model_config("muq_layer", 6, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "muq-5_1-1": _make_exp(_make_model_config("muq_layer", 5, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "muq-4_1-1": _make_exp(_make_model_config("muq_layer", 4, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "muq-3_1-1": _make_exp(_make_model_config("muq_layer", 3, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "muq-2_1-1": _make_exp(_make_model_config("muq_layer", 2, 1024),
                            n_saes=1, aug_order=aug_order_1),
    "muq-0_1-1": _make_exp(_make_model_config("muq_layer", 0, 1024),
                            n_saes=1, aug_order=aug_order_1),

    "muq-5_3-1-sync": _make_exp(_make_model_config("muq_layer", 5, 1024),
                            n_saes=3, aug_order=aug_order_1),

    "mert-4_3-1": _make_exp(_make_model_config("mert_layer", 4, 1024),
                            n_saes=3, aug_order=aug_order_1,
                            norm_stats_path="sae-data/slakh2100-val/mert/slakh2100-val_mert_stats_4.npz"),
    "mert-2_3-1": _make_exp(_make_model_config("mert_layer", 2, 1024),
                            n_saes=3, aug_order=aug_order_1,
                            norm_stats_path="sae-data/slakh2100-val/mert/slakh2100-val_mert_stats_2.npz"),
    "mert-3_3-1": _make_exp(_make_model_config("mert_layer", 3, 1024),
                            n_saes=3, aug_order=aug_order_1,
                            norm_stats_path="sae-data/slakh2100-val/mert/slakh2100-val_mert_stats_3.npz"),
    "mert-5_3-1": _make_exp(_make_model_config("mert_layer", 5, 1024),
                            n_saes=3, aug_order=aug_order_1,
                            norm_stats_path="sae-data/slakh2100-val/mert/slakh2100-val_mert_stats_5.npz"),

    "cqt-0_3-1": _make_exp(_make_model_config("cqt_layer", 0, 288),
                          n_saes=3, aug_order=aug_order_1),

    "cqt-0_1-1": _make_exp(_make_model_config("cqt_layer", 0, 288),
                          n_saes=1, aug_order=aug_order_1),

}
