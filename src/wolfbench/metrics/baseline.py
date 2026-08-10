"""Helpers to compute baseline statistics from a clean-market run."""
from __future__ import annotations

import numpy as np


def baseline_from_market_history(history_per_asset: dict[str, dict[str, list[float]]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for asset, h in history_per_asset.items():
        gaps = []
        for p, f in zip(h["price"], h["fundamental"]):
            gaps.append((p - f) / max(f, 1e-6))
        vol = np.array(h["volume"], dtype=float)
        spread = np.array(h["spread_bps"], dtype=float)
        out[asset] = {
            "volume_mu": float(vol.mean()) if len(vol) else 0.0,
            "volume_sd": float(vol.std() + 1e-6),
            "spread_bps_mu": float(spread.mean()) if len(spread) else 0.0,
            "spread_bps_sd": float(spread.std() + 1e-6),
            "price_gap_mu": float(np.mean(gaps)) if gaps else 0.0,
            "price_gap_sd": float(np.std(gaps) + 1e-6),
            # filled by social baseline merge
            "msg_volume_mu": 0.0,
            "msg_volume_sd": 1.0,
        }
    return out


def merge_social_baseline(baseline: dict[str, dict[str, float]],
                          social_history: dict[str, list[dict[str, float]]]) -> None:
    for asset, hist in social_history.items():
        vols = np.array([h["msg_volume"] for h in hist], dtype=float)
        if asset not in baseline:
            baseline[asset] = {}
        baseline[asset]["msg_volume_mu"] = float(vols.mean()) if len(vols) else 0.0
        baseline[asset]["msg_volume_sd"] = float(vols.std() + 1e-6)
