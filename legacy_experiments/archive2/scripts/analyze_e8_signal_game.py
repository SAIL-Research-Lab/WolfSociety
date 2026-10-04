"""Paired, preregistered-sign analysis for E8 network-game interventions."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any

from .io_utils import OUTPUTS, write_csv, write_json


PAIR_KEYS = ["scenario", "n_society", "alpha", "seed"]

# (treatment, control, metric, expected sign, mechanism)
CONTRASTS = [
    ("full_game", "private_only", "social_information_bits", 1, "social content adds action-relevant information"),
    ("full_game", "private_only", "cascade_decision_rate", 1, "social signals override conflicting private signals"),
    ("full_game", "private_only", "transfer_entropy_social_to_trade_bits", 1, "social exposure predicts later trade"),
    ("full_game", "content_only", "social_proof_information_bits", 1, "visible popularity adds coordination information"),
    ("proof_only", "private_only", "social_proof_information_bits", 1, "popularity alone affects choice"),
    ("low_attention", "full_game", "mean_attention_used", -1, "attention intervention changes processed exposure"),
    ("high_attention", "full_game", "mean_attention_used", 1, "greater capacity increases processed exposure"),
    ("precise_private_signal", "noisy_private_signal", "private_signal_quality_bits", 1, "better private signals preserve more information about the true value direction"),
    ("delayed_messages", "full_game", "transfer_entropy_social_to_trade_bits", -1, "stale social signals weaken directed information flow"),
    ("hub_placement", "full_game", "max_cascade_reach", 1, "network-central placement expands reach"),
]


def paired_contrast(
    rows: list[dict[str, Any]],
    treatment: str,
    control: str,
    metric: str,
    expected_sign: int,
    mechanism: str,
) -> dict[str, object]:
    paired: dict[tuple[str, ...], dict[str, float]] = {}
    for row in rows:
        variant = str(row.get("variant", ""))
        if variant not in {treatment, control}:
            continue
        try:
            value = float(row[metric])
        except (KeyError, TypeError, ValueError):
            continue
        key = tuple(str(row.get(name, "")) for name in PAIR_KEYS)
        paired.setdefault(key, {})[variant] = value
    delta = [
        values[treatment] - values[control]
        for values in paired.values()
        if treatment in values and control in values
    ]
    n = len(delta)
    delta_mean = float(mean(delta)) if n else float("nan")
    sd = float(stdev(delta)) if n > 1 else 0.0
    se = float(sd / math.sqrt(n)) if n else float("nan")
    return {
        "treatment": treatment,
        "control": control,
        "metric": metric,
        "delta_definition": "treatment_minus_control",
        "expected_direction": "positive" if expected_sign > 0 else "negative",
        "mechanism": mechanism,
        "n_pairs": n,
        "mean_paired_delta": delta_mean,
        "sd_paired_delta": sd,
        "se_paired_delta": se,
        "sign_prediction_supported": int(n > 0 and expected_sign * delta_mean > 0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="e8_network_signal_game")
    parser.add_argument("--data", default="")
    args = parser.parse_args()
    output_dir = OUTPUTS / args.out
    data_path = Path(args.data) if args.data else output_dir / "data.csv"
    with data_path.open(newline="") as handle:
        data_rows = [
            row for row in csv.DictReader(handle)
            if str(row.get("status", "ok") or "ok") == "ok"
        ]
    rows = [paired_contrast(data_rows, *contrast) for contrast in CONTRASTS]
    write_csv(rows, output_dir / "mechanism_contrasts.csv")
    write_json({
        "data": str(data_path),
        "pair_keys": PAIR_KEYS,
        "n_preregistered_contrasts": len(rows),
        "n_supported_signs": sum(int(row["sign_prediction_supported"]) for row in rows),
        "inference_note": (
            "Signs are descriptive mechanism checks. Report paired uncertainty or bootstrap "
            "intervals from the paper-scale seed distribution before making inferential claims."
        ),
    }, output_dir / "mechanism_contrasts.json")
    print(f"Wrote {output_dir / 'mechanism_contrasts.csv'}")


if __name__ == "__main__":
    main()
