"""Paired seed-level mechanism contrasts for P05 information cascades."""
from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from statistics import mean, stdev

from ..runtime.io_utils import OUTPUTS, write_csv, write_json


PAIR_KEYS = ("scenario", "n_society", "alpha", "seed")
CONTRASTS = [
    ("full_game", "private_only", "social_information_bits", 1),
    ("full_game", "private_only", "cascade_decision_rate", 1),
    ("full_game", "private_only", "transfer_entropy_social_to_trade_bits", 1),
    ("full_game", "content_only", "social_proof_information_bits", 1),
    ("proof_only", "private_only", "social_proof_information_bits", 1),
    ("low_attention", "full_game", "mean_attention_used", -1),
    ("high_attention", "full_game", "mean_attention_used", 1),
    ("precise_private_signal", "noisy_private_signal", "private_signal_quality_bits", 1),
    ("delayed_messages", "full_game", "transfer_entropy_social_to_trade_bits", -1),
    ("hub_placement", "full_game", "max_cascade_reach", 1),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="p05_information_cascade")
    args = parser.parse_args()
    out_dir = OUTPUTS / args.run
    with (out_dir / "data.csv").open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("status", "ok") == "ok"]
    output = []
    for treatment, control, metric, expected_sign in CONTRASTS:
        paired: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
        for row in rows:
            if row["variant"] not in {treatment, control}:
                continue
            key = tuple(str(row[name]) for name in PAIR_KEYS)
            paired[key][row["variant"]] = float(row[metric])
        delta = [
            values[treatment] - values[control]
            for values in paired.values()
            if treatment in values and control in values
        ]
        n = len(delta)
        effect = mean(delta) if n else float("nan")
        sd = stdev(delta) if n > 1 else 0.0
        output.append({
            "treatment": treatment,
            "control": control,
            "metric": metric,
            "expected_direction": "positive" if expected_sign > 0 else "negative",
            "n_pairs": n,
            "mean_paired_delta": effect,
            "sd_paired_delta": sd,
            "se_paired_delta": sd / math.sqrt(n) if n else float("nan"),
            "sign_prediction_supported": int(n > 0 and expected_sign * effect > 0),
        })
    write_csv(output, out_dir / "information_contrasts.csv")
    write_json({
        "n_preregistered_contrasts": len(output),
        "n_supported": sum(row["sign_prediction_supported"] for row in output),
        "independent_unit": "paired episode seed",
    }, out_dir / "information_contrasts.json")
    print(f"Wrote {out_dir / 'information_contrasts.csv'}")


if __name__ == "__main__":
    main()
