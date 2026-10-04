"""E1c: fixed-count and liquidity-scaling decomposition.

This reviewer-facing experiment asks whether the falling critical harmful
fraction is merely induced by the simulator's sub-linear liquidity rule. It
holds the requested harmful count K fixed across society sizes and crosses the
social channel with three liquidity exponents: fixed depth (q=0), WolfBench
baseline (q=1/2), and per-capita depth (q=1).
"""
from __future__ import annotations

import argparse
import os
import time
from copy import deepcopy

from wolfbench.scenarios.base import load_scenario

from .hybrid_runtime import make_population_backend, run_hybrid_episode
from .run_common import add_common_args, backend_delta, write_run_outputs


VARIANTS = {
    "baseline_q05": {"liquidity_exponent": 0.5, "social": True},
    "no_social_q05": {"liquidity_exponent": 0.5, "social": False},
    "fixed_depth_q0": {"liquidity_exponent": 0.0, "social": True},
    "per_capita_depth_q1": {"liquidity_exponent": 1.0, "social": True},
    "per_capita_no_social_q1": {"liquidity_exponent": 1.0, "social": False},
}


def _ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, "e1_size_decomposition")
    parser.add_argument("--n-values", default=os.getenv("WOLFBENCH_E1D_N", "100,300,1000"))
    parser.add_argument("--harmful-counts", default=os.getenv("WOLFBENCH_E1D_K", "1,3,5,8,12"))
    parser.add_argument("--seeds", default=os.getenv("WOLFBENCH_E1D_SEEDS", "1,2,3"))
    args = parser.parse_args()
    n_values, harmful_counts, seeds = _ints(args.n_values), _ints(args.harmful_counts), _ints(args.seeds)
    rows = []
    total = len(VARIANTS) * len(n_values) * len(harmful_counts) * len(seeds)
    index = 0
    for variant, settings in VARIANTS.items():
        scenario = deepcopy(load_scenario("s1"))
        scenario.retail["controller_mode"] = "mixed_roles"
        scenario.market_makers["liquidity_exponent"] = settings["liquidity_exponent"]
        if not settings["social"]:
            scenario.social["p_expose"] = 0.0
            scenario.social["p_reshare"] = 0.0
            scenario.retail["conformity_scale"] = 0.0
        for n_society in n_values:
            for harmful_count in harmful_counts:
                alpha = harmful_count / n_society
                for seed in seeds:
                    index += 1
                    print(
                        f"[{index}/{total}] {args.out} variant={variant} N={n_society} "
                        f"K={harmful_count} alpha={alpha:.6g} seed={seed}",
                        flush=True,
                    )
                    started = time.time()
                    backend = make_population_backend(
                        model=args.model,
                        experiment_name=args.out,
                        mock=args.mock,
                        strict=not args.mock,
                    )
                    before = backend.snapshot()
                    try:
                        row, _ = run_hybrid_episode(
                            deepcopy(scenario),
                            n_society=n_society,
                            alpha=alpha,
                            seed=seed,
                            population_backend=backend,
                            quota_mode=args.quota_mode,
                            plan_interval=args.plan_interval,
                        )
                        row.update({"status": "ok", "error_type": "", "error": ""})
                    except Exception as exc:
                        if not args.continue_on_error:
                            raise
                        row = {
                            "scenario": "s1",
                            "n_society": n_society,
                            "alpha": alpha,
                            "seed": seed,
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "error": repr(exc),
                        }
                    row.update({
                        "experiment": args.out,
                        "variant": variant,
                        "requested_harmful_count": harmful_count,
                        "social_channel_enabled": int(settings["social"]),
                        "liquidity_exponent_requested": settings["liquidity_exponent"],
                        "defense": "noguard",
                        "mock_openrouter": int(args.mock),
                        "runtime_sec": time.time() - started,
                    })
                    row.update(backend_delta(before, backend.snapshot(), "population_llm_episode"))
                    rows.append(row)
    out_dir = write_run_outputs(
        args.out,
        rows,
        {
            "experiment": "E1c fixed-count and liquidity decomposition",
            "research_question": (
                "Does alpha_c fall with N after separating network amplification from "
                "the simulator liquidity exponent?"
            ),
            "variants": VARIANTS,
            "n_values": n_values,
            "harmful_counts": harmful_counts,
            "seeds": seeds,
            "population_model": args.model,
            "mock_openrouter": args.mock,
            "quota_mode": args.quota_mode,
        },
        ["variant", "n_society", "requested_harmful_count"],
    )
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
