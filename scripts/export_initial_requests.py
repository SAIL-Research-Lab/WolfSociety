#!/usr/bin/env python3
"""Export distinct initial agent observations for an endpoint load check.

No inference is performed. Empty initial inboxes/memories make this a startup
benchmark, not a prediction of full-episode throughput. Replay late-round pilot
backend_audit traces separately to measure actual steady-state context costs.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

from wolfbench.llm_runtime.backend import DecisionRequest, MockBackend
from wolfbench.llm_runtime.simulation import ACTION_SCHEMA, LanguageSocietyEnv, stable_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-agents", type=int, choices=(1000, 2000), required=True)
    parser.add_argument("--alpha", type=float, default=.05)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output file; existing requests are not overwritten")
    config = {"scenario": "s1", "n_society": args.n_agents, "alpha": args.alpha,
              "seed": args.seed, "horizon_days": 30, "controller": "llm"}
    env = LanguageSocietyEnv(config, MockBackend())
    prices = {asset: state.price for asset, state in env.market.assets.items()}
    observation = env._build_observation(0, prices, {asset: 0.0 for asset in prices})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        for agent in env.population:
            payload = env._build_payload(agent, 0, observation)
            request = DecisionRequest(f"initial:{args.n_agents}:{agent.public_id}",
                env._system_prompt(agent), json.dumps(payload, sort_keys=True),
                stable_seed(args.seed, "model", 0, agent.public_id))
            row = {**asdict(request), "schema": ACTION_SCHEMA, "source_stage": "initial_observation_only"}
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"requests": len(env.population), "model_calls": 0,
                      "source_stage": "initial_observation_only", "output": str(args.output)}))


if __name__ == "__main__":
    main()
