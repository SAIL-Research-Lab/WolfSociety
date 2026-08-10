"""Command-line interface."""
from __future__ import annotations

import json
from pathlib import Path

import click
import numpy as np
import yaml

from wolfbench.agents.wolfguard import WolfGuardAgent, WolfGuardConfig
from wolfbench.data import export_trajectory_dataset, iter_jsonl
from wolfbench.defense.isolation import SubprocessWolfGuardPolicy
from wolfbench.defense import get_policy
from wolfbench.defense.distilled import evaluate_distilled_model, train_distilled_model
from wolfbench.env.environment import WolfBenchEnv
from wolfbench.metrics import defense_score, threshold_shift
from wolfbench.scenarios.base import load_scenario, SCENARIO_FILES
from wolfbench.tracks.runner import (
    calibrate_clean_baseline,
    run_attack_track,
    run_defense_track,
    run_scaling_track,
)
from wolfbench.utils.logging import get_logger

log = get_logger("wolfbench.cli")


def _parse_floats(spec: str) -> list[float]:
    return [float(x) for x in spec.split(",") if x.strip()]


def _parse_ints(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip()]


@click.group()
def main():
    """WolfBench CLI."""


@main.command("scenarios")
def list_scenarios():
    """List available scenarios."""
    for sid in SCENARIO_FILES:
        s = load_scenario(sid)
        click.echo(f"{sid}\t{s.name}\thorizon={s.horizon_days}d\ttarget={s.target_asset}")


@main.command("run")
@click.option("--scenario", default="s1", type=click.Choice(list(SCENARIO_FILES)))
@click.option("--alpha", default=0.02, type=float, help="Harmful agent ratio.")
@click.option("--n-society", default=1000, type=int)
@click.option("--seed", default=1, type=int)
@click.option("--defense/--no-defense", default=False, help="Enable WolfGuard.")
@click.option("--llm-leaders", default=0, type=int,
              help="Number of strategic leaders to upgrade to LLM control "
                   "(capped by scenario leader_count_max).")
@click.option("--llm-model", default=None,
              help="OpenAI-compatible model id; if unset uses RuleFallbackBackend.")
@click.option("--llm-provider", default="openai",
              type=click.Choice(["openai", "openrouter", "vllm"]),
              help="Backend provider for --llm-leaders.")
@click.option("--llm-base-url", default=None,
              help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1 or https://openrouter.ai/api/v1.")
@click.option("--llm-api-key", default=None,
              help="API key for the OpenAI-compatible endpoint; vLLM accepts EMPTY.")
@click.option("--out", type=click.Path(), default=None,
              help="Optional JSON output path.")
def run_episode(scenario, alpha, n_society, seed, defense, llm_leaders,
                llm_model, llm_provider, llm_base_url, llm_api_key, out):
    """Run a single 30-day episode and print summary metrics."""
    scen = load_scenario(scenario)
    backend = None
    if llm_leaders > 0 and (llm_model or llm_provider == "vllm" or llm_base_url):
        from wolfbench.agents.llm import make_chat_backend
        backend = make_chat_backend(
            provider=llm_provider, model=llm_model,
            base_url=llm_base_url, api_key=llm_api_key,
            strict=True,
        )
    wg = None
    base = None
    if defense:
        base = calibrate_clean_baseline(n_society=min(n_society, 1000))
        wg = WolfGuardAgent(config=WolfGuardConfig())
    env = WolfBenchEnv(scen, n_society=n_society, alpha=alpha, seed=seed,
                       wolfguard=wg, baseline=base,
                       llm_backend=backend, n_llm_leaders=llm_leaders)
    res = env.run()
    summary = _summarise(res)
    summary["n_llm_leaders"] = llm_leaders
    summary["llm_model"] = llm_model
    summary["llm_provider"] = llm_provider
    click.echo(json.dumps(summary, indent=2))
    if out:
        Path(out).write_text(json.dumps(summary, indent=2))


@main.command("attack")
@click.option("--scenario", default="s1", type=click.Choice(list(SCENARIO_FILES)))
@click.option("--alpha", default=0.02, type=float)
@click.option("--n-society", default=1000, type=int)
@click.option("--seeds", default="1,2,3", type=str)
def attack_track(scenario, alpha, n_society, seeds):
    res = run_attack_track(scenario, alpha, n_society, _parse_ints(seeds))
    click.echo(json.dumps({
        "scenario": scenario,
        "alpha": alpha,
        "n_society": n_society,
        "attack_score": res.attack_score,
        "collapse_rate": float(np.mean([e.metrics.collapse_rate for e in res.episodes])),
        "retail_loss_pct": float(np.mean([e.metrics.retail_loss_pct_30d for e in res.episodes])),
        "wealth_transfer": float(np.mean([e.metrics.wealth_transfer for e in res.episodes])),
    }, indent=2))


@main.command("defense")
@click.option("--scenario", default="s1", type=click.Choice(list(SCENARIO_FILES)))
@click.option("--alpha", default=0.02, type=float)
@click.option("--n-society", default=1000, type=int)
@click.option("--seeds", default="1,2,3", type=str)
@click.option("--mode", default="full",
              type=click.Choice(["off", "warning", "cooldown", "block", "full"]))
def defense_track(scenario, alpha, n_society, seeds, mode):
    cfg = WolfGuardConfig(mode=mode)
    res = run_defense_track(scenario, alpha, n_society, _parse_ints(seeds), cfg)
    click.echo(json.dumps({
        "scenario": scenario,
        "alpha": alpha,
        "n_society": n_society,
        "mode": mode,
        "defense_score": res.defense_score,
        "no_def_collapse_rate": float(np.mean([e.metrics.collapse_rate for e in res.episodes_no_def])),
        "def_collapse_rate": float(np.mean([e.metrics.collapse_rate for e in res.episodes_def])),
        "no_def_retail_loss": float(np.mean([e.metrics.retail_loss_pct_30d for e in res.episodes_no_def])),
        "def_retail_loss": float(np.mean([e.metrics.retail_loss_pct_30d for e in res.episodes_def])),
        "utility_loss": float(np.mean([e.metrics.utility_loss for e in res.episodes_def])),
    }, indent=2))


@main.command("scaling")
@click.option("--scenario", default="s1", type=click.Choice(list(SCENARIO_FILES)))
@click.option("--alpha", default="0,0.005,0.01,0.02,0.05,0.1", type=str)
@click.option("--n-society", default="100,1000", type=str)
@click.option("--seeds", default=3, type=int)
@click.option("--defense/--no-defense", default=False)
@click.option("--out", type=click.Path(), default=None)
def scaling_track(scenario, alpha, n_society, seeds, defense, out):
    alphas = _parse_floats(alpha)
    n_grid = _parse_ints(n_society)
    seed_list = list(range(1, seeds + 1))
    res = run_scaling_track(scenario, alphas, n_grid, seed_list,
                            with_defense=defense)
    out_dict = {
        "scenario": scenario,
        "alphas": alphas,
        "n_society": n_grid,
        "seeds": seed_list,
        "with_defense": defense,
        "p_collapse": {f"N={N},a={a}": p for (N, a), p in res.p_collapse.items()},
        "alpha_critical": {str(N): ac for N, ac in res.alpha_critical.items()},
    }
    click.echo(json.dumps(out_dict, indent=2))
    if out:
        Path(out).write_text(json.dumps(out_dict, indent=2))


SPLITS_PATH = Path(__file__).parent / "config" / "splits.yaml"
EVALUATION_SPLITS = ("public_dev", "public_test", "paper_final", "stress")


def _load_splits() -> dict:
    return yaml.safe_load(SPLITS_PATH.read_text())


def _resolve_seeds(split: str | None, seeds_arg: str | None) -> list[int]:
    if seeds_arg:
        return _parse_ints(seeds_arg)
    splits = _load_splits()
    if split not in splits:
        raise click.UsageError(f"Unknown split '{split}'.")
    return list(splits[split]["seeds"])


def _canonical_config() -> dict:
    return _load_splits().get("canonical", {})


def _resolve_scenarios(spec: str) -> list[str]:
    if spec in {"all", "canonical"}:
        return list(_canonical_config().get("scenarios", SCENARIO_FILES.keys()))
    scenarios = [x.strip() for x in spec.split(",") if x.strip()]
    unknown = [s for s in scenarios if s not in SCENARIO_FILES]
    if unknown:
        raise click.UsageError(f"Unknown scenario(s): {', '.join(unknown)}")
    return scenarios


def _resolve_alpha_grid(spec: str) -> list[float]:
    if spec == "canonical":
        return [float(a) for a in _canonical_config().get("alphas", [0.0, 0.02])]
    return _parse_floats(spec)


def _resolve_n_grid(spec: str) -> list[int]:
    if spec == "canonical":
        return [int(n) for n in _canonical_config().get("n_society", [1000])]
    return _parse_ints(spec)


def _import_policy(spec: str, **kwargs):
    """Load a user-supplied policy via 'pkg.mod:ClassName'."""
    import importlib
    mod_name, cls_name = spec.split(":", 1)
    mod = importlib.import_module(mod_name)
    return getattr(mod, cls_name)(**kwargs)


def _policy_kwargs(llm_model=None, llm_provider=None, llm_base_url=None,
                   llm_api_key=None, model_path=None) -> dict:
    kwargs = {
        "model": llm_model,
        "provider": llm_provider,
        "base_url": llm_base_url,
        "api_key": llm_api_key,
        "model_path": model_path,
    }
    return {k: v for k, v in kwargs.items() if v is not None}


def _make_policy(spec: str, kwargs: dict, isolated: bool):
    policy_kwargs = {} if ":" in spec else kwargs
    if isolated:
        return SubprocessWolfGuardPolicy(spec, kwargs=policy_kwargs)
    return _import_policy(spec, **policy_kwargs) if ":" in spec else get_policy(spec, **policy_kwargs)


def _run_one(scenario_id, n_society, alpha, seed, defense_spec, policy_kwargs=None,
             isolated=True):
    scen = load_scenario(scenario_id)
    base = calibrate_clean_baseline(n_society=min(n_society, 1000))
    policy_kwargs = policy_kwargs or {}
    policy = _make_policy(defense_spec, policy_kwargs, isolated=isolated)
    try:
        is_noguard = defense_spec.lower() == "noguard" if ":" not in defense_spec else policy.__class__.__name__ == "NoGuardPolicy"
        wg = None if is_noguard else policy
        env = WolfBenchEnv(
            scen, n_society=n_society, alpha=alpha, seed=seed,
            wolfguard=wg, baseline=base,
            expose_oracle=(defense_spec.lower() == "oracle" if ":" not in defense_spec else False),
        )
        res = env.run()
        row = _summarise(res)
        row["defense"] = getattr(policy, "name", "NoGuard")
    finally:
        if hasattr(policy, "close"):
            policy.close()
    return row


@main.command("evaluate")
@click.option("--defense", "defense_name", default="rule",
            help="Defense baseline: noguard | random | rule | oracle | distilled | llm "
                   "or a dotted spec 'pkg.mod:ClassName'.")
@click.option("--scenario", default="s1", type=click.Choice(list(SCENARIO_FILES)))
@click.option("--alphas", default="0,0.005,0.01,0.02,0.05,0.1", type=str)
@click.option("--n-society", "n_society", default=1000, type=int)
@click.option("--split", default="public_dev",
              type=click.Choice(EVALUATION_SPLITS))
@click.option("--seeds", default=None,
              help="Comma-separated seeds, overrides --split.")
@click.option("--llm-model", default=None,
              help="Used when --defense=llm.")
@click.option("--llm-provider", default=None,
                  type=click.Choice(["openai", "openrouter", "vllm"]),
                  help="Used when --defense=llm; --defense=qwen defaults to vllm.")
@click.option("--llm-base-url", default=None,
                  help="OpenAI-compatible base URL, e.g. http://127.0.0.1:8000/v1 or https://openrouter.ai/api/v1.")
@click.option("--llm-api-key", default=None,
                  help="API key for the OpenAI-compatible endpoint; vLLM accepts EMPTY.")
@click.option("--model-path", default=None, type=click.Path(),
              help="Model artifact path for --defense=distilled.")
@click.option("--out", type=click.Path(), default=None)
@click.option("--isolate/--no-isolate", default=True,
              help="Run defense policy in a spawned subprocess JSON RPC sandbox.")
def evaluate_cmd(defense_name, scenario, alphas, n_society, split, seeds,
                      llm_model, llm_provider, llm_base_url, llm_api_key,
                      model_path, out,
                      isolate):
    """Evaluate a defense submission. Reports DefenseScore + ThresholdShift."""
    alpha_grid = _parse_floats(alphas)
    seed_list = _resolve_seeds(split, seeds)
    kwargs = _policy_kwargs(llm_model, llm_provider, llm_base_url, llm_api_key, model_path)
    if ":" in defense_name:
        display_name = defense_name
    else:
        display_policy = _make_policy(defense_name, kwargs, isolated=False)
        display_name = getattr(display_policy, "name", defense_name)

    rows_no, rows_def = [], []
    episode_grid = [(a, s) for a in alpha_grid for s in seed_list]
    rng = np.random.default_rng(0)
    rng.shuffle(episode_grid)
    for a, s in episode_grid:
        rows_no.append(_run_one(scenario, n_society, a, s, "noguard", isolated=False))
        rows_def.append(_run_one(scenario, n_society, a, s, defense_name,
                                 policy_kwargs=kwargs, isolated=isolate))

    report = {
        "defense": display_name,
        "scenario": scenario,
        "n_society": n_society,
        "alphas": alpha_grid,
        "split": split,
        "seeds": seed_list,
        "isolated": isolate,
        **defense_score(rows_no, rows_def, alphas=alpha_grid),
        **threshold_shift(rows_no, rows_def, alpha_grid),
    }
    click.echo(json.dumps(report, indent=2))
    if out:
        Path(out).write_text(json.dumps(report, indent=2))


@main.command("protocol")
def protocol_cmd():
    """Print the public benchmark protocol and canonical grid."""
    splits = _load_splits()
    report = {
        "splits": {name: splits[name] for name in EVALUATION_SPLITS if name in splits},
        "canonical": splits.get("canonical", {}),
        "hidden_seeds": "server-side only for future official leaderboard audits",
    }
    click.echo(json.dumps(report, indent=2))


@main.command("export-trajectories")
@click.option("--scenario", "scenario_spec", default="s1",
              help="Comma-separated scenario ids, or 'canonical'/'all'.")
@click.option("--alphas", default="0,0.02",
              help="Comma-separated alpha grid, or 'canonical'.")
@click.option("--n-society", "n_society", default="500",
              help="Comma-separated N grid, or 'canonical'.")
@click.option("--split", default="public_dev", type=click.Choice(EVALUATION_SPLITS))
@click.option("--seeds", default=None,
              help="Comma-separated seeds, overrides --split.")
@click.option("--labels-policy", default="auto",
              type=click.Choice(["auto", "include", "hide"]),
              help="auto emits oracle/outcome labels only for training splits.")
@click.option("--out", type=click.Path(), default=None)
def export_trajectories_cmd(scenario_spec, alphas, n_society, split, seeds,
                            labels_policy, out):
    """Export public-observation trajectory JSONL for defense training."""
    out = out or f"paperoutputs/benchmark/trajectory_dataset/{split}.jsonl"
    metadata = export_trajectory_dataset(
        out_path=out,
        scenarios=_resolve_scenarios(scenario_spec),
        alphas=_resolve_alpha_grid(alphas),
        n_society_grid=_resolve_n_grid(n_society),
        seeds=_resolve_seeds(split, seeds),
        split=split,
        labels_policy=labels_policy,
    )
    click.echo(json.dumps(metadata, indent=2))


@main.command("train-distilled")
@click.option("--dataset", type=click.Path(exists=True), required=True,
              help="Labeled public-dev trajectory JSONL.")
@click.option("--dev-dataset", type=click.Path(exists=True), default=None,
              help="Optional labeled trajectory JSONL for offline validation.")
@click.option("--out", type=click.Path(),
              default="paperoutputs/benchmark/distilled_wolfguard/model.json")
@click.option("--epochs", default=300, type=int)
@click.option("--lr", default=0.1, type=float)
@click.option("--l2", default=1e-4, type=float)
@click.option("--seed", default=0, type=int)
@click.option("--class-balance/--no-class-balance", default=True)
def train_distilled_cmd(dataset, dev_dataset, out, epochs, lr, l2, seed, class_balance):
    """Train the simulator-trained Distilled-WolfGuard baseline."""
    records = list(iter_jsonl(dataset))
    model = train_distilled_model(
        records,
        epochs=epochs,
        lr=lr,
        l2=l2,
        seed=seed,
        class_balance=class_balance,
    )
    model.save(out)
    report = {
        "model_path": out,
        "train": model.metadata,
    }
    if dev_dataset:
        report["dev"] = evaluate_distilled_model(model, iter_jsonl(dev_dataset))
    click.echo(json.dumps(report, indent=2))


def _summarise(res) -> dict:
    m = res.metrics
    return {
        "scenario": res.scenario_id,
        "n_society": res.n_society,
        "alpha": res.alpha,
        "seed": res.seed,
        "target_asset": res.target_asset,
        "collapse_rate": m.collapse_rate,
        "collapse_day": m.collapse_day,
        "primary_metric": m.primary_metric,
        "primary_failure_rate": m.primary_failure_rate,
        "primary_failure_day": m.primary_failure_day,
        "primary_failure_score_max": m.primary_failure_score_max,
        "max_collapse_score": m.max_collapse_score,
        "retail_loss_pct_30d": m.retail_loss_pct_30d,
        "harmful_profit": m.harmful_profit,
        "wealth_transfer": m.wealth_transfer,
        "price_dislocation_max": m.price_dislocation_max,
        "liquidity_stress_max": m.liquidity_stress_max,
        "social_cascade_peak": m.social_cascade_peak,
        "wash_share_max": m.wash_share_max,
        "volume_distortion_max": m.volume_distortion_max,
        "volume_signal_z_max": m.volume_signal_z_max,
        "cancel_rate_max": m.cancel_rate_max,
        "spoof_depth_to_liquidity_max": m.spoof_depth_to_liquidity_max,
        "withdrawal_loss_max": m.withdrawal_loss_max,
        "intervention_cost": m.intervention_cost,
        "utility_loss": m.utility_loss,
        "false_positive_rate": m.false_positive_rate,
    }


if __name__ == "__main__":
    main()