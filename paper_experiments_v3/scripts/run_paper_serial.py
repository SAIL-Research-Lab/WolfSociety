"""Run v3 paper experiments one seed shard at a time.

This wrapper is intentionally conservative: each child process receives exactly
one paper seed through WOLFBENCH_PAPER_SEEDS. That keeps P05 from accumulating
all seeds of detailed decision/message/exposure logs in one Python process.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "paper_experiments_v3" / "outputs"
LOGS = OUTPUTS / "_runner_logs"


@dataclass(frozen=True)
class Experiment:
    key: str
    module: str
    out_prefix: str
    expected_rows_per_seed: int
    paper_ready: bool = True


EXPERIMENTS: dict[str, Experiment] = {
    "p01": Experiment(
        key="p01",
        module="paper_experiments_v3.experiments.p01_nonlinear_scaling",
        out_prefix="p01_nonlinear_scaling_paper",
        expected_rows_per_seed=42,
    ),
    "p02": Experiment(
        key="p02",
        module="paper_experiments_v3.experiments.p02_size_decomposition",
        out_prefix="p02_size_decomposition_paper",
        expected_rows_per_seed=150,
    ),
    "p03": Experiment(
        key="p03",
        module="paper_experiments_v3.experiments.p03_cross_scenario",
        out_prefix="p03_cross_scenario_paper",
        expected_rows_per_seed=46,
    ),
    "p04": Experiment(
        key="p04",
        module="paper_experiments_v3.experiments.p04_game_phase",
        out_prefix="p04_game_phase_paper",
        expected_rows_per_seed=128,
    ),
    "p05": Experiment(
        key="p05",
        module="paper_experiments_v3.experiments.p05_information_cascade",
        out_prefix="p05_information_cascade_paper",
        expected_rows_per_seed=192,
    ),
    "p06": Experiment(
        key="p06",
        module="paper_experiments_v3.experiments.p06_role_robustness",
        out_prefix="p06_role_robustness_paper",
        expected_rows_per_seed=116,
    ),
    "p07": Experiment(
        key="p07",
        module="paper_experiments_v3.experiments.p07_llm_robustness",
        out_prefix="p07_llm_robustness_paper",
        expected_rows_per_seed=64,
    ),
    "p09": Experiment(
        key="p09",
        module="paper_experiments_v3.experiments.p09_depth_scaling",
        out_prefix="p09_depth_scaling_paper",
        expected_rows_per_seed=126,
    ),
    "p10": Experiment(
        key="p10",
        module="paper_experiments_v3.experiments.p10_llm_fraction_scaling",
        out_prefix="p10_llm_fraction_scaling_paper",
        expected_rows_per_seed=126,
    ),
    "p11": Experiment(
        key="p11",
        module="paper_experiments_v3.experiments.p11_watts_null",
        out_prefix="p11_watts_null_paper",
        expected_rows_per_seed=42,
    ),
}

ALIASES = {
    "remaining": ["p02", "p03", "p04", "p05", "p06"],
    "core": ["p02", "p03", "p04", "p05"],
    "appendix": ["p06", "p07"],
    "reviewer_p0": ["p09", "p10", "p11"],
    "all": ["p01", "p02", "p03", "p04", "p05", "p06", "p07", "p09", "p10", "p11"],
}


def parse_seeds(raw: str) -> list[int]:
    seeds: list[int] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            seeds.extend(range(int(start), int(end) + 1))
        else:
            seeds.append(int(part))
    return sorted(dict.fromkeys(seeds))


def expand_experiments(raw: list[str]) -> list[Experiment]:
    keys: list[str] = []
    for item in raw:
        if item in ALIASES:
            keys.extend(ALIASES[item])
        elif item in EXPERIMENTS:
            keys.append(item)
        else:
            raise SystemExit(f"Unknown experiment or alias: {item}")
    return [EXPERIMENTS[key] for key in dict.fromkeys(keys)]


def shard_name(exp: Experiment, seed: int) -> str:
    return f"{exp.out_prefix}_s{seed}"


def read_status(exp: Experiment, seed: int) -> dict[str, int | str] | None:
    out_dir = OUTPUTS / shard_name(exp, seed)
    status_path = out_dir / "run_status.json"
    if not status_path.exists():
        rows = _read_partial_rows(out_dir)
        if not rows:
            return None
        return {
            "n_ok": sum(row.get("status", "ok") == "ok" for row in rows),
            "n_error": sum(row.get("status", "ok") != "ok" for row in rows),
            "n_rows": len(rows),
        }
    with status_path.open() as handle:
        status = json.load(handle)
    return {
        "n_ok": int(status.get("n_ok", 0)),
        "n_error": int(status.get("n_error", 0)),
        "n_rows": int(status.get("n_rows", 0)),
    }


def _read_partial_rows(out_dir: Path) -> list[dict[str, str]]:
    rows_jsonl = out_dir / "rows.jsonl"
    if rows_jsonl.exists():
        rows: list[dict[str, str]] = []
        with rows_jsonl.open() as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    data_csv = out_dir / "data.csv"
    if data_csv.exists():
        with data_csv.open(newline="") as handle:
            return list(csv.DictReader(handle))
    return []


def is_complete(exp: Experiment, seed: int) -> bool:
    status = read_status(exp, seed)
    if status is None:
        return False
    return (
        status["n_error"] == 0
        and status["n_rows"] == exp.expected_rows_per_seed
        and status["n_ok"] == exp.expected_rows_per_seed
    )


def print_status(experiments: list[Experiment], seeds: list[int]) -> None:
    key_loaded = bool(os.getenv("OPENROUTER_API_KEY"))
    print(f"[env] OPENROUTER_API_KEY={'loaded' if key_loaded else 'missing'}")
    for exp in experiments:
        print(f"\n[{exp.key}] expected_rows_per_seed={exp.expected_rows_per_seed}")
        for seed in seeds:
            status = read_status(exp, seed)
            if status is None:
                print(f"  seed={seed:02d} missing")
            else:
                state = "complete" if is_complete(exp, seed) else "incomplete"
                print(
                    f"  seed={seed:02d} {state} "
                    f"ok={status['n_ok']} err={status['n_error']} rows={status['n_rows']}"
                )


def run_one(
    exp: Experiment,
    seed: int,
    *,
    model: str | None,
    quota_mode: str,
    mock: bool,
    force: bool,
    dry_run: bool,
    stream_output: bool = True,
) -> int:
    out = shard_name(exp, seed)
    if not force and is_complete(exp, seed):
        print(f"[skip] {exp.key} seed={seed} already complete as {out}", flush=True)
        return 0

    env = os.environ.copy()
    for key in [
        "WOLFBENCH_PAPER_N",
        "WOLFBENCH_PAPER_ALPHAS",
        "WOLFBENCH_PAPER_ALPHAS_S1",
        "WOLFBENCH_PAPER_ALPHAS_S2",
        "WOLFBENCH_PAPER_ALPHAS_S3",
        "WOLFBENCH_PAPER_ALPHAS_S4",
        "WOLFBENCH_PAPER_QUOTA",
    ]:
        env.pop(key, None)
    env["PYTHONPATH"] = f"{ROOT / 'src'}:{ROOT}:{env.get('PYTHONPATH', '')}"
    env["WOLFBENCH_PAPER_SEEDS"] = str(seed)
    if model:
        env["WOLFBENCH_PAPER_MODEL"] = model

    cmd = [
        sys.executable,
        "-m",
        exp.module,
        "--profile",
        "paper",
        "--out",
        out,
        "--quota-mode",
        quota_mode,
        "--continue-on-error",
    ]
    if mock:
        cmd.append("--mock")

    print(f"[run] {exp.key} seed={seed} out={out}", flush=True)
    print("      " + " ".join(cmd), flush=True)
    if dry_run:
        return 0

    if stream_output:
        completed = subprocess.run(cmd, cwd=ROOT, env=env, check=False)
    else:
        LOGS.mkdir(parents=True, exist_ok=True)
        log_path = LOGS / f"{out}.log"
        with log_path.open("w") as log:
            log.write("[cmd] " + " ".join(cmd) + "\n")
            log.flush()
            completed = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        print(f"[log] {exp.key} seed={seed} {log_path}", flush=True)
    status = read_status(exp, seed)
    if status is not None:
        print(
            f"[done] {exp.key} seed={seed} "
            f"ok={status['n_ok']} err={status['n_error']} rows={status['n_rows']}",
            flush=True,
        )
    else:
        print(f"[done] {exp.key} seed={seed} no run_status.json written", flush=True)
    return completed.returncode


def run_tasks(
    experiments: list[Experiment],
    seeds: list[int],
    *,
    jobs: int,
    model: str | None,
    quota_mode: str,
    mock: bool,
    force: bool,
    dry_run: bool,
) -> int:
    if not mock:
        key_loaded = bool(os.getenv("OPENROUTER_API_KEY"))
        print(f"[env] OPENROUTER_API_KEY={'loaded' if key_loaded else 'missing'}", flush=True)
        if not key_loaded:
            print(
                "[warn] Real OpenRouter calls will fail. Put OPENROUTER_API_KEY=... in .env "
                "or export it in this shell.",
                flush=True,
            )
    tasks = [(exp, seed) for exp in experiments for seed in seeds]
    if jobs <= 1:
        worst = 0
        for exp, seed in tasks:
            worst = max(
                worst,
                run_one(
                    exp,
                    seed,
                    model=model,
                    quota_mode=quota_mode,
                    mock=mock,
                    force=force,
                    dry_run=dry_run,
                    stream_output=True,
                ),
            )
        return worst

    print(f"[parallel] jobs={jobs} tasks={len(tasks)}", flush=True)
    worst = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {
            executor.submit(
                run_one,
                exp,
                seed,
                model=model,
                quota_mode=quota_mode,
                mock=mock,
                force=force,
                dry_run=dry_run,
                stream_output=False,
            ): (exp, seed)
            for exp, seed in tasks
        }
        for future in concurrent.futures.as_completed(futures):
            exp, seed = futures[future]
            try:
                code = future.result()
            except Exception as exc:
                code = 1
                print(f"[failed] {exp.key} seed={seed} wrapper_error={exc!r}", flush=True)
            worst = max(worst, code)
    return worst


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiments",
        nargs="+",
        help="Experiment keys p02/p03/p04/p05/p06/p07 or aliases remaining/core/appendix/all.",
    )
    parser.add_argument("--seeds", default="1-12", help="Comma/range list, e.g. 1-12 or 5,7,8.")
    parser.add_argument("--quota-mode", default="standard")
    parser.add_argument("--model", default=None)
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun even if the shard is complete.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--status", action="store_true", help="Only print shard status.")
    parser.add_argument("--jobs", type=int, default=1, help="Number of seed shards to run concurrently.")
    args = parser.parse_args()

    experiments = expand_experiments(args.experiments)
    seeds = parse_seeds(args.seeds)

    if args.status:
        print_status(experiments, seeds)
        return 0

    return run_tasks(
        experiments,
        seeds,
        jobs=args.jobs,
        model=args.model,
        quota_mode=args.quota_mode,
        mock=args.mock,
        force=args.force,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
