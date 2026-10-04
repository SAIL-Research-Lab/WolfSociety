"""Final manuscript families and their explicit full-LLM replacements.

The grids below are PILOT candidates, not estimates imported from the hybrid
paper. A main manifest requires a separately frozen grid file.
"""
from __future__ import annotations

from copy import deepcopy
from itertools import product

PAPER_SIZES = [100, 200, 300, 500, 1000, 2000]
PILOT_ALPHAS = [0.0, 0.01, 0.03, 0.06, 0.10, 0.20, 0.40]
FAMILIES = {
    "p01": "Main S1 scaling; midpoint, count, width, uncertainty and threshold sensitivity",
    "p02": "Main fixed harmful count; social and liquidity decomposition",
    "p02b": "Appendix boundary-support counts and held-out response prediction",
    "p03": "Main cross-scenario S1–S4 scope, including continuous outcomes",
    "p04": "Main coupling interventions; prompt deliberation replaces numerical QRE precision",
    "p05": "Main information mechanisms and message-channel interventions",
    "p06": "Appendix role homogeneity and clean-population behavioral diversity",
    "p07": "Appendix cross-backbone comparison using separately served local models",
    "p09": "Appendix liquidity-depth scaling",
    "p10": "Appendix controller audit: full LLM, explicit rule control, matched mixed populations",
    "p11": "Appendix Watts threshold null; distinct mathematical control, zero LLM calls",
    "language": "New language-content ablation at matched budgets",
    "controller_factorial": "New benign/harmful LLM-versus-rule factorial",
}
ALIASES = {
    "main": ["p01", "p02", "p03", "p04", "p05"],
    "appendix": ["p02b", "p06", "p07", "p09", "p10", "p11"],
    "llm_controls": ["language", "controller_factorial"],
    "all": list(FAMILIES),
}
FIXED_COUNTS = [0, 1, 3, 5, 8, 12]
BOUNDARY_COUNTS = [0, 2, 4, 8, 16, 32, 64]


def expand_families(values):
    result = []
    for value in values:
        for family in ALIASES.get(value, [value]):
            if family not in FAMILIES:
                raise ValueError(f"Unknown family: {family}")
            if family not in result:
                result.append(family)
    return result


def variants(family):
    """Pairs of publication label and actual, simulation-visible controls."""
    if family == "p02":
        return [(name, {"liquidity_exponent": q, "intervention": iv}) for name, q, iv in [
            ("baseline_q05", .5, "baseline"), ("no_social_q05", .5, "no_social"),
            ("fixed_depth_q0", 0., "baseline"), ("per_capita_depth_q1", 1., "baseline"),
            ("per_capita_no_social_q1", 1., "no_social")]]
    if family == "p04":
        return [("baseline", {}), ("weak_feedback", {"intervention": "weak_feedback"}),
                ("strong_feedback", {"intervention": "strong_feedback"}),
                ("high_conformity", {"intervention": "high_conformity"}),
                ("high_reach", {"intervention": "high_reach"}),
                ("high_attention", {"intervention": "high_attention"}),
                ("high_deliberation", {"analytical_deliberation": "high"}),
                ("no_feedback", {"intervention": "no_feedback"}),
                ("no_multihop", {"intervention": "no_multihop"})]
    if family == "p05":
        return [
            ("full_game", {}), ("private_only", {"intervention": "no_social"}),
            ("content_only", {"scenario_overrides": {"social": {"social_proof_visible": False}}}),
            ("proof_only", {"scenario_overrides": {"social": {"content_visible": False}}}),
            ("low_attention", {"attention_capacity": 2}),
            ("high_attention", {"attention_capacity": 12}),
            ("precise_private_signal", {"scenario_overrides": {"retail": {"private_noise_scale": .5}}}),
            ("noisy_private_signal", {"scenario_overrides": {"retail": {"private_noise_scale": 2.}}}),
            ("sender_return_annotation_off", {"scenario_overrides": {"retail": {"trust_learning_scale": 0.}}}),
            ("shuffled_sender", {"scenario_overrides": {"social": {"shuffle_sender": True}}}),
            ("delayed_messages", {"scenario_overrides": {"social": {"message_delay": 2}}}),
            ("hub_placement", {"placement": "high_degree"}),
        ]
    if family == "p06":
        return [(role, {"role_mode": role}) for role in ["mixed_roles", "all_risk_averse", "all_value", "all_trend", "all_social", "all_aggressive"]] + [
            ("rule_reference", {"controller": "rule"})]
    if family == "p09":
        return [(f"depth_q{q:g}", {"liquidity_exponent": q}) for q in [0., .5, 1.]]
    if family == "p10":
        return [("full_llm", {}), ("all_rule", {"controller": "rule"}),
                ("half_llm", {"controller": "mixed", "llm_fraction": .5})]
    if family == "language":
        return [(mode, {"language_mode": mode}) for mode in ["full", "sentiment_only", "neutral_text"]]
    if family == "controller_factorial":
        return [(f"benign_{b}_harmful_{h}", {"benign_controller": b, "harmful_controller": h})
                for b, h in product(["llm", "rule"], repeat=2)]
    return [("baseline", {})]


def grid_key(family, scenario, variant, n):
    return f"{family}:{scenario}:{variant}:{n}"


def grid_for(grids, family, scenario, variant, n, stage):
    key = grid_key(family, scenario, variant, n)
    if stage == "smoke":
        return [0., .2]
    if grids is None:
        if stage == "main":
            raise ValueError("Main experiments require --grid with frozen=true; run separate pilots first")
        return PILOT_ALPHAS[:]
    entries = grids.get("grids", {})
    if key not in entries:
        raise ValueError(f"Grid missing {key}; pilot/freeze this size and condition explicitly")
    values = sorted(set(float(a) for a in entries[key]))
    if not values or values[0] != 0 or any(not 0 <= a < 1 for a in values):
        raise ValueError(f"Grid {key} must include alpha=0 and values in [0,1)")
    return values


def build_cells(families, *, stage="pilot", sizes=None, seeds=None, grids=None, horizon=30):
    if stage not in {"pilot", "main", "smoke"}:
        raise ValueError(stage)
    if stage == "main" and (not grids or grids.get("frozen") is not True):
        raise ValueError("A main sweep requires an explicitly frozen pilot-derived grid")
    sizes = list(sizes or ([20] if stage == "smoke" else PAPER_SIZES))
    seeds = list(seeds or ([9001] if stage == "smoke" else list(range(1, 13)) if stage == "main" else [1001, 1002, 1003]))
    if len(set(seeds)) != len(seeds) or any(n < 2 for n in sizes):
        raise ValueError("Sizes must be >=2 and seeds unique")
    if stage == "main" and set(seeds) & set(grids.get("pilot_seeds", [])):
        raise ValueError("Pilot and main seed sets must be disjoint")
    cells = []
    restricted = {"p03", "p04", "p05", "p06", "language", "controller_factorial"}
    for family in expand_families(families):
        family_sizes = sizes
        if stage != "smoke" and family in restricted:
            family_sizes = [n for n in sizes if n in (300, 1000)]
            if not family_sizes:
                family_sizes = sizes  # explicit pilot overrides remain usable
        if stage != "smoke" and family == "p07":
            family_sizes = [n for n in sizes if n in (100, 1000)] or sizes
        for n, (variant, changes), scenario in product(family_sizes, variants(family), ["s1", "s2", "s3", "s4"] if family == "p03" else ["s1"]):
            counts = FIXED_COUNTS if family == "p02" else BOUNDARY_COUNTS if family == "p02b" else None
            if counts is not None:
                coordinates = [(k / n, k) for k in counts if k < n and (family != "p02b" or k / n <= .32)]
            else:
                coordinates = [(a, None) for a in grid_for(grids, family, scenario, variant, n, stage)]
            for (alpha, harmful_count), seed in product(coordinates, seeds):
                config = {"scenario": scenario, "n_society": n, "alpha": alpha,
                          "seed": seed, "horizon_days": horizon, "controller": "llm",
                          "role_mode": "mixed_roles", "intervention": "baseline",
                          "language_mode": "full", "attention_capacity": 6,
                          "memory_rounds": 3, "liquidity_exponent": .5}
                config.update(deepcopy(changes))
                if harmful_count is not None:
                    config["harmful_count"] = harmful_count
                cells.append({"family": family, "variant": variant, "config": config,
                              "engine": "watts_null" if family == "p11" else "society"})
        # Clean diversity audit across every N, beyond the two role transition sizes.
        if family == "p06":
            for n, seed in product([n for n in sizes if n not in family_sizes], seeds):
                cells.append({"family": family, "variant": "mixed_roles", "engine": "society",
                              "config": {"scenario": "s1", "n_society": n, "alpha": 0., "seed": seed,
                                         "horizon_days": horizon, "controller": "llm", "role_mode": "mixed_roles",
                                         "intervention": "baseline", "language_mode": "full", "attention_capacity": 6,
                                         "memory_rounds": 3, "liquidity_exponent": .5}})
    if stage == "main" and "protocols" in grids:
        for cell in cells:
            c = cell["config"]
            key = grid_key(cell["family"], c["scenario"], cell["variant"], c["n_society"])
            current = {k: v for k, v in c.items() if k not in {"alpha", "seed", "harmful_count"}}
            if grids["protocols"].get(key) != current:
                raise ValueError(f"Main protocol differs from frozen pilot: {key}. Match horizon/controls and rerun pilots after changes")
    return cells
