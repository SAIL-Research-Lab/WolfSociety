"""Refine pilot grids and freeze the inspected grid before independent main seeds.

Refinement only proposes a new PILOT grid. Main grids are never adapted using
main outcomes. All grid provenance and unresolved cells remain recorded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .analysis import crossing, curve, groups, read_runs
from .registry import grid_key
from .runner import atomic_json, digest


def derive_grid(paths, *, freeze=False, allow_censored=False, allow_mock=False):
    rows, manifests, coverage = read_runs(paths, allow_mock)
    if not rows or any(m["stage"] not in {"pilot", "smoke"} for m in manifests):
        raise ValueError("Only separate pilot episodes can define a grid")
    if freeze and any(c["missing"] for c in coverage):
        raise ValueError("Complete every pilot manifest job before freezing its grid")
    if len({r["simulated"] for r in rows}) > 1:
        raise ValueError("Cannot combine mock and real pilot data")
    if freeze and rows[0]["simulated"] and not allow_mock:
        raise ValueError("Mock pilot grids cannot define scientific main experiments")
    grids, diagnostics, protocols = {}, [], {}
    for key, members in groups(rows).items():
        family, variant, scenario, _, _, _, _, n = key
        name = grid_key(family, scenario, variant, n)
        members_protocol = [{k: v for k, v in row["config"].items() if k not in {"alpha", "seed", "harmful_count"}} for row in members]
        if any(p != members_protocol[0] for p in members_protocol):
            raise ValueError(f"Pilot protocol changed inside {name}")
        protocols[name] = members_protocol[0]
        if family in {"p02", "p02b"}:
            continue
        points = curve(members)
        boundary = crossing(points)
        values = sorted({p[0] for p in points} | {0.})
        if freeze:
            if boundary["status"] != "resolved" and not allow_censored:
                raise ValueError(f"{name} is {boundary['status']}; refine and rerun pilot or explicitly --allow-censored (recorded)")
        elif boundary["status"] == "resolved":
            lo, hi = boundary["bracket_low"], boundary["bracket_high"]
            values.extend(lo + (hi - lo) * i / 4 for i in (1, 2, 3))
        elif boundary["status"] == "right_censored":
            values.extend([min(.8, max(values) * 1.5), min(.9, max(values) * 2)])
        elif boundary["status"] == "left_censored":
            # Baseline failure cannot be repaired by increasing attacker fraction.
            diagnostics.append({"grid_key": name, "action": "inspect clean-market validity before interpreting an attack threshold"})
        else:
            values.extend((a[0] + b[0]) / 2 for a, b in zip(points, points[1:]))
        values = sorted({round(v, 12) for v in values if 0 <= v < 1})
        if name in grids and grids[name] != values:
            raise ValueError(f"Different pilot grids for {name}; freeze backbones separately or provide one matched protocol")
        grids[name] = values
        diagnostics.append({"grid_key": name, **boundary})
    return {"schema": "full-llm-pilot-grid-v1", "frozen": freeze,
            "simulated": rows[0]["simulated"], "allow_censored": allow_censored,
            "pilot_seeds": sorted({r["config"]["seed"] for r in rows}),
            "pilot_manifest_hashes": [m["manifest_hash"] for m in manifests],
            "pilot_source_hashes": sorted({m["source_hash"] for m in manifests}),
            "pilot_backend_fingerprints": sorted({digest(m["backend"]) for m in manifests}),
            "pilot_backend_settings": list({digest(m["backend"]): m["backend"] for m in manifests}.values()),
            "grids": grids, "protocols": protocols, "diagnostics": diagnostics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["refine", "freeze"])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-censored", action="store_true", help="Record unresolved brackets explicitly; analysis still reports censoring")
    parser.add_argument("--allow-mock", action="store_true", help="QA only; such grids cannot define real main manifests")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Grid output already exists; choose a new versioned path")
    grid = derive_grid(args.runs, freeze=args.command == "freeze", allow_censored=args.allow_censored, allow_mock=args.allow_mock)
    atomic_json(args.output, grid)
    print(json.dumps({"output": str(args.output), "frozen": grid["frozen"], "grids": len(grid["grids"]), "simulated": grid["simulated"]}, indent=2))


if __name__ == "__main__":
    main()
