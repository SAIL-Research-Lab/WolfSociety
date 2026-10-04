"""Rebuild historical source provenance without publishing cached responses.

Run from any directory: python legacy_experiments/build_inventory.py.
This only writes inventory.json, and never moves, deletes or stages files.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "legacy_experiments"
BASELINE_COMMIT = "7be61904cb5cd9dc383bc1b843fc32106267ef14"
SOURCE_SUFFIXES = {".py", ".sh", ".md", ".yaml", ".yml", ".toml", ".txt"}
SKIP_DIRS = {
    ".git", "__pycache__", ".pytest_cache", ".venv", "cache", "outputs",
    "manifests", "generated", "results", "node_modules",
}
BUNDLES = [
    {
        "path": "legacy_experiments/archive",
        "original_path": "archive",
        "action": "moved",
        "reason": "Early scaling and defense exploration, superseded by final v3 protocol.",
        "used_by_final_paper": False,
        "controller": "rule_based_or_bounded_strategic_llm",
    },
    {
        "path": "legacy_experiments/archive2",
        "original_path": "archive2",
        "action": "moved",
        "reason": "Earlier e-series hybrid exploration, not direct final manuscript inputs.",
        "used_by_final_paper": False,
        "controller": "quota_limited_hybrid_or_mock",
    },
    {
        "path": "legacy_experiments/reinforcement_experiments",
        "original_path": "reinforcement_experiments",
        "action": "moved",
        "reason": "Original supplement postprocessing retained for publication provenance.",
        "used_by_final_paper": True,
        "controller": "postprocessing_original_v3_hybrid_results",
    },
    {
        "path": "legacy_experiments/nonpaper",
        "original_path": "paper_experiments_v3/experiments",
        "action": "moved",
        "reason": "P08 defense-utility demonstration is absent from final main and supplement.",
        "used_by_final_paper": False,
        "controller": "quota_limited_hybrid_or_mock",
    },
    {
        "path": "legacy_experiments/p0_causal_reruns_20260821",
        "original_path": "../p0_causal_reruns_20260821",
        "action": "copied_external_original_preserved",
        "reason": "Corrective behavioral-only causal reruns not incorporated into final publication text.",
        "used_by_final_paper": False,
        "controller": "behavioral_only_zero_population_llm_calls",
    },
    {
        "path": "legacy_experiments/closure_validity_audit",
        "original_path": "../tmp/closure_validity_audit",
        "original_script": "../tmp/closure_validity_audit.py",
        "action": "copied_external_original_preserved",
        "reason": "Definition-implied versus empirical response audit used in final supplement.",
        "used_by_final_paper": True,
        "controller": "postprocessing_original_v3_hybrid_results",
    },
    {
        "path": "paper_experiments_v3",
        "original_path": "paper_experiments_v3",
        "action": "retained_compatibility_baseline",
        "reason": "Final publication output paths, theory imports and original runners retained; deprecated for new work.",
        "used_by_final_paper": True,
        "controller": "rule_based_or_quota_limited_hybrid_or_mock",
    },
]


def source_files(path: Path):
    for current, dirs, files in os.walk(path):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            file = Path(current) / name
            if name.startswith(".env") or file.is_symlink():
                continue
            if file.suffix in SOURCE_SUFFIXES or (
                file.suffix == ".json" and ("configs" in file.parts or file.name == "config.json")
            ):
                yield file


def baseline_hash(original_path: str) -> str | None:
    if original_path.startswith("../"):
        return None
    result = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:{original_path}"],
        cwd=ROOT, capture_output=True, check=False,
    )
    return hashlib.sha256(result.stdout).hexdigest() if result.returncode == 0 else None


def artifact_tree_summary(path: Path) -> dict:
    """Count moved historical artifacts without reading model-response bytes."""
    files = size = 0
    for current, _, names in os.walk(path):
        for name in names:
            file = Path(current) / name
            if file.is_file() and not file.is_symlink():
                files += 1
                size += file.stat().st_size
    return {"file_count": files, "size_bytes": size, "contents_hashed": False}


def main() -> None:
    catalog = json.loads((ARCHIVE / "final_paper_experiment_catalog.json").read_text())
    uses: dict[str, list[str]] = {}
    for script in catalog["shared_source_dependencies"]:
        uses.setdefault(script, []).append("shared_publication_runtime")
    for family in catalog["families"]:
        if family["used_by_final_paper"]:
            for script in family["source_scripts"]:
                uses.setdefault(script, []).append(family["id"])
    for analysis in catalog["required_posthoc_analyses"]:
        for script in analysis["source_scripts"]:
            uses.setdefault(script, []).append(analysis["id"])

    records = []
    generated_bundles = []
    for bundle in BUNDLES:
        path = ROOT / bundle["path"]
        for file in source_files(path):
            relative = file.relative_to(path)
            original_path = str(Path(bundle["original_path"]) / relative)
            if bundle.get("original_script") and file.name == "closure_validity_audit.py":
                original_path = bundle["original_script"]
            if bundle["path"].endswith("p0_causal_reruns_20260821") and file.name == "CLAIM_AUDIT.md":
                original_path = "../p0_causal_reruns_20260821/results/analysis/CLAIM_AUDIT.md"
            archived_path = str(file.relative_to(ROOT))
            direct_uses = sorted(set(uses.get(archived_path, [])))
            records.append({
                "original_path": original_path,
                "current_path": archived_path,
                "action": bundle["action"],
                "reason": bundle["reason"],
                "used_by_final_paper": bool(direct_uses),
                "publication_family_dependencies": direct_uses,
                "publication_bundle_used": bundle["used_by_final_paper"],
                "controller": bundle["controller"],
                "is_full_llm_evidence": False,
                "size_bytes": file.stat().st_size,
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                "baseline_git_sha256": baseline_hash(original_path),
            })

        # v3 generated trees are retained at the original location; do not
        # traverse their multi-GB event logs and private model-response cache.
        artifact_dirs = []
        for current, dirs, _ in os.walk(path):
            ignored = [d for d in dirs if d in SKIP_DIRS or (
                bundle["path"].endswith("archive2") and d in {"figures", "tables"}
            )]
            for name in ignored:
                if name in {".git", "node_modules", ".venv"}:
                    continue
                artifact_dirs.append(Path(current) / name)
            dirs[:] = [d for d in dirs if d not in ignored]
        for artifact in sorted(artifact_dirs):
            info = {
                "current_path": str(artifact.relative_to(ROOT)),
                "original_path": str(Path(bundle["original_path"]) / artifact.relative_to(path)),
                "action": bundle["action"],
                "reason": "Preserved local generated/compiled evidence; excluded from Git publication.",
                "used_by_final_paper": bundle["used_by_final_paper"],
                "controller": bundle["controller"],
                "is_full_llm_evidence": False,
                "publish": False,
            }
            if bundle["path"] == "paper_experiments_v3":
                info.update(file_count=None, size_bytes=None, contents_hashed=False,
                            scan_policy="retained_in_place_not_traversed")
            else:
                info.update(artifact_tree_summary(artifact))
            generated_bundles.append(info)

    document = {
        "schema_version": 1,
        "archive_date": "2026-10-04",
        "baseline_git_commit": BASELINE_COMMIT,
        "baseline_git_hash_policy": "null means original source was local/untracked or external",
        "publication_controller_warning": "Historical evidence is NOT full-LLM evidence.",
        "preservation": "Moves preserve complete original directory trees. External audits were copied; originals remain.",
        "catalog": "legacy_experiments/final_paper_experiment_catalog.json",
        "source_files": records,
        "artifact_bundles": generated_bundles,
        "read_only_external_material": [
            "../output/overleaf_source/aaai2027_overleaf_source/",
            "../output/code/wolfbench_aaai2027_code/",
        ],
    }
    destination = ARCHIVE / "inventory.json"
    destination.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {len(records)} source records and {len(generated_bundles)} local artifact bundles.")


if __name__ == "__main__":
    main()
