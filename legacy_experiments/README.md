# Historical experiment archive / 历史实验归档

New research runs use `paper_experiments/`. This directory preserves earlier
exploration, original publication analyses and corrective audits. **No result
here is evidence for a fully LLM-controlled society.** Historical runs used
rule controllers, bounded LLM-quota wrappers or explicit mocks. Preserve their
original labels, benchmark versions and frozen configurations.

新实验入口是 `paper_experiments/`。本目录归档原有探索性实验、论文旧版分析及
后续纠偏审计；**这些旧结果不是全 LLM 社会的实验证据**。不删除原始数据，也
不把历史数值混进新实验。源代码可上传，生成结果、缓存、模型文件和凭据不上传。

| Location | Original location | Purpose and final-paper relationship |
|---|---|---|
| `archive/` | repository `archive/` | Early scaling/defense exploration; not direct final-paper evidence |
| `archive2/` | repository `archive2/` | Earlier e-series controllers and exploration; not direct final-paper evidence |
| `reinforcement_experiments/` | repository `reinforcement_experiments/` | Original v3 postprocessing feeding final supplement sensitivity tables |
| `nonpaper/p08_defense_utility.py` | `paper_experiments_v3/experiments/p08_defense_utility.py` | Defense demonstration absent from the final main paper and supplement |
| `p0_causal_reruns_20260821/` | external sibling directory of that name | Copied corrective audit; 1,776 behavior-only episodes, zero population LLM calls; not incorporated into the final publication text |
| `closure_validity_audit/` | external `../tmp/closure_validity_audit.py` and result directory | Copied postprocessing audit used by final response-surface / no-midpoint analysis |
| `perturbation_experiment_feasibility.md` | external `../output/` report | Optional design, never a completed experiment |

`paper_experiments_v3/` remains at the repository root as a **publication
compatibility baseline** because existing tests import its theory utilities
and publication analyses depend on its output paths. Its runners and figures
are deprecated for new work. The original simulator is recoverable at Git
commit `7be61904cb5cd9dc383bc1b843fc32106267ef14`; current `src/` may change.
The original 9 GB v3 local output/cache tree is retained where it was and is
ignored by Git. External originals and packaged publication source were not
modified.

`final_paper_experiment_catalog.json` maps every retained final main/appendix
experiment family, manuscript anchors, source scripts and data dependencies.
`inventory.json` records original/current paths, archive reasons, publication
use and hashes for publishable source files. Large generated trees are listed
as artifact bundles with file counts and sizes, rather than copying model
responses into a public manifest. `build_inventory.py` regenerates the source
and artifact inventory after an intentional archive change.

## Reproduction boundaries / 复现边界

Do not run the whole archive as an active suite. Earlier `archive` imports can
be resolved with `PYTHONPATH=legacy_experiments:src`; other older scripts may
require their original frozen environment and paths. These are preserved
research records, not promises of compatibility with the new controller.

The moved reinforcement generator locates the repository root and continues
to read local `paper_experiments_v3/outputs`, writing only to its archived
`generated/` folder. Its inputs are listed in the experiment catalog. The
archived P08 module imports the retained v3 runtime explicitly; it is optional
and not selected by the active final-paper registry.

The P0 and closure scripts are copied historical records: original absolute
paths/configuration fingerprints are retained for auditability. Do not alter
their old result claims to describe new experiments. In particular, P0 found
that removing environmental feedback barely moved the joint failure boundary;
its evidence favors multi-hop social propagation over a necessary market
feedback mechanism. The closure audit separates algebraic normalization from
independent response prediction.

No API keys, `.env` files, generated response caches or local model weights
belong in tracked archive source. Ignore rules preserve those files locally
without publishing them. New full-LLM experiments must retain separate
controller/version hashes and freshly generated results.
