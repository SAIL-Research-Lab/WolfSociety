# Scenario Calibration Audit

WolfBench scenarios are case-inspired controlled environments. Public claims
need more than a real-world label: each scenario should document its
order-of-magnitude constraints, the YAML parameters that encode them, and the
knobs that remain synthetic stress-test controls.

## Calibration Table

| Scenario | External anchor | WolfBench parameters | Audit target |
|---|---|---|---|
| S1 Pump-and-Dump | Microcap/penny-stock pump campaigns with low liquidity, promotion windows, and post-promotion exits | `asset_2.initial_liquidity`, `promote_days`, `dump_days`, `target_inventory_share`, `bot_amplifier_share` | Collapse persists across liquidity scales and promotion intensities, not only at one microcap depth |
| S2 Finfluencer Scalping | High-centrality finfluencer accounts induce follower copy-trading before coordinated selling | `placement=high_degree`, `post_intensity`, `copy_trust_boost`, retail `beta_social` | Critical behavior weakens under random placement and strengthens under hub placement |
| S3 Spoofing / Layering | Large non-bona-fide displayed depth and fast cancellation alter perceived order-book imbalance | `spoof_size_mult`, `cancel_latency_steps`, retail `beta_imbalance`, `base_spread_bps` | Collapse indicators track cancel rate/depth imbalance and remain visible over spoof-size ranges |
| S4 Wash Trading / Fake Liquidity | Controlled accounts manufacture volume/liquidity signals before withdrawal | `wash_volume_multiplier`, `wash_days`, `withdraw_days`, retail `beta_volume` | Retail loss and collapse are sensitive to volume-as-signal, not only price drift |

## Current v3 audit mapping

The public experiment entry points live under `paper_experiments_v3`:

- `p00_validate` checks S1--S4 integration and the alpha-zero sanity condition.
- `p02_size_decomposition` separates fixed-count social and liquidity channels.
- `p03_cross_scenario` checks whether the qualitative transition is resolved
  or censored across S1--S4.
- `p04_game_phase` and `p05_information_cascade` test feedback and information
  mechanisms rather than relying on labels alone.
- `p06_role_robustness` audits behavioral heterogeneity and role separation.

Run the smoke profile with the mock backend before any external-model pilot:

```bash
PYTHONPATH=src:. python -m paper_experiments_v3.experiments.p00_validate \
  --profile smoke --mock --quota-mode standard
```

The repository does not publish generated audit tables or figures. Reports
should regenerate them from a frozen protocol and retain the local run metadata
with the derived evidence.

## Interpretation Rules

- Treat parameters without strong public measurement as stress/control knobs.
- Report robust regions instead of a single tuned configuration.
- Include component-level metrics, especially `price_dislocation_max`,
  `liquidity_stress_max`, and `social_cascade_peak`, to show which mechanism is
  responsible for collapse.
- Keep S0 clean-market calibration separate from harmful-scenario calibration.
- Distinguish a resolved midpoint from a left- or right-censored grid.
- Treat generated figures as views of episode-level evidence, not as primary
  data.