# Formal experiment registry

| ID | Paper purpose | Central claim | Main output |
|---|---|---|---|
| P00 | integration and alpha=0 sanity | none | validation rows |
| P01 | nonlinear S1 response and finite-size scaling | C1, C2 | alpha curves, alpha_c(N), K_c(N), width |
| P02 | fixed-K social/liquidity decomposition | C2 | size effect under q=0, 0.5, 1 and NoSocial |
| P03 | S1--S4 scope | scope of C1/C2 | resolved/censored scenario summary |
| P04 | game-theoretic coupling phase diagram | C3 | coupling proxy, response width, alpha_c |
| P05 | private-to-social information cascade | C3 | CMI, social dominance, transfer entropy |
| P06 | role robustness and behavioral diversity audit | reviewer validity | role/action MI, behavior separation, diversity-vs-size controls |
| P07 | LLM allocation robustness | implementation validity | matched quota response curves |
| P08 | compact defense utility demonstration | benchmark utility | risk reduction, cost, false positives |
| P09 | depth-specific dense scaling audit | reviewer W1 | observed \(\widehat\nu\) under q=0, 0.5, 1 |
| P10 | LLM-fraction dense scaling audit | reviewer W3 | observed \(\widehat\nu\) under 0x, 1x, 2x quotas |
| P11 | Watts threshold null | reviewer W5 | null-model \(\alpha_c(N)\) and \(\widehat\nu\) |

## Planned follow-up experiments

- P09--P11 are reviewer-response audits. P09 repeats the P01 dense alpha_c(N)
   sweep under fixed-depth (q=0), baseline-depth (q=0.5), and per-capita-depth
   (q=1). P10 repeats the dense sweep under behavioral-only, standard, and
   double LLM quotas. P11 runs a Watts-style threshold null on matched graph
   sizes and alpha grids.

## Paper order

The main text should present P01, P02, and P04/P05 in that order. P03 defines
scope. P06/P07 belong in robustness or the appendix. P08 is not a fourth
scientific contribution.

## Freeze rules

1. A paper profile can run only after its pilot grid brackets the intended
   response region.
2. Never report alpha_c when 50% failure is not bracketed.
3. Never combine different `benchmark_version` values.
4. Always report alpha=0 sanity, critical fraction, and critical harmful count.
5. Bootstrap episode seeds, not agent-day events.
6. Generate P04 theory predictions before inspecting paper-scale P04 outcomes.
7. P05 contrasts and signs are fixed in `analysis/information.py`.
8. P06 must report behavioral diversity as a validity/control variable; role
   names alone are not evidence of heterogeneity.

## Status

- Runners rewritten: P00--P08.
- Analyses rewritten: P01, P04, P05, P06.
- v3 protocol: exploratory until new pilot runs finish.
- Old v1/v2 real outputs: historical only; not valid evidence for the rewritten
  paper.
