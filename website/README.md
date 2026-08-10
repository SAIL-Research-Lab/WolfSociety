# When Harm Scales project website

Interactive academic project page for **When Harm Scales: Social Dynamics of Nonlinear Collapse in Financial Agent Societies**.

The page follows a compact paper-first structure: title and authors, two conceptual animations, abstract, the Agent Society Dynamics framework, three core findings, future work, and a final citation section. The animations communicate nonlinear transition and finite-size fragility without presenting themselves as simulations or statistical charts.

The name **WolfBench** is intentionally reserved for the final future-work section. It describes the working goal of turning the current controlled environment into a broader, better validated evaluation environment; the page does not present it as a finished benchmark.

## Development

- `npm install` installs dependencies.
- `npm run dev` starts the local Vite server.
- `npm run typecheck` checks TypeScript.
- `npm run lint` runs ESLint.
- `npm test` runs canonical-data unit tests.
- `npm run test:e2e` runs desktop/mobile Playwright and axe checks.
- `npm run build` creates the production bundle.
- `npm run preview` previews the production bundle.

## Content configuration

Project title, authors, affiliations, Paper/Code links, and BibTeX are centralized in the site configuration under the source config directory. Replace the temporary code URL before public deployment.

The findings values are centralized in the results module and currently mirror:

- `paper_experiments_v3/figures/generated/table2_scaling_results.csv`
- `paper_experiments_v3/figures/generated/table3_scaling_exponent_bootstrap.csv`
- the P04 intervention summary in the final supplementary manuscript

The mathematical framework displays the manuscript relation $\alpha_c(N)\propto N^{-\nu}$, $K_c(N)\propto N^{1-\nu}$, and $\nu=\delta+\zeta$ directly in HTML.

## Paper assets

Before public deployment, place the final paper at the path configured by the site module:

- `public/papers/when-harm-scales.pdf`

## GitHub Pages

The repository workflow builds the website on pushes to `main`. It injects the project repository name as the Vite base path, then uploads the production artifact to GitHub Pages. For a custom domain, set `VITE_BASE_PATH=/` in the deployment workflow.

## Scientific display policy

- Conceptual animation, mathematical assumptions, and measured findings must remain visually and textually distinct.
- Do not source public claims from `archive/`, `archive2/`, or older manuscript versions.
- Browser animation is not presented as an online multi-agent or LLM simulation.
