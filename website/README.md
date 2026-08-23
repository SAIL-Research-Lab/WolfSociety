# WolfSociety project website

Interactive academic project page for **WolfSociety: Harmful-Agent Scaling and Collective Collapse in Financial Agent Societies**.

The page follows a classic paper-first project-page structure: title and authors, Paper/Code resources, the paper teaser, three compact animations, abstract, Agent Society Dynamics, scaling and intervention figures, the WolfBench release scope, and BibTeX. The research figures carry most of the visual identity; the surrounding interface stays quiet and academic.

**WolfSociety** is the paper title. **WolfBench** names the released simulator, scenarios, evaluation interfaces, Python package, and CLI. The page presents WolfBench as a controlled research environment rather than a production market model.

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

Project title, authors, affiliations, Paper/Code links, and BibTeX are centralized in the site configuration under the source config directory.

The findings values are centralized in the results module and currently mirror:

- `paper_experiments_v3/figures/generated/table2_scaling_results.csv`
- `paper_experiments_v3/figures/generated/table3_scaling_exponent_bootstrap.csv`
- the interaction-condition intervention summary in the final manuscript

The mathematical framework displays the manuscript relation $\alpha_c(N)\propto N^{-\nu}$, $K_c(N)\propto N^{1-\nu}$, and $\nu=\delta+\zeta$ directly in HTML.

## Paper assets

The public project-page assets are:

- `public/teaser.png`
- `public/finite-size-scaling.png`
- `public/intervention-effects.png`
- `public/papers/when-harm-scales.pdf`

## GitHub Pages

The repository workflow builds the website on pushes to `main`. It injects the project repository name as the Vite base path, then uploads the production artifact to GitHub Pages. For a custom domain, set `VITE_BASE_PATH=/` in the deployment workflow.

## Scientific display policy

- Conceptual animation, mathematical assumptions, and measured findings must remain visually and textually distinct.
- Do not source public claims from `archive/`, `archive2/`, or older manuscript versions.
- Browser animation is not presented as an online multi-agent or LLM simulation.
