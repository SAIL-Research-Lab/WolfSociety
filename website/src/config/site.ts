export const site = {
  title: 'WolfSociety',
  subtitle: 'How Harmful-Agent Scaling Drives Collective Collapse in Financial Agent Societies?',
  description:
    'This work studies how harmful-agent composition, society size, and interaction structure shape collective collapse in a controlled financial agent society.',
  abstract: [
    'Safety evaluations typically focus on individual agents, but interacting agents can spread harmful information and influence the environment in which later decisions are made. We study how collective failure changes with harmful-agent fraction and society size in a controlled financial agent society, where agents communicate over a social network and trade in a shared market. In the primary scenario, collective failure requires broad harmful diffusion together with severe price dislocation or liquidity stress.',
    'Across society sizes, failure remains rare at low harmful fractions but rises sharply over a narrow range. As the society grows from 100 to 2,000 agents, the harmful fraction associated with a 50% failure probability decreases from 4.7% to 2.2%, while the corresponding number of harmful agents increases from approximately 5 to 44. At a fixed harmful count, larger societies experience less severe disruption, and this pattern persists under fixed, square-root, and per-capita liquidity scaling. Controlled interventions further show that broader network reach shifts the collapse boundary toward lower harmful fractions, whereas stronger conformity alone has little effect.',
    'To characterize these size-dependent effects, we introduce Agent Society Dynamics, a finite-size framework for relating harmful-agent composition, society size, and interaction structure to collective failure. Together, our results show that collective safety cannot be inferred from harmful-agent prevalence alone: the same harmful minority can have different system-level effects as the surrounding society grows.',
  ],
  authors: [
    'Lejun Zhang',
    'Sarah Lu-Liang',
    'Xin Jiang',
    'Muning Wen',
    'Weinan Zhang',
    'Shangding Gu',
  ],
  affiliations: [
    'Shanghai Jiao Tong University',
    'University of California, Berkeley',
    'University of Toronto, Canada',
  ],
  authorAffiliations: ['1', '2,3', '2', '1', '1', '2'],
  links: {
    paper: './papers/when-harm-scales.pdf',
    code: 'https://github.com/SafeRL-Lab/WolfSociety',
  },
}

export const bibtex = `@unpublished{zhang2027wolfsociety,
  title={{WolfSociety}: Harmful-Agent Scaling and Collective Collapse in Financial Agent Societies},
  author={Zhang, Lejun and Lu-Liang, Sarah and Jiang, Xin and Wen, Muning and Zhang, Weinan and Gu, Shangding},
  note={Manuscript under review},
  year={2027}
}`
