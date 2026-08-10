export const site = {
  title: 'When Harm Scales',
  subtitle: 'Social Dynamics of Nonlinear Collapse in Financial Agent Societies',
  description:
    'This work studies how harmful-agent scaling, social diffusion, and shared-state feedback shape critical transitions in interacting agent societies.',
  abstract:
    'Safety evaluations typically assess agents in isolation, yet interacting agents exchange messages and reshape shared environments, allowing local harm to escalate into society-level failure. We study how collapse varies with harmful-agent fraction and society size N. We introduce Agent Society Dynamics, a framework that relates the collapse boundary to size-dependent harmful pressure and shared-state feedback, and evaluate it through fixed-count analyses and controlled interventions in a case-inspired hybrid social–market simulation. Across society sizes, collapse is rare at low harmful fractions, rises sharply near a size-dependent boundary, and then saturates. As N increases from 100 to 2,000, the 50% collapse boundary falls from 4.7% to 2.2%, while the effective harmful count rises from 4.7 to 44. A power-law fit yields a boundary exponent of 0.222; bootstrap estimates remain between 0 and 1, implying sublinear growth in the effective harmful count. At fixed harmful counts, the maximum joint severity of harmful diffusion and market disruption decreases with society size, revealing subcritical dilution. Weakening feedback shifts the boundary toward higher harmful fractions, whereas stronger feedback or greater network reach shifts it toward lower fractions. Together, these results show that society size and shared-state feedback jointly shape collective failure under the tested protocol.',
  authors: [
    'Lejun Zhang',
    'Muning Wen',
    'Sarah Lu-Liang',
    'Xin Jiang',
    'Weinan Zhang',
    'Shangding Gu',
  ],
  affiliations: [
    'Shanghai Jiao Tong University',
    'University of California, Berkeley',
  ],
  authorAffiliations: [1, 1, 2, 2, 1, 2],
  links: {
    paper: './papers/when-harm-scales.pdf',
    code: 'https://github.com/',
  },
}

export const bibtex = `@article{zhang2027whenharm,
  title={When Harm Scales: Social Dynamics of Nonlinear Collapse in Financial Agent Societies},
  author={Zhang, Lejun and Wen, Muning and Lu-Liang, Sarah and Jiang, Xin and Zhang, Weinan and Gu, Shangding},
  year={2027}
}`
