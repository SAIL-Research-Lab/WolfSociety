export const site = {
  title: 'When Harm Scales',
  subtitle: 'Social Dynamics of Nonlinear Collapse in Financial Agent Societies',
  description:
    'This work studies how harmful-agent scaling, social diffusion, and shared-state feedback shape critical transitions in interacting agent societies.',
  abstract: [
    'Safety evaluations typically assess agents in isolation. Interacting agents, however, exchange messages and reshape shared environments, allowing local harmful behavior to propagate through social evidence, collective action, and market feedback. We study how society-level collapse varies with the harmful-agent fraction and population size in a controlled hybrid social–market environment.',
    'We introduce Agent Society Dynamics, which separates size-dependent harmful pressure from amplification through shared-state feedback. Collapse is rare below a size-dependent boundary and rises sharply near it. As society size grows from 100 to 2,000, the measured 50% collapse boundary falls from 4.7% to 2.2%, while the effective harmful count rises from 4.7 to 44. Fixed-count analyses and matched interventions further show subcritical dilution and identify feedback and network reach as mechanisms that move the boundary.',
  ],
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
    supplement: './papers/when-harm-scales-supplement.pdf',
    code: 'https://github.com/zhanglejun02/when-harm-scales',
  },
}

export const bibtex = `@article{zhang2027whenharm,
  title={When Harm Scales: Social Dynamics of Nonlinear Collapse in Financial Agent Societies},
  author={Zhang, Lejun and Wen, Muning and Lu-Liang, Sarah and Jiang, Xin and Zhang, Weinan and Gu, Shangding},
  year={2027}
}`
