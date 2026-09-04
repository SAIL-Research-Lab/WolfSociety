export const site = {
  title: 'WolfSociety',
  subtitle: 'Understanding Collective Risk from Harmful-Agent Scaling in Financial Agent Societies',
  description:
    'We study how harmful-agent fraction, society size, and interaction structure shape collective failure in a controlled financial agent society.',
  abstract: [
    'Safety evaluations usually examine agents one at a time, but interacting agents can spread harmful information and influence the environment seen by later agents. We study this process in a controlled financial society where agents communicate over a social network and trade in a shared market. In the primary scenario, collapse requires broad harmful diffusion together with severe price dislocation or liquidity stress.',
    'At every tested society size, collapse remains rare at low harmful fractions and then rises sharply. As the population grows from 100 to 2,000 agents, the harmful fraction associated with a 50% collapse probability falls from 4.7% to 2.2%, while the corresponding harmful count rises from about 5 to 44. When the harmful count is fixed, its impact becomes weaker as the society grows. Broader network reach also moves collapse toward lower harmful fractions, whereas stronger conformity alone has little effect.',
    'Agent Society Dynamics provides a finite-size description of how harmful-agent fraction, society size, and interaction structure relate to collective failure. The results show that collective risk depends not only on the prevalence of harmful agents, but also on the size and structure of the surrounding society.',
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
    'University of Toronto',
  ],
  authorAffiliations: ['1', '2,3', '2', '1', '1', '1,2'],
  links: {
    paper: './papers/when-harm-scales.pdf',
    code: 'https://github.com/SAIL-Research-Lab/WolfSociety',
  },
}

export const bibtex = `@unpublished{zhang2027wolfsociety,
  title={{WolfSociety}: Understanding Collective Risk from Harmful-Agent Scaling in Financial Agent Societies},
  author={Zhang, Lejun and Lu-Liang, Sarah and Jiang, Xin and Wen, Muning and Zhang, Weinan and Gu, Shangding},
  note={Manuscript under review},
  year={2027}
}`
