import { exponent, interventions, provenance } from '../data/results'

const findings = [
  {
    index: '01',
    title: 'A size-dependent collapse boundary',
    lead: '4.7% → 2.2%',
    detail:
      'As society size grows from 100 to 2,000, the harmful fraction at the 50% collapse boundary falls, while the effective harmful count rises from 4.7 to 44.',
    note: `Boundary exponent ν̂ = ${exponent.estimate.toFixed(3)} · conditional 95% CI [${exponent.confidenceInterval[0].toFixed(3)}, ${exponent.confidenceInterval[1].toFixed(3)}]`,
  },
  {
    index: '02',
    title: 'Dilution below the transition',
    lead: 'bN = −0.894',
    detail:
      'When the harmful count K is held fixed, episode-level peak risk decreases as society size grows. The same direction remains in subcritical and zero-failure cells.',
    note: 'Subcritical response contour νresponse = 0.469 · 95% CI [0.435, 0.501]',
  },
  {
    index: '03',
    title: 'Feedback and reach move the boundary',
    lead: `${interventions.weakFeedback > 0 ? '+' : ''}${interventions.weakFeedback.toFixed(3)} / ${interventions.strongFeedback.toFixed(3)}`,
    detail:
      'Weak feedback shifts the boundary rightward; strong feedback and increased network reach shift it leftward. Social information matters when induced actions feed back into shared state.',
    note: `Mean Δαc: weak feedback +0.046 · strong feedback −0.018 · increased reach ${interventions.increasedReach.toFixed(3)}`,
  },
]

export function ScalingStory() {
  return (
    <section className="section findings-section" id="findings">
      <div className="section-heading section-heading--center">
        <p className="section-label">Core findings</p>
        <h2>Three results define the work.</h2>
        <p>Boundary measurements, an independent fixed-count audit, and matched interventions test complementary parts of the same finite-size account.</p>
      </div>

      <div className="finding-grid">
        {findings.map((finding) => (
          <article className="finding-card" key={finding.index}>
            <span className="finding-index">{finding.index}</span>
            <h3>{finding.title}</h3>
            <strong className="finding-lead">{finding.lead}</strong>
            <p>{finding.detail}</p>
            <small>{finding.note}</small>
          </article>
        ))}
      </div>
      <div className="findings-discussion">
        <p>The three analyses address complementary parts of the same account. Boundary scaling measures how the transition moves with society size, while the fixed-count analysis checks whether harmful impact is diluted below that transition.</p>
        <p>Matched interventions then test the mechanism directly: weakening feedback makes collapse harder, whereas stronger feedback or broader network reach makes the society more fragile.</p>
      </div>
      <p className="provenance">{provenance.label}</p>
    </section>
  )
}
