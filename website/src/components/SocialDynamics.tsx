export function SocialDynamics() {
  return (
    <section className="paper-section-block" id="framework">
      <div className="academic-content">
        <h2>Agent Society Dynamics</h2>
        <p>
          We model collective failure as a closed-loop process. A harmful minority contributes pressure through communication and action; heterogeneous agents combine that pressure with private and social evidence; their responses then alter the shared social and market state observed by the rest of the population.
        </p>
        <p>
          The framework separates attack aggregation, which determines how harmful pressure grows with society size, from society susceptibility, which captures amplification through shared-state feedback. Their sum predicts how the critical harmful fraction moves across population sizes.
        </p>
      </div>

      <figure className="paper-figure theory-figure">
        <div className="theory-flow" aria-label="Harmful pressure passes through heterogeneous decisions and shared-state feedback to create a collective response">
          <div className="theory-step">
            <span className="theory-icon theory-icon--harm" aria-hidden="true">♠</span>
            <strong>Harmful pressure</strong>
            <small>αN<sup>δ</sup></small>
          </div>
          <span className="theory-arrow" aria-hidden="true">→</span>
          <div className="theory-step">
            <span className="theory-icon" aria-hidden="true">◎</span>
            <strong>Agent response</strong>
            <small>private + social evidence</small>
          </div>
          <span className="theory-arrow" aria-hidden="true">→</span>
          <div className="theory-step theory-step--feedback">
            <span className="theory-icon" aria-hidden="true">↻</span>
            <strong>Shared-state feedback</strong>
            <small>N<sup>ζ</sup></small>
          </div>
          <span className="theory-arrow" aria-hidden="true">→</span>
          <div className="theory-step">
            <span className="theory-icon theory-icon--society" aria-hidden="true">●●●</span>
            <strong>Collective response</strong>
            <small>ΔR<sub>N</sub></small>
          </div>
        </div>
        <div className="theory-equation" aria-label="Critical fraction scales as society size to the negative nu">
          <span>ΔR<sub>N</sub> ∼ αN<sup>δ+ζ</sup></span>
          <span>α<sub>c</sub>(N) ∝ N<sup>−ν</sup></span>
          <span>ν = δ + ζ</span>
        </div>
        <figcaption>
          <strong>Figure 2.</strong> Agent Society Dynamics decomposes finite-size fragility into harmful-pressure aggregation (<em>δ</em>) and amplification by shared-state feedback (<em>ζ</em>). Collapse occurs when their combined response reaches a protocol-specific boundary.
        </figcaption>
      </figure>
    </section>
  )
}