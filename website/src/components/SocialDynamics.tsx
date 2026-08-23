export function SocialDynamics() {
  return (
    <section className="paper-section-block" id="framework">
      <div className="academic-content">
        <h2>Agent Society Dynamics</h2>
        <p>
          We model collective failure as a closed-loop process. A harmful minority contributes pressure through communication and action; heterogeneous agents combine that pressure with private and social evidence; their responses then alter the shared social and market state observed by the rest of the population.
        </p>
        <p>
          The framework connects the collapse boundary to two size-dependent components: initial harmful input, represented by αN<sup>δ</sup>, and the response that develops through subsequent interactions, represented by N<sup>ζ</sup>. Under a comparable collapse criterion across society sizes, their combined scaling gives ν = δ + ζ. We use this relation as a conceptual description; the experiments estimate the overall boundary exponent rather than identify δ and ζ as separate causal effects.
        </p>
      </div>

      <figure className="paper-figure theory-figure">
        <div className="theory-flow" aria-label="Harmful input passes through heterogeneous decisions and repeated interaction to create a collective response">
          <div className="theory-step">
            <span className="theory-icon theory-icon--harm" aria-hidden="true">♠</span>
            <strong>Harmful input</strong>
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
            <strong>Repeated interaction</strong>
            <small>N<sup>ζ</sup></small>
          </div>
          <span className="theory-arrow" aria-hidden="true">→</span>
          <div className="theory-step">
            <span className="theory-icon theory-icon--society" aria-hidden="true">●●●</span>
            <strong>Collective response</strong>
            <small>ΔR<sub>N</sub></small>
          </div>
        </div>
        <div className="theory-equation" aria-label="The critical fraction scales as society size raised to the power of negative nu">
          <span>ΔR<sub>N</sub> ∼ αN<sup>δ+ζ</sup></span>
          <span>α<sub>c</sub>(N) ∝ N<sup>−ν</sup></span>
          <span>ν = δ + ζ</span>
        </div>
        <figcaption>
          <strong>Figure 2.</strong> Agent Society Dynamics relates the collapse boundary to the scaling of initial harmful input (<em>δ</em>) and the response that develops through subsequent interactions (<em>ζ</em>). Under a comparable collapse criterion across society sizes, their combined scaling determines how the critical harmful fraction changes with society size.
        </figcaption>
      </figure>
    </section>
  )
}