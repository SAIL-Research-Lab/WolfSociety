export function SocialDynamics() {
  const base = import.meta.env.BASE_URL

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

      <figure className="paper-figure paper-figure--result">
        <img
          src={`${base}nonlinear-response.png`}
          alt="Collapse probability and episode severity rise with the harmful-agent fraction at size-dependent boundaries."
        />
        <figcaption>
          <strong>Figure 2.</strong> Nonlinear collapse transition in the primary scenario. Collapse probability rises sharply around a size-dependent boundary, while episode severity <em>R</em><sub>S1</sub> provides a continuous view of the same transition.
        </figcaption>
      </figure>
    </section>
  )
}
