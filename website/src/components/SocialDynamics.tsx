export function SocialDynamics() {
  const base = import.meta.env.BASE_URL

  return (
    <section className="paper-section-block" id="framework">
      <div className="academic-content">
        <h2>Agent Society Dynamics</h2>
        <p>
          The financial society forms a closed loop: agents communicate and trade, their actions change the shared social and market conditions, and later agents respond to those changes. Agent Society Dynamics describes how collective failure varies with harmful-agent fraction, society size, and these repeated interactions.
        </p>
        <p>
          In the framework, initial harmful input scales as αN<sup>δ</sup>, while the response that develops through later interactions scales as N<sup>ζ</sup>. Their combination gives α<sub>c</sub>(N) ∝ N<sup>−ν</sup>, with ν = δ + ζ. This is a compact description of the measured size dependence: the experiments estimate ν, but do not identify δ and ζ as separate causal effects.
        </p>
      </div>

      <figure className="paper-figure paper-figure--result">
        <img
          src={`${base}nonlinear-response.png`}
          alt="Collapse probability and episode severity rise with the harmful-agent fraction at size-dependent boundaries."
        />
        <figcaption>
          <strong>Figure 2.</strong> Nonlinear collapse transition in the primary S1 scenario. Collapse probability rises sharply around a size-dependent boundary, while episode severity <em>R</em><sub>S1</sub> shows the same transition on a continuous scale.
        </figcaption>
      </figure>
    </section>
  )
}
