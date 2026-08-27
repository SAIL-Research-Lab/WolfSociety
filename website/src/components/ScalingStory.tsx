import { exponent, interventions } from '../data/results'

export function ScalingStory() {
  const base = import.meta.env.BASE_URL

  return (
    <>
      <section className="paper-section-block" id="findings">
        <div className="academic-content">
          <h2>Scaling of Collective Collapse</h2>
          <p>
            Across six measured society sizes, collapse remains rare at low harmful fractions and rises sharply near a size-dependent boundary. Increasing the population from 100 to 2,000 moves the measured midpoint from 4.7% to 2.2%.
          </p>
          <p>
            The corresponding critical harmful count nevertheless grows from 4.7 to 44. Over the tested size range, this combination—a decreasing fraction and an increasing count—indicates sublinear growth in the harmful count and is inconsistent with both constant-fraction and constant-count boundaries.
          </p>
        </div>

        <figure className="paper-figure paper-figure--result">
          <img
            src={`${base}finite-size-scaling.png`}
            alt="Measured critical fraction decreases with society size while the harmful count grows sublinearly."
          />
          <figcaption>
            <strong>Figure 3.</strong> The critical fraction follows α<sub>c</sub>(N) ∝ N<sup>−0.222</sup>, while the harmful count follows K<sub>c</sub>(N) ∝ N<sup>0.778</sup>. The conditional 95% bootstrap confidence interval for the boundary exponent is [{exponent.confidenceInterval[0].toFixed(3)}, {exponent.confidenceInterval[1].toFixed(3)}].
          </figcaption>
        </figure>
      </section>

      <section className="paper-section-block" id="mechanisms">
        <div className="academic-content">
          <h2>Interaction Conditions and Network Reach</h2>
          <p>
            The largest boundary shifts occur when several interaction parameters change together and when network reach increases. Jointly decreasing response precision, conformity, attention capacity, and graph degree moves the boundary toward a larger harmful fraction; jointly increasing them and separately increasing network reach move it toward a smaller fraction.
          </p>
          <p>
            At a fixed harmful count, the maximum joint severity of harmful diffusion and market disruption decreases with population size under fixed, square-root, and per-capita liquidity scaling. High conformity alone changes the collapse boundary little, whereas increased network reach produces a clear leftward shift.
          </p>
        </div>

        <figure className="paper-figure paper-figure--result paper-figure--mechanism">
          <img
            src={`${base}intervention-effects.png`}
            alt="Jointly decreasing four interaction parameters shifts the critical harmful fraction rightward, while jointly increasing them and increasing network reach shift it leftward."
          />
          <figcaption>
            <strong>Figure 4.</strong> Mean paired changes in the critical fraction under matched interventions. Joint decrease increases α<sub>c</sub> by {interventions.jointDecrease.toFixed(3)}; joint increase and increased reach decrease it by {Math.abs(interventions.jointIncrease).toFixed(3)} and {Math.abs(interventions.increasedReach).toFixed(3)}, respectively. The two joint conditions change response precision, conformity, attention capacity, and graph degree together. Error bars show paired-bootstrap 95% confidence intervals; positive values indicate greater robustness.
          </figcaption>
        </figure>
      </section>
    </>
  )
}
