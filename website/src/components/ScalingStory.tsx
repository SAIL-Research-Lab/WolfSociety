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
            The corresponding harmful count nevertheless grows from 4.7 to 44. This combination—a decreasing fraction and an increasing count—is consistent with sublinear finite-size fragility rather than either a constant harmful fraction or a constant harmful count.
          </p>
        </div>

        <figure className="paper-figure paper-figure--result">
          <img
            src={`${base}finite-size-scaling.png`}
            alt="Measured critical fraction decreases with society size while the effective harmful count grows sublinearly."
          />
          <figcaption>
            <strong>Figure 3.</strong> The critical fraction follows α<sub>c</sub>(N) ∝ N<sup>−0.222</sup>, while the effective count follows K<sub>c</sub>(N) ∝ N<sup>0.778</sup>. The conditional bootstrap interval for the boundary exponent is [{exponent.confidenceInterval[0].toFixed(3)}, {exponent.confidenceInterval[1].toFixed(3)}].
          </figcaption>
        </figure>
      </section>

      <section className="paper-section-block" id="mechanisms">
        <div className="academic-content">
          <h2>Why Does the Boundary Shift?</h2>
          <p>
            Matched interventions test whether the observed size effect is connected to the closed feedback loop rather than population size alone. Weakening feedback moves the collapse boundary toward a larger harmful fraction, whereas stronger coupling and broader network reach move it toward a smaller fraction.
          </p>
          <p>
            The fixed-count analysis provides a complementary check below the transition: when the harmful count is held fixed, peak joint social–market severity decreases with population size. Harm is diluted below the boundary, but feedback can amplify it once the society approaches the critical regime.
          </p>
        </div>

        <figure className="paper-figure paper-figure--result paper-figure--mechanism">
          <img
            src={`${base}intervention-effects.png`}
            alt="Weak feedback shifts the critical harmful fraction rightward, while strong feedback and increased reach shift it leftward."
          />
          <figcaption>
            <strong>Figure 4.</strong> Mean change in the critical fraction under matched interventions. Weak feedback increases α<sub>c</sub> by {interventions.weakFeedback.toFixed(3)}; strong feedback and increased reach decrease it by {Math.abs(interventions.strongFeedback).toFixed(3)} and {Math.abs(interventions.increasedReach).toFixed(3)}, respectively. Positive values indicate greater robustness.
          </figcaption>
        </figure>
      </section>
    </>
  )
}