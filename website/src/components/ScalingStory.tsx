import { exponent, interventions } from '../data/results'

export function ScalingStory() {
  const base = import.meta.env.BASE_URL

  return (
    <>
      <section className="paper-section-block" id="findings">
        <div className="academic-content">
          <h2>Scaling of Collective Collapse</h2>
          <p>
            Across six society sizes, collapse remains rare at low harmful fractions and rises sharply near a size-dependent boundary. We define this boundary as the harmful fraction at which collapse probability reaches 50%. Increasing the population from 100 to 2,000 moves it from 4.7% to 2.2%.
          </p>
          <p>
            The corresponding harmful count nevertheless grows from 4.7 to 44. The harmful count is therefore neither constant nor proportional to society size: it grows, but more slowly than the society itself.
          </p>
        </div>

        <figure className="paper-figure paper-figure--result">
          <img
            src={`${base}finite-size-scaling.png`}
            alt="Measured critical fraction decreases with society size while the harmful count grows sublinearly."
          />
          <figcaption>
            <strong>Figure 3.</strong> The six-size fit gives α<sub>c</sub>(N) ∝ N<sup>−{exponent.estimate.toFixed(3)}</sup>. The corresponding harmful count follows K<sub>c</sub>(N) ∝ N<sup>{(1 - exponent.estimate).toFixed(3)}</sup>: the fraction falls while the count grows.
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
            When the harmful count is fixed, episode severity decreases as the society grows. We rerun this experiment under three market designs—holding total liquidity fixed, growing it with the square root of society size, or growing it in direct proportion to society size—and observe the same decline in all three. High conformity alone changes the collapse boundary little, whereas increased network reach produces a clear shift toward lower harmful fractions.
          </p>
        </div>

        <figure className="paper-figure paper-figure--result paper-figure--mechanism">
          <img
            src={`${base}intervention-effects.png`}
            alt="Jointly decreasing four interaction parameters shifts the critical harmful fraction rightward, while jointly increasing them and increasing network reach shift it leftward."
          />
          <figcaption>
            <strong>Figure 4.</strong> Mean paired changes in the collapse boundary under matched interventions. Joint decrease raises α<sub>c</sub> by {interventions.jointDecrease.toFixed(3)}; joint increase and increased reach lower it by {Math.abs(interventions.jointIncrease).toFixed(3)} and {Math.abs(interventions.increasedReach).toFixed(3)}, respectively. The joint conditions change response precision, conformity, attention capacity, and graph degree together. Error bars show paired-bootstrap 95% confidence intervals; positive values mean that a higher harmful fraction is required for collapse.
          </figcaption>
        </figure>
      </section>
    </>
  )
}
