import { useEffect, useState, type CSSProperties } from 'react'
import { scalingPoints } from '../data/results'

const tippingStages = [
  { share: 2, harm: 7, outcome: 'Limited harm', state: 'stable' },
  { share: 3, harm: 11, outcome: 'Limited harm', state: 'stable' },
  { share: 4, harm: 24, outcome: 'Harm rising', state: 'warning' },
  { share: 5, harm: 92, outcome: 'Collapse', state: 'collapse' },
] as const

function TippingScene() {
  const [activeStage, setActiveStage] = useState(0)

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const timer = window.setInterval(() => {
      setActiveStage((current) => (current + 1) % tippingStages.length)
    }, 1700)
    return () => window.clearInterval(timer)
  }, [])

  const stage = tippingStages[activeStage]

  return (
    <article className="motion-card">
      <div className="finding-demo tipping-demo" aria-label="Illustration showing collective harm jumping after a small increase in harmful-agent share">
        <span className="demo-badge">Illustrative progression</span>
        <div className="demo-readout">
          <span>Harmful-agent share</span>
          <strong>{stage.share.toFixed(1)}%</strong>
          <div className="harm-share-track" aria-hidden="true">
            <i style={{ width: `${stage.share * 12}%` }} />
          </div>
        </div>
        <span className="demo-arrow" aria-hidden="true">→</span>
        <div className={`outcome-readout outcome-readout--${stage.state}`}>
          <span>Collective outcome</span>
          <strong>{stage.outcome}</strong>
          <div className="collective-harm-track" aria-hidden="true">
            <i style={{ width: `${stage.harm}%` }} />
          </div>
          <small>collective harm</small>
        </div>
        <div className="stage-controls" aria-label="Select illustrative harmful-agent share">
          {tippingStages.map((item, index) => (
            <button
              key={item.share}
              type="button"
              className={activeStage === index ? 'is-active' : ''}
              aria-pressed={activeStage === index}
              onClick={() => setActiveStage(index)}
            >
              {item.share}%
            </button>
          ))}
        </div>
      </div>
      <p className="motion-explanation"><strong>Finding 1.</strong> Harm remains limited at first, then jumps after only a small increase in the harmful-agent share.</p>
    </article>
  )
}

function SizeScene() {
  return (
    <article className="motion-card">
      <div className="finding-demo size-demo" aria-label="Measured comparison of collapse points for societies of 100 and 2,000 agents">
        <div className="size-comparison-row size-comparison-row--small">
          <div className="size-name">
            <span>Small society</span>
            <strong>100 agents</strong>
          </div>
          <div className="share-measure">
            <span>Harmful share at 50% collapse</span>
            <div><i style={{ width: '94%' }} /></div>
            <strong>4.7%</strong>
          </div>
          <div className="count-measure"><span>Harmful agents</span><strong>≈5</strong></div>
        </div>
        <div className="size-comparison-row size-comparison-row--large">
          <div className="size-name">
            <span>Large society</span>
            <strong>2,000 agents</strong>
          </div>
          <div className="share-measure">
            <span>Harmful share at 50% collapse</span>
            <div><i style={{ width: '44%' }} /></div>
            <strong>2.2%</strong>
          </div>
          <div className="count-measure"><span>Harmful agents</span><strong>44</strong></div>
        </div>
        <div className="size-summary" aria-label="A smaller harmful share but a larger harmful headcount">
          <span>Harmful share <strong>↓ 53%</strong></span>
          <span>Harmful headcount <strong>↑ about 9×</strong></span>
        </div>
      </div>
      <p className="motion-explanation"><strong>Finding 2.</strong> Larger societies collapse at a smaller harmful percentage, even though more harmful agents are needed in total.</p>
    </article>
  )
}

const curveSizes = [100, 500, 2000] as const
const curveColors: Record<(typeof curveSizes)[number], string> = {
  100: '#d6365c',
  500: '#b58bdd',
  2000: '#4d3ca3',
}

function criticalFraction(n: (typeof curveSizes)[number]) {
  return scalingPoints.find((point) => point.n === n)?.alphaC ?? 0
}

function curvePath(alphaC: number) {
  return Array.from({ length: 51 }, (_, index) => {
    const alpha = index * 0.0014
    const probability = 1 / (1 + Math.exp(-(alpha - alphaC) / 0.0055))
    const x = 22 + (alpha / 0.07) * 210
    const y = 91 - probability * 70
    return `${index === 0 ? 'M' : 'L'}${x.toFixed(1)},${y.toFixed(1)}`
  }).join(' ')
}

function CollapseCurveScene() {
  const [activeSize, setActiveSize] = useState<(typeof curveSizes)[number]>(2000)

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const timer = window.setInterval(() => {
      setActiveSize((current) => curveSizes[(curveSizes.indexOf(current) + 1) % curveSizes.length])
    }, 2200)
    return () => window.clearInterval(timer)
  }, [])

  const activeAlpha = criticalFraction(activeSize)
  const activeX = 22 + (activeAlpha / 0.07) * 210

  return (
    <article className="motion-card motion-card--curves">
      <div className="finding-demo curve-scene">
        <svg viewBox="0 0 250 112" role="img" aria-label="Interactive collapse curves shifting toward a lower harmful fraction as society size increases">
          <line className="curve-axis" x1="22" y1="12" x2="22" y2="91" />
          <line className="curve-axis" x1="22" y1="91" x2="238" y2="91" />
          <line className="curve-midline" x1="22" y1="56" x2="238" y2="56" />
          <text className="curve-axis-label" x="4" y="58">.5</text>
          <text className="curve-axis-label" x="191" y="106">harmful share</text>
          {curveSizes.map((n) => (
            <path
              key={n}
              className={`collapse-curve${activeSize === n ? ' collapse-curve--active' : ''}`}
              d={curvePath(criticalFraction(n))}
              style={{ '--curve-color': curveColors[n] } as CSSProperties}
            />
          ))}
          <line
            className="curve-threshold"
            x1={activeX}
            y1="56"
            x2={activeX}
            y2="91"
            style={{ '--curve-color': curveColors[activeSize] } as CSSProperties}
          />
          <circle className="curve-marker" cx={activeX} cy="56" r="3.2" style={{ '--curve-color': curveColors[activeSize] } as CSSProperties} />
        </svg>
        <div className="curve-controls" aria-label="Select society size">
          {curveSizes.map((n) => (
            <button
              key={n}
              type="button"
              className={activeSize === n ? 'is-active' : ''}
              style={{ '--curve-color': curveColors[n] } as CSSProperties}
              aria-pressed={activeSize === n}
              onClick={() => setActiveSize(n)}
            >
              {n.toLocaleString()} agents
            </button>
          ))}
          <output>50% collapse at <strong>{(activeAlpha * 100).toFixed(1)}%</strong> harmful</output>
        </div>
      </div>
      <p className="motion-explanation"><strong>Finding 3.</strong> The measured collapse curve moves left as society size grows: from 4.7% at 100 agents to 2.2% at 2,000 agents.</p>
    </article>
  )
}

export function AnimatedTeasers() {
  return (
    <section className="motion-section" aria-labelledby="motion-title">
      <div className="compact-heading">
        <h2 id="motion-title">Three findings in motion</h2>
        <p>Simple illustrations of the paper's main results.</p>
      </div>
      <div className="motion-grid">
        <TippingScene />
        <SizeScene />
        <CollapseCurveScene />
      </div>
    </section>
  )
}
