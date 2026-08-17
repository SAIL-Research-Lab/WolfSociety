import { useEffect, useState, type CSSProperties } from 'react'
import { scalingPoints } from '../data/results'

type AgentFaceProps = {
  kind: 'normal' | 'devil'
  className?: string
}

function AgentFace({ kind, className = '' }: AgentFaceProps) {
  if (kind === 'devil') {
    return (
      <svg className={`agent-face devil-face ${className}`} viewBox="0 0 36 36" aria-hidden="true">
        <path className="devil-horn" d="M8 10 5 2l9 6M28 10l3-8-9 6" />
        <circle className="devil-head" cx="18" cy="19" r="13" />
        <path className="devil-eye" d="m11 16 4 2m10-2-4 2" />
        <path className="devil-smile" d="M12 23c3 3 9 3 12 0" />
      </svg>
    )
  }

  return (
    <svg className={`agent-face normal-face ${className}`} viewBox="0 0 36 36" aria-hidden="true">
      <circle className="normal-head" cx="18" cy="18" r="13" />
      <circle className="normal-eye" cx="13" cy="16" r="1.4" />
      <circle className="normal-eye" cx="23" cy="16" r="1.4" />
      <path className="normal-smile" d="M13 22c3 2 7 2 10 0" />
    </svg>
  )
}

function SwitchingAgent({ stage }: { stage?: number }) {
  return (
    <span className={stage === undefined ? 'agent-cell' : `agent-cell agent-cell--switch agent-cell--stage-${stage}`}>
      <AgentFace kind="normal" />
      {stage !== undefined && <AgentFace kind="devil" />}
    </span>
  )
}

function SharedState({ compact = false }: { compact?: boolean }) {
  return (
    <div className={compact ? 'shared-state shared-state--compact' : 'shared-state'} aria-hidden="true">
      <span />
      <span />
      <span />
      <i />
    </div>
  )
}

const tippingStages = new Map([
  [2, 1],
  [7, 2],
  [10, 3],
  [13, 4],
  [16, 5],
  [19, 6],
])

function TippingScene() {
  return (
    <article className="motion-card">
      <div className="motion-copy">
        <span>01 · Sudden collapse</span>
        <h3>Collapse can arrive suddenly.</h3>
        <p>Paper finding: harm stays limited at first, then rises sharply after only a small increase in harmful agents.</p>
      </div>
      <div className="tipping-scene" aria-label="Conceptual animation in which normal agents progressively become harmful and the shared state suddenly collapses">
        <div className="tipping-agents">
          {Array.from({ length: 20 }, (_, index) => (
            <SwitchingAgent key={index} stage={tippingStages.get(index)} />
          ))}
        </div>
        <div className="pressure-flow" aria-hidden="true"><span /></div>
        <div className="collapsing-state">
          <SharedState />
          <strong>market</strong>
        </div>
      </div>
      <small className="motion-note">Conceptual illustration</small>
    </article>
  )
}

function FixedSociety({ count, harmful }: { count: number; harmful: number[] }) {
  return (
    <div className="fixed-society" style={{ '--agent-count': count > 12 ? 10 : 5 } as CSSProperties}>
      {Array.from({ length: count }, (_, index) => (
        <AgentFace key={index} kind={harmful.includes(index) ? 'devil' : 'normal'} />
      ))}
    </div>
  )
}

function SizeScene() {
  return (
    <article className="motion-card">
      <div className="motion-copy">
        <span>02 · Society size</span>
        <h3>A larger society can fail at a smaller harmful share.</h3>
        <p>Paper finding: the harmful share needed for collapse falls as the number of agents grows, even though the harmful headcount increases.</p>
      </div>
      <div className="size-scene" aria-label="Conceptual comparison showing a small society and a larger society with a visually smaller harmful share">
        <div className="society-row society-row--small">
          <div>
            <span>smaller society</span>
            <FixedSociety count={10} harmful={[1, 7]} />
          </div>
          <SharedState compact />
        </div>
        <div className="society-row society-row--large">
          <div>
            <span>larger society · lower share</span>
            <FixedSociety count={30} harmful={[2, 11, 20, 27]} />
          </div>
          <SharedState compact />
        </div>
      </div>
      <small className="motion-note">Conceptual comparison</small>
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
      <div className="motion-copy">
        <span>03 · Measured curves</span>
        <h3>The collapse curve moves left as society size grows.</h3>
        <p>Paper finding: the 50% collapse point falls from 4.7% at 100 agents to 2.2% at 2,000 agents.</p>
      </div>
      <div className="curve-scene">
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
      <small className="motion-note">Values shown are measured results</small>
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
