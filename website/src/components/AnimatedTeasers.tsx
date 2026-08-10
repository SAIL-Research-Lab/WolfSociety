import type { CSSProperties } from 'react'

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
        <span>01 · Nonlinear transition</span>
        <h3>More harmful agents, then sudden collapse.</h3>
        <p>Pressure can accumulate quietly before shared social and market state fails abruptly.</p>
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
          <strong>shared state</strong>
        </div>
      </div>
      <small className="motion-note">Conceptual animation · the transition is abrupt, not frame-by-frame probability.</small>
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
        <span>02 · Finite-size fragility</span>
        <h3>A larger society can tip at a smaller share.</h3>
        <p>The harmful fraction required for collapse falls with society size, even as the absolute critical count grows.</p>
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
      <small className="motion-note">Conceptual animation · icon counts are illustrative, not experimental measurements.</small>
    </article>
  )
}

export function AnimatedTeasers() {
  return (
    <section className="motion-section" aria-labelledby="motion-title">
      <div className="compact-heading">
        <p className="section-label">The idea in motion</p>
        <h2 id="motion-title">Two visual intuitions.</h2>
      </div>
      <div className="motion-grid">
        <TippingScene />
        <SizeScene />
      </div>
    </section>
  )
}
