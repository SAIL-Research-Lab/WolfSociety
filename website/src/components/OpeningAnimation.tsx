import { useEffect, useState, type CSSProperties } from 'react'

const stages = [
  { harmfulCount: 1, exposedCount: 0, routeCount: 2, label: 'A harmful minority appears', shortLabel: 'Few', reach: 'Local', stress: 'Low', state: 'stable' },
  { harmfulCount: 3, exposedCount: 6, routeCount: 8, label: 'Social signals begin to spread', shortLabel: 'Spreading', reach: 'Growing', stress: 'Low', state: 'stable' },
  { harmfulCount: 5, exposedCount: 16, routeCount: 18, label: 'The society nears a tipping point', shortLabel: 'Near tipping', reach: 'Broad', stress: 'Rising', state: 'warning' },
  { harmfulCount: 7, exposedCount: 28, routeCount: 26, label: 'Collective collapse', shortLabel: 'Collapse', reach: 'System-wide', stress: 'Severe', state: 'collapse' },
] as const

const harmfulOrder = [6, 34, 18, 42, 25, 11, 38]
const harmfulPositions = new Set(harmfulOrder)
const exposedOrder = [5, 7, 15, 16, 17, 24, 26, 33, 35, 43, 41, 19, 28, 27, 12, 10, 3, 4, 8, 9, 20, 21, 29, 30, 31, 39, 40, 44]
const communicationRoutes = [
  [6, 5], [6, 7], [6, 16], [34, 33], [34, 35], [34, 24], [18, 17], [18, 28],
  [18, 19], [18, 27], [42, 41], [42, 43], [42, 32], [25, 24], [25, 26], [25, 15],
  [34, 44], [6, 15], [11, 10], [11, 12], [38, 37], [38, 39], [38, 48], [25, 35],
  [42, 31], [18, 9],
] as const
const agents = Array.from({ length: 50 }, (_, index) => index)

function communicationRouteStyle(from: number, to: number, index: number) {
  const fromColumn = from % 10
  const fromRow = Math.floor(from / 10)
  const toColumn = to % 10
  const toRow = Math.floor(to / 10)
  const deltaX = (toColumn - fromColumn) * 10
  const deltaY = (toRow - fromRow) * 20
  const adjustedY = deltaY * 0.36

  return {
    '--route-x': `${(fromColumn + 0.5) * 10}%`,
    '--route-y': `${(fromRow + 0.5) * 20}%`,
    '--route-length': `${Math.sqrt(deltaX ** 2 + adjustedY ** 2)}%`,
    '--route-angle': `${Math.atan2(adjustedY, deltaX) * 180 / Math.PI}deg`,
    '--route-delay': `${(index % 7) * -0.19}s`,
  } as CSSProperties
}

export function OpeningAnimation() {
  const [activeStage, setActiveStage] = useState(0)
  const [isPlaying, setIsPlaying] = useState(true)

  useEffect(() => {
    if (!isPlaying || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const timer = window.setInterval(() => {
      setActiveStage((current) => (current + 1) % stages.length)
    }, 2200)
    return () => window.clearInterval(timer)
  }, [isPlaying])

  const stage = stages[activeStage]

  return (
    <section className="opening-animation" aria-labelledby="opening-animation-title">
      <div className={`opening-scene opening-scene--${stage.state}`}>
        <div className="opening-scene__header">
          <div>
            <span className="opening-kicker">Conceptual illustration—not measured data</span>
            <h2 id="opening-animation-title">A society near its tipping point</h2>
          </div>
          <output aria-live="polite">
            <span>{stage.label}</span>
            <small>{stage.harmfulCount} harmful · {stage.reach.toLowerCase()} reach</small>
          </output>
        </div>

        <div
          className="agent-society"
          role="img"
          aria-label="A conceptual society in which harmful messages travel between agents, widen their social reach, and precede sudden collective collapse"
        >
          <div className="communication-network" aria-hidden="true">
            {communicationRoutes.slice(0, stage.routeCount).map(([from, to], index) => (
              <span
                className="communication-route"
                key={`${from}-${to}`}
                style={communicationRouteStyle(from, to, index)}
              >
                <i />
              </span>
            ))}
          </div>
          <div className="agent-grid" aria-hidden="true">
            {agents.map((index) => {
              const isHarmfulPosition = harmfulPositions.has(index)
              const harmfulRank = harmfulOrder.indexOf(index)
              const isHarmful = isHarmfulPosition && harmfulRank < stage.harmfulCount
              const exposedRank = exposedOrder.indexOf(index)
              const isExposed = !isHarmful && exposedRank >= 0 && exposedRank < stage.exposedCount
              const pose = index % 4
              const style = {
                '--agent-index': index,
                '--agent-delay': `${(index % 10) * -0.11}s`,
                '--exposure-delay': `${(Math.max(exposedRank, 0) % 8) * -0.16}s`,
                '--scatter-x': `${((index * 17) % 47) - 23}px`,
                '--scatter-y': `${28 + ((index * 13) % 34)}px`,
                '--scatter-angle': `${((index * 29) % 80) - 40}deg`,
              } as CSSProperties

              return (
                <span
                  key={index}
                  className={`society-agent society-agent--pose-${pose}${isExposed ? ' society-agent--exposed' : ''}${isHarmful ? ' society-agent--harmful' : ''}`}
                  style={style}
                >
                  <svg viewBox="0 0 24 38" focusable="false">
                    {isHarmful && <path className="agent-horns" d="M7.5 3 5 0.8l.3 5M16.5 3 19 .8l-.3 5" />}
                    <circle className="agent-head" cx="12" cy="6" r="4.3" />
                    {isHarmful ? (
                      <path className="agent-face agent-face--harmful" d="m9 5 1.8 1M15 5l-1.8 1M9.6 8.2q2.4-1.7 4.8 0" />
                    ) : (
                      <path className="agent-face" d="M10 5.6h.1M13.9 5.6h.1M10.2 8q1.8 1.2 3.6 0" />
                    )}
                    <path className="agent-body" d="M12 11v12M12 23 6.2 34M12 23l5.8 11" />
                    {pose === 0 && <path className="agent-arms" d="M12 17 5 21M12 17l7 4" />}
                    {pose === 1 && <path className="agent-arms" d="M12 18 5.5 14 3 9M12 18l7 2" />}
                    {pose === 2 && <path className="agent-arms" d="M12 17 6 13M12 17l6-4" />}
                    {pose === 3 && <path className="agent-arms" d="M12 18 5 17M12 18l7-5 1-4" />}
                  </svg>
                </span>
              )
            })}
          </div>
          <div className="society-readout" key={`${stage.reach}-${stage.stress}`} aria-hidden="true">
            <span>Social reach <b>{stage.reach}</b></span>
            <span>Market stress <b>{stage.stress}</b></span>
          </div>
          <div className="society-ground" aria-hidden="true"><i /><i /><i /></div>
          <strong className="collapse-signal" aria-hidden="true">COLLAPSE</strong>
        </div>

        <div className="opening-controls" aria-label="Select a conceptual society stage">
          {stages.map((item, index) => (
            <button
              key={item.shortLabel}
              type="button"
              className={activeStage === index ? 'is-active' : ''}
              aria-pressed={activeStage === index}
              onClick={() => {
                setActiveStage(index)
                setIsPlaying(false)
              }}
            >
              <i aria-hidden="true" />
              {item.shortLabel}
            </button>
          ))}
          <button
            type="button"
            className="opening-playback"
            aria-label={isPlaying ? 'Pause conceptual animation' : 'Play conceptual animation'}
            onClick={() => setIsPlaying((current) => !current)}
          >
            <span aria-hidden="true">{isPlaying ? 'Ⅱ' : '▶'}</span>
            {isPlaying ? 'Pause' : 'Play'}
          </button>
        </div>
      </div>
      <p>Harmful messages move from agent to agent, widening their social reach and raising system pressure until the collective state gives way.</p>
    </section>
  )
}
