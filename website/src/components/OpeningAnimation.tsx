import { useEffect, useState, type CSSProperties } from 'react'

const stages = [
  { harmfulCount: 1, label: 'A harmful minority appears', shortLabel: 'Few', state: 'stable' },
  { harmfulCount: 3, label: 'Harmful agents spread', shortLabel: 'More', state: 'stable' },
  { harmfulCount: 5, label: 'The society nears a tipping point', shortLabel: 'Near tipping', state: 'warning' },
  { harmfulCount: 7, label: 'Collective collapse', shortLabel: 'Collapse', state: 'collapse' },
] as const

const harmfulOrder = [6, 34, 18, 42, 25, 11, 38]
const harmfulPositions = new Set(harmfulOrder)
const agents = Array.from({ length: 50 }, (_, index) => index)

export function OpeningAnimation() {
  const [activeStage, setActiveStage] = useState(0)

  useEffect(() => {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    const timer = window.setInterval(() => {
      setActiveStage((current) => (current + 1) % stages.length)
    }, 1750)
    return () => window.clearInterval(timer)
  }, [])

  const stage = stages[activeStage]

  return (
    <section className="opening-animation" aria-labelledby="opening-animation-title">
      <div className={`opening-scene opening-scene--${stage.state}`}>
        <div className="opening-scene__header">
          <div>
            <span className="opening-kicker">Conceptual illustration—not measured data</span>
            <h2 id="opening-animation-title">A society near its tipping point</h2>
          </div>
          <output aria-live="polite">{stage.label}</output>
        </div>

        <div
          className="agent-society"
          role="img"
          aria-label="A conceptual society in which a small but growing number of harmful agents precedes sudden collective collapse"
        >
          <div className="agent-grid" aria-hidden="true">
            {agents.map((index) => {
              const isHarmfulPosition = harmfulPositions.has(index)
              const harmfulRank = harmfulOrder.indexOf(index)
              const isHarmful = isHarmfulPosition && harmfulRank < stage.harmfulCount
              const style = {
                '--agent-index': index,
                '--scatter-x': `${((index * 17) % 47) - 23}px`,
                '--scatter-y': `${28 + ((index * 13) % 34)}px`,
                '--scatter-angle': `${((index * 29) % 80) - 40}deg`,
              } as CSSProperties

              return (
                <span
                  key={index}
                  className={`society-agent${isHarmful ? ' society-agent--harmful' : ''}`}
                  style={style}
                >
                  <svg viewBox="0 0 24 38" focusable="false">
                    {isHarmful && <path className="agent-horns" d="M7.5 3 5 0.8l.3 5M16.5 3 19 .8l-.3 5" />}
                    <circle cx="12" cy="6" r="4.3" />
                    <path d="M12 11v12M4.8 15.5 12 19l7.2-3.5M12 23 6.2 34M12 23l5.8 11" />
                  </svg>
                </span>
              )
            })}
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
              onClick={() => setActiveStage(index)}
            >
              <i aria-hidden="true" />
              {item.shortLabel}
            </button>
          ))}
        </div>
      </div>
      <p>A small harmful minority grows while the society appears stable—until the collective state suddenly gives way.</p>
    </section>
  )
}
