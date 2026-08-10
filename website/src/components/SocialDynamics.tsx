export function SocialDynamics() {
  return (
    <section className="framework-section" id="framework">
      <div className="compact-heading compact-heading--center">
        <p className="section-label">Agent Society Dynamics</p>
        <h2>From local pressure to collective collapse.</h2>
        <p>Harmful pressure scales with population, while shared-state feedback amplifies its effect.</p>
      </div>

      <div className="framework-prose">
        <p>At the local level, harmful pressure combines with private and social evidence to change an agent’s response. Those responses become consequential when communication and trading actions alter the social and market state observed by the rest of the society.</p>
        <p>At the population level, we separate the aggregation of harmful pressure from the amplification created by this shared-state feedback. Collapse occurs when their combined response reaches a size-invariant boundary under the tested protocol.</p>
      </div>

      <div className="mechanism-story" aria-label="Harmful pressure enters a shared-state feedback loop and produces a collective response">
        <div className="pressure-cluster">
          <span className="story-caption">Harmful pressure</span>
          <div className="pressure-agents" aria-hidden="true">
            <i />
            <i />
            <b>😈</b>
            <i />
            <i />
            <i />
            <i />
          </div>
        </div>

        <div className="story-arrow" aria-hidden="true"><span /></div>

        <div className="feedback-system">
          <span className="story-caption">Shared-state feedback</span>
          <div className="feedback-orbit" aria-hidden="true">
            <svg viewBox="0 0 220 150">
              <defs>
                <marker id="feedback-arrow" markerWidth="7" markerHeight="7" refX="5" refY="3.5" orient="auto">
                  <path d="M0 0 7 3.5 0 7Z" />
                </marker>
              </defs>
              <path d="M52 39 C93 5 165 12 185 56" />
              <path d="M184 91 C151 137 76 139 42 95" />
            </svg>
            <i className="orbit-node orbit-node--one" />
            <i className="orbit-node orbit-node--two" />
            <i className="orbit-node orbit-node--three" />
            <i className="orbit-node orbit-node--four" />
            <strong>SHARED<br />STATE</strong>
          </div>
        </div>

        <div className="story-arrow" aria-hidden="true"><span /></div>

        <div className="response-wave">
          <span className="story-caption">Collective response</span>
          <div aria-hidden="true">
            <i />
            <i />
            <i />
            <b>!</b>
          </div>
        </div>
      </div>

      <p className="mechanism-annotation">
        pressure ∼ αN<sup>δ</sup>
        <span>·</span>
        feedback ∼ N<sup>ζ</sup>
        <span>→</span>
        ΔR<sub>N</sub> ∼ αN<sup>δ+ζ</sup>
      </p>

      <div className="framework-divider" />

      <div className="scaling-consequence">
        <p className="scaling-kicker">As society grows</p>
        <div className="size-axis" aria-label="Society size increases from 100 to 2000">
          <strong>100</strong>
          <span><i /></span>
          <strong>2,000</strong>
        </div>

        <div className="scaling-lines" aria-label="Critical fraction decreases from 4.7 percent to 2.2 percent while critical harmful count increases from 4.7 to 44">
          <div className="scaling-row scaling-row--down">
            <span>Critical fraction</span>
            <strong>4.7%</strong>
            <svg viewBox="0 0 420 72" preserveAspectRatio="none" aria-hidden="true">
              <path d="M4 12 C145 12 275 38 416 60" />
              <path className="trend-arrow" d="m404 50 12 10-16 4" />
            </svg>
            <strong>2.2%</strong>
          </div>
          <div className="boundary-label">collapse boundary</div>
          <div className="scaling-row scaling-row--up">
            <span>Critical count</span>
            <strong>4.7</strong>
            <svg viewBox="0 0 420 72" preserveAspectRatio="none" aria-hidden="true">
              <path d="M4 60 C145 60 275 34 416 12" />
              <path className="trend-arrow" d="m400 8 16 4-12 11" />
            </svg>
            <strong>44</strong>
          </div>
        </div>

        <div className="scaling-insight">
          <strong>Smaller fraction.</strong>
          <strong>More harmful agents.</strong>
        </div>

        <p className="scaling-annotation">
          α<sub>c</sub>(N) ∝ N<sup>−ν</sup>
          <span>·</span>
          K<sub>c</sub>(N) ∝ N<sup>1−ν</sup>
          <span>·</span>
          0 &lt; ν &lt; 1
        </p>
      </div>
    </section>
  )
}
