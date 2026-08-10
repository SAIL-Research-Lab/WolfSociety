export function FutureWork() {
  return (
    <section className="future-section" id="future">
      <div className="content-column future-layout">
        <p className="section-label">Future work</p>
        <h2>Toward WolfBench.</h2>
        <p className="future-lede">
          WolfBench is the working name for a future, more complete evaluation environment built from the controlled system used in this study. It is an ongoing research direction rather than a finished benchmark release.
        </p>
        <div className="future-priorities">
          <article>
            <strong>Broader environments</strong>
            <p>Extend beyond the current social–market setting with more scenarios, agent roles, and interaction structures.</p>
          </article>
          <article>
            <strong>Stronger evaluation</strong>
            <p>Improve realism, reproducibility, controller audits, and sensitivity analysis as the environment evolves.</p>
          </article>
          <article>
            <strong>Defense benchmarking</strong>
            <p>Develop interfaces for comparing interventions that target propagation, feedback, and collective failure.</p>
          </article>
        </div>
        <p className="future-note">The environment will continue to be revised and validated before being presented as a mature benchmark.</p>
      </div>
    </section>
  )
}