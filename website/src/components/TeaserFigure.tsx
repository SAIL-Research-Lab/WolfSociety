export function TeaserFigure() {
  const base = import.meta.env.BASE_URL

  return (
    <section className="teaser-section" aria-label="Paper teaser">
      <figure className="paper-figure paper-figure--teaser">
        <img
          src={`${base}teaser.png`}
          alt="Overview of the financial agent society, its closed feedback loop, and the main scaling pattern."
        />
        <figcaption>
          <strong>Figure 1.</strong> Overview of the financial agent society and its main scaling pattern. Harmful agents influence others through communication and trading, while the resulting actions change market and social conditions observed by later agents. As society size grows, the harmful fraction associated with collapse decreases while the corresponding number of harmful agents increases. Agent Society Dynamics relates this pattern to direct harmful influence and feedback through the shared environment.
        </figcaption>
      </figure>
    </section>
  )
}
