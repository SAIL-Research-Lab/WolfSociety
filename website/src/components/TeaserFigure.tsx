export function TeaserFigure() {
  const base = import.meta.env.BASE_URL

  return (
    <section className="teaser-section" aria-label="Paper teaser">
      <figure className="paper-figure paper-figure--teaser">
        <img
          src={`${base}teaser.png`}
          alt="Overview of the closed-loop agent society and the three core signatures: nonlinear collapse, finite-size scaling, and Agent Society Dynamics."
        />
        <figcaption>
          <strong>Figure 1.</strong> A harmful minority acts through social propagation and shared market state. Collapse appears nonlinearly, while larger societies reach the boundary at a smaller harmful fraction but a larger absolute harmful count.
        </figcaption>
      </figure>
    </section>
  )
}
