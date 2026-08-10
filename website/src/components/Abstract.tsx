import { site } from '../config/site'

export function Abstract() {
  return (
    <section className="abstract-section" id="abstract">
      <div className="academic-content">
        <h2>Abstract</h2>
        {site.abstract.map((paragraph) => <p key={paragraph}>{paragraph}</p>)}
      </div>
    </section>
  )
}
