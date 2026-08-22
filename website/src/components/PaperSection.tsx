import { useState } from 'react'
import { BookOpen, Check, Clipboard, Sparkles } from 'lucide-react'
import { bibtex } from '../config/site'

export function PaperSection() {
  const [copied, setCopied] = useState(false)

  async function copyBibtex() {
    await navigator.clipboard.writeText(bibtex)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1800)
  }

  return (
    <section className="paper-section-block citation-section" id="paper">
      <div className="citation-heading">
        <span className="citation-heading__icon" aria-hidden="true"><BookOpen /></span>
        <div>
          <span className="citation-kicker"><Sparkles size={13} /> Cite this work</span>
          <h2>BibTeX</h2>
          <p className="citation-note">If you find this work useful, please cite the manuscript.</p>
        </div>
      </div>
      <div className={`bibtex-panel${copied ? ' bibtex-panel--copied' : ''}`}>
        <div className="bibtex-toolbar">
          <span><i aria-hidden="true" /><i aria-hidden="true" /><i aria-hidden="true" /> when-harm-scales.bib</span>
          <button type="button" onClick={copyBibtex} aria-label="Copy BibTeX citation">
            {copied ? <Check size={16} /> : <Clipboard size={16} />}
            {copied ? 'Copied!' : 'Copy citation'}
          </button>
        </div>
        <pre><code>{bibtex}</code></pre>
      </div>
      <footer>
        <p>When Harm Scales · Academic project page</p>
      </footer>
    </section>
  )
}
