import { useState } from 'react'
import { Check, Clipboard } from 'lucide-react'
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
      <h2>BibTeX</h2>
      <p className="citation-note">If this work is useful in your research, please cite the paper.</p>
      <div className="bibtex-panel">
        <div><span>BibTeX</span><button type="button" onClick={copyBibtex}>{copied ? <Check size={16} /> : <Clipboard size={16} />}{copied ? 'Copied' : 'Copy'}</button></div>
        <pre>{bibtex}</pre>
      </div>
      <footer>
        <p>When Harm Scales · Academic project page</p>
      </footer>
    </section>
  )
}
