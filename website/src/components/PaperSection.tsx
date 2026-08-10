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
    <section className="section paper-section" id="paper">
      <div className="paper-kicker">Reference</div>
      <h2>Citation</h2>
      <p className="citation-note">If this work is useful in your research, please cite the paper.</p>
      <div className="bibtex-panel">
        <div><span>BibTeX</span><button type="button" onClick={copyBibtex}>{copied ? <Check size={16} /> : <Clipboard size={16} />}{copied ? 'Copied' : 'Copy'}</button></div>
        <pre>{bibtex}</pre>
      </div>
      <footer>
        <span>When Harm Scales</span>
        <p>A study of harmful-agent scaling and collective failure.</p>
      </footer>
    </section>
  )
}
