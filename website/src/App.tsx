import './App.css'
import { Abstract } from './components/Abstract'
import { AnimatedTeasers } from './components/AnimatedTeasers'
import { FutureWork } from './components/FutureWork'
import { Hero } from './components/Hero'
import { OpeningAnimation } from './components/OpeningAnimation'
import { PaperSection } from './components/PaperSection'
import { ScalingStory } from './components/ScalingStory'
import { SocialDynamics } from './components/SocialDynamics'
import { TeaserFigure } from './components/TeaserFigure'

function App() {
  return (
    <>
      <a className="skip-link" href="#abstract">Skip to abstract</a>
      <main>
        <Hero />
        <OpeningAnimation />
        <TeaserFigure />
        <AnimatedTeasers />
        <Abstract />
        <SocialDynamics />
        <ScalingStory />
        <FutureWork />
        <PaperSection />
      </main>
    </>
  )
}

export default App
