
import { useState, useEffect } from 'react'

const FALLBACK_SAMPLES = [
  {
    meter: 'aataveladi',
    meter_telugu: 'ఆటవెలది',
    source: 'Vemana',
    text: 'ఉప్పు కప్పురంబు నొక్క పోలిక నుండు\nచూడ చూడ రుచుల జాడ తెలియు\nపురుషులందు పుణ్య పురుషులు వేరయా\nవిశ్వదాభిరామ వినుర వేమ'
  },
  {
    meter: 'kandamu',
    meter_telugu: 'కందము',
    source: 'Sumathi',
    text: 'అక్కరకు రాని చుట్టము\nమ్రొక్కిన వరమియ్యని వేల్పు మొదలుం దన పే\nరెక్కిన పనికిమాలిన\nచొక్కపు బంగారమును రాసులకు రావు సుమీ'
  },
  {
    meter: 'seesamu',
    meter_telugu: 'సీసము',
    source: 'Sample Poem',
    text: 'శ్రీమదనంత లక్ష్మీ యుతోరః స్థల\nచతురాననాండ పూరిత పిచండ\nధర చక్ర ఖడ్గ గదా శరాసనహస్త\nనిఖిల వేదాంత వర్ణిత చరిత్ర'
  }
]

export default function PoemInput({ onAnalyze, isLoading }) {
  const [text, setText] = useState('')
  const [samples, setSamples] = useState(FALLBACK_SAMPLES)
  const [showSamples, setShowSamples] = useState(false)

  useEffect(() => {
    const apiBase = (
      import.meta.env.VITE_API_URL || 'http://localhost:8000'
    ).replace(/\/+$/, '')

    fetch(`${apiBase}/api/samples`)
      .then(response => {
        if (!response.ok) {
          throw new Error(`Samples API error: ${response.status}`)
        }
        return response.json()
      })
      .then(data => {
        const list = Array.isArray(data)
          ? data
          : data?.samples || data?.poems || data?.data || []

        if (Array.isArray(list) && list.length > 0) {
          const validSamples = list.filter(
            sample => sample && typeof sample.text === 'string'
          )

          if (validSamples.length > 0) {
            setSamples(validSamples)
          }
        }
      })
      .catch(error => {
        console.error('Could not load API samples; using fallback poems:', error)
      })
  }, [])

  const handleSubmit = () => {
    if (text.trim().length >= 10) {
      onAnalyze(text)
    }
  }

  const handleKeyDown = (event) => {
    if (event.ctrlKey && event.key === 'Enter') {
      handleSubmit()
    }
  }

  const loadSample = (sample) => {
    setText(sample.text || '')
    setShowSamples(false)
  }

  return (
    <div>
      <div className="section-header">
        <span className="section-header-icon">✍️</span>
        <span className="section-header-text">
          Enter Your Telugu Poem
        </span>
      </div>

      <div className="samples-container">
        <button
          type="button"
          className="samples-toggle"
          onClick={() => setShowSamples(previous => !previous)}
          aria-expanded={showSamples}
        >
          <span>⚡ Try Example Poem</span>
          <span className={`arrow ${showSamples ? 'open' : ''}`}>
            ▶
          </span>
        </button>

        {showSamples && (
          <div className="samples-dropdown">
            {samples.map((sample, index) => (
              <button
                type="button"
                key={`${sample.source || 'sample'}-${index}`}
                className="sample-item"
                onClick={() => loadSample(sample)}
              >
                <span>{sample.meter_telugu || sample.meter || 'Poem'}</span>
                <span>{sample.source || 'Sample'}</span>
                <span>{sample.meter || ''}</span>
              </button>
            ))}
          </div>
        )}
      </div>

      <textarea
        id="poem-input"
        className="poem-textarea"
        value={text}
        onChange={event => setText(event.target.value)}
        onKeyDown={handleKeyDown}
        placeholder="శ్రీమదనంత లక్ష్మీ యుతోరః స్థల చతురాననాండ పూరిత పిచండ..."
        disabled={isLoading}
      />

      <button
        id="analyze-btn"
        type="button"
        className="btn-primary"
        onClick={handleSubmit}
        disabled={isLoading || text.trim().length < 10}
      >
        {isLoading ? (
          <>
            <span className="spinner" />
            Analyzing…
          </>
        ) : (
          <>🔍 Analyze Poem</>
        )}
      </button>
    </div>
  )
}
