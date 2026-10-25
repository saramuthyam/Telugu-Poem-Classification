import { useState, useEffect } from 'react'
import './index.css'
import Sidebar from './components/Sidebar'
import PoemInput from './components/PoemInput'
import PoemGenerator from './components/PoemGenerator'
import ResultCard from './components/ResultCard'
import InterpretationPanel from './components/InterpretationPanel'
import ProbabilityChart from './components/ProbabilityChart'
import LaghuGuruViz from './components/LaghuGuruViz'
import ScoreRadar from './components/ScoreRadar'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

export default function App() {
  const [health, setHealth] = useState(null)
  const [results, setResults] = useState(null)
  const [genResults, setGenResults] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)
  const [mode, setMode] = useState('analyze') // 'analyze' or 'generate'

  // Fetch health on mount
  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then(r => r.json())
      .then(data => setHealth(data))
      .catch(() => setHealth({ status: 'error', models_loaded: false, available_models: [], gemini_available: false }))
  }, [])

  const handleAnalyze = async (text) => {
    setIsLoading(true)
    setError(null)
    setResults(null)

    try {
      const res = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || `HTTP ${res.status}`)
      }

      const data = await res.json()
      setResults(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setIsLoading(false)
    }
  }

  const handleGenerate = async ({ chandas, emotion, theme }) => {
    setIsLoading(true)
    setError(null)
    setGenResults(null)

    try {
      const res = await fetch(`${API_BASE}/api/generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chandas, emotion, theme }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || `HTTP ${res.status}`)
      }

      const data = await res.json()
      setGenResults(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setIsLoading(false)
    }
  }

  const switchMode = (newMode) => {
    setMode(newMode)
    setError(null)
    // Don't clear results — let user switch back and see them
  }

  const activeResults = mode === 'analyze' ? results : null
  const activeGenResults = mode === 'generate' ? genResults : null

  return (
    <div className="app-layout">
      <Sidebar health={health} />

      <main className="main-content">
        {/* Hero Header */}
        <div className="hero-container">
          <span className="hero-icon">📜</span>
          <h1 className="hero-title">తెలుగు పద్య విశ్లేషణ</h1>
          <p className="hero-subtitle">Automated Chandas Classification of Telugu Classical Poetry Using Deep Learning</p>
        </div>
        <div className="styled-divider" />

        {/* Mode Toggle */}
        <div className="mode-toggle">
          <button
            className={`mode-btn ${mode === 'analyze' ? 'active' : ''}`}
            onClick={() => switchMode('analyze')}
          >
            🔍 Analyze Poem
          </button>
          <button
            className={`mode-btn ${mode === 'generate' ? 'active' : ''}`}
            onClick={() => switchMode('generate')}
          >
            ✨ Generate Poem
          </button>
        </div>

        {/* Two-column layout */}
        <div className="content-grid">
          {/* Left: Input */}
          <div>
            {mode === 'analyze' ? (
              <PoemInput onAnalyze={handleAnalyze} isLoading={isLoading} />
            ) : (
              <PoemGenerator onGenerate={handleGenerate} isLoading={isLoading} />
            )}
          </div>

          {/* Right: Results */}
          <div>
            <div className="section-header">
              <span className="section-header-icon">{mode === 'analyze' ? '📊' : '📝'}</span>
              <span className="section-header-text">
                {mode === 'analyze' ? 'Analysis Results' : 'Generated Poem'}
              </span>
            </div>

            {error && (
              <div className="error-alert">⚠️ {error}</div>
            )}

            {/* === ANALYZE MODE RESULTS === */}
            {mode === 'analyze' && activeResults ? (
              <>
                {activeResults.chandas && (
                  <ResultCard
                    label="Predicted Chandas · ఛందస్సు"
                    value={
                      activeResults.chandas_telugu
                        ? `${activeResults.chandas_telugu} (${activeResults.chandas})`
                        : activeResults.chandas
                    }
                    accent="gold"
                    confidence={activeResults.chandas_confidence}
                    confColor="gold"
                  />
                )}

                {activeResults.poem_class && (
                  <ResultCard
                    label="Meter Class · వర్గం"
                    value={activeResults.poem_class}
                    accent="emerald"
                  />
                )}

                {activeResults.source && (
                  <ResultCard
                    label="Predicted Source · శతకం"
                    value={activeResults.source}
                    accent="cyan"
                    confidence={activeResults.source_confidence}
                    confColor="cyan"
                  />
                )}
              </>
            ) : mode === 'analyze' && !isLoading && !error && (
              <div className="result-card">
                <div className="empty-state">
                  <div className="empty-state-icon">📜</div>
                  <div className="empty-state-text">
                    Paste a Telugu poem or click <strong>Try Example</strong><br />
                    then click <strong>Analyze Poem</strong> to see results.
                  </div>
                </div>
              </div>
            )}

            {/* === GENERATE MODE RESULTS === */}
            {mode === 'generate' && activeGenResults ? (
              <>
                {/* Generated Poem Card */}
                <div className="gen-poem-card">
                  <div className="gen-poem-header">
                    <span className="gen-method-badge">
                      📜 పద్యం · Poem
                    </span>
                    <button
                      className="gen-copy-btn"
                      onClick={() => {
                        navigator.clipboard.writeText(activeGenResults.poem_text)
                      }}
                      title="Copy poem"
                    >
                      📋 Copy
                    </button>
                  </div>
                  <div className="gen-poem-text">{activeGenResults.poem_text}</div>
                  <div className="gen-poem-meta">
                    <span className="gen-meta-chip gold">
                      {activeGenResults.chandas_telugu} · {activeGenResults.chandas}
                    </span>
                    <span className="gen-meta-chip violet">
                      {activeGenResults.emotion_telugu}
                    </span>
                    {activeGenResults.theme && activeGenResults.theme !== '(auto-selected)' && (
                      <span className="gen-meta-chip cyan">
                        {activeGenResults.theme}
                      </span>
                    )}
                  </div>
                </div>

                {/* Verification Card */}
                {activeGenResults.verification && (
                  <div className={`result-card ${activeGenResults.verification.match ? 'emerald' : 'gold'}`}>
                    <div className="result-label">CNN Verification · యంత్ర ధృవీకరణ</div>
                    <div className="result-value" style={{ fontSize: '1.1rem' }}>
                      {activeGenResults.verification.match ? (
                        <span style={{ color: 'var(--accent-emerald)' }}>✓ Meter Verified — {activeGenResults.verification.predicted_chandas_telugu}</span>
                      ) : (
                        <span style={{ color: 'var(--accent-gold)' }}>
                          ⚠ CNN predicted: {activeGenResults.verification.predicted_chandas_telugu || activeGenResults.verification.predicted_chandas}
                        </span>
                      )}
                    </div>
                    {activeGenResults.verification.chandas_confidence != null && (
                      <>
                        <div className="conf-track">
                          <div
                            className={`conf-fill ${activeGenResults.verification.match ? 'emerald' : 'gold'}`}
                            style={{ width: `${activeGenResults.verification.chandas_confidence * 100}%` }}
                          />
                        </div>
                        <div className="conf-text">
                          {(activeGenResults.verification.chandas_confidence * 100).toFixed(1)}% confidence
                        </div>
                      </>
                    )}
                  </div>
                )}
              </>
            ) : mode === 'generate' && !isLoading && !error && (
              <div className="result-card">
                <div className="empty-state">
                  <div className="empty-state-icon">✨</div>
                  <div className="empty-state-text">
                    Select a meter and emotion, then click<br />
                    <strong>Generate Poem</strong> to create a Telugu verse.
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Full-width sections below the grid */}
        {mode === 'analyze' && activeResults && (
          <div className="full-width-results">
            <LaghuGuruViz lgPattern={activeResults.lg_pattern} />
            <ScoreRadar scores={activeResults.prosodic_scores} />
            <InterpretationPanel
              interpretation={activeResults.interpretation}
              aiInterpretation={activeResults.ai_interpretation}
              chandasInfo={activeResults.chandas_info}
            />
            <ProbabilityChart probabilities={activeResults.chandas_all} />
          </div>
        )}

        {mode === 'generate' && activeGenResults && (
          <div className="full-width-results">
            <LaghuGuruViz lgPattern={activeGenResults.lg_pattern} />
            <ScoreRadar scores={activeGenResults.prosodic_scores} />
            {activeGenResults.chandas_info && (
              <div style={{ marginTop: '1rem' }}>
                <div className="section-header">
                  <span className="section-header-icon">📐</span>
                  <span className="section-header-text">Meter Details · ఛందస్సు వివరాలు</span>
                </div>
                <div className="chandas-desc">
                  <div className="chandas-desc-title">
                    {activeGenResults.chandas_info.telugu} · {activeGenResults.chandas_info.class}
                  </div>
                  {activeGenResults.chandas_info.structure && (
                    <div className="chandas-desc-structure">{activeGenResults.chandas_info.structure}</div>
                  )}
                  <div className="chandas-desc-text">{activeGenResults.chandas_info.description}</div>
                </div>
              </div>
            )}
          </div>
        )}
      </main>
    </div>
  )
}
