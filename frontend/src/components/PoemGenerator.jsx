import { useState, useEffect } from 'react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const METERS = [
  { id: 'aataveladi', telugu: 'ఆటవెలది' },
  { id: 'kandamu', telugu: 'కందము' },
  { id: 'teytageethi', telugu: 'తేటగీతి' },
  { id: 'seesamu', telugu: 'సీసము' },
  { id: 'mattebhamu', telugu: 'మత్తేభము' },
  { id: 'champakamaala', telugu: 'చంపకమాల' },
  { id: 'vutpalamaala', telugu: 'ఉత్పలమాల' },
  { id: 'saardulamu', telugu: 'శార్దూలము' },
]

export default function PoemGenerator({ onGenerate, isLoading }) {
  const [chandas, setChandas] = useState('aataveladi')
  const [emotion, setEmotion] = useState('')
  const [theme, setTheme] = useState('')
  const [emotions, setEmotions] = useState([])

  useEffect(() => {
    fetch(`${API_BASE}/api/emotions`)
      .then(r => r.json())
      .then(data => {
        setEmotions(data)
        if (data.length > 0) setEmotion(data[0].id)
      })
      .catch(() => {
        // Fallback emotions if API is not available
        const fallback = [
          { id: 'bhakti', telugu: 'భక్తి', english: 'Devotion' },
          { id: 'sringaaram', telugu: 'శృంగారం', english: 'Romance' },
          { id: 'haasyam', telugu: 'హాస్యం', english: 'Humor' },
          { id: 'karuna', telugu: 'కరుణ', english: 'Compassion' },
          { id: 'veeram', telugu: 'వీరం', english: 'Heroism' },
          { id: 'shaantam', telugu: 'శాంతం', english: 'Peace' },
        ]
        setEmotions(fallback)
        setEmotion('bhakti')
      })
  }, [])

  const handleSubmit = () => {
    if (chandas && emotion) {
      onGenerate({ chandas, emotion, theme })
    }
  }

  return (
    <div>
      <div className="section-header">
        <span className="section-header-icon">✨</span>
        <span className="section-header-text">Generate Telugu Poem</span>
      </div>

      {/* Chandas Selection */}
      <div className="gen-field">
        <label className="gen-label">ఛందస్సు · Meter</label>
        <div className="gen-select-wrapper">
          <select
            className="gen-select"
            value={chandas}
            onChange={(e) => setChandas(e.target.value)}
            disabled={isLoading}
          >
            {METERS.map(m => (
              <option key={m.id} value={m.id}>
                {m.telugu} — {m.id}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Emotion Selection */}
      <div className="gen-field">
        <label className="gen-label">రసం · Emotion</label>
        <div className="gen-select-wrapper">
          <select
            className="gen-select"
            value={emotion}
            onChange={(e) => setEmotion(e.target.value)}
            disabled={isLoading}
          >
            {emotions.map(e => (
              <option key={e.id} value={e.id}>
                {e.telugu} — {e.english}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Theme Input */}
      <div className="gen-field">
        <label className="gen-label">అంశం · Theme <span className="gen-optional">(optional)</span></label>
        <input
          type="text"
          className="gen-input"
          value={theme}
          onChange={(e) => setTheme(e.target.value)}
          placeholder="e.g. Nature, Lord Vishnu, Mother's love..."
          disabled={isLoading}
        />
      </div>

      {/* Generate Button */}
      <button
        className="btn-primary btn-generate"
        onClick={handleSubmit}
        disabled={isLoading || !chandas || !emotion}
      >
        {isLoading ? (
          <>
            <span className="spinner" />
            Generating…
          </>
        ) : (
          <>✨  Generate Poem</>
        )}
      </button>
    </div>
  )
}
