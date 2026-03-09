import { useState } from 'react'
import { FloorPlan, Constraints } from './types/floorplan'
import { generatePlans } from './api/client'
import ConstraintForm from './components/ConstraintForm'
import FloorPlanGallery from './components/FloorPlanGallery'
import FloorPlanEditor from './components/FloorPlanEditor'

type Screen = 'gallery' | 'editor'

export default function App() {
  const [screen, setScreen] = useState<Screen>('gallery')
  const [plans, setPlans] = useState<FloorPlan[]>([])
  const [selected, setSelected] = useState<FloorPlan | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function handleGenerate(c: Constraints) {
    setLoading(true)
    setError(null)
    setPlans([])
    setScreen('gallery')
    try {
      const result = await generatePlans(c)
      setPlans(result)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Generation failed'
      setError(msg)
    } finally {
      setLoading(false)
    }
  }

  function handleSelect(plan: FloorPlan) {
    setSelected(plan)
    setScreen('editor')
  }

  function handleUpdate(updated: FloorPlan) {
    setSelected(updated)
    setPlans(prev => prev.map(p => (p.id === updated.id ? updated : p)))
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="logo">
          <svg className="logo-icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
            <path d="M3 9L12 3L21 9V21H15V15H9V21H3V9Z" fill="currentColor" />
          </svg>
          <span className="logo-text">Buildify</span>
        </div>

        <ConstraintForm onGenerate={handleGenerate} loading={loading} />

        {error && <div className="error-msg">{error}</div>}

        {plans.length > 0 && (
          <div className="sidebar-plans">
            <div className="sidebar-plans-title">Generated Plans</div>
            {plans.map(p => (
              <div
                key={p.id}
                className={`plan-chip ${selected?.id === p.id ? 'active' : ''}`}
                onClick={() => handleSelect(p)}
              >
                <div className="plan-chip-name">{p.name}</div>
                <div className="plan-chip-meta">
                  {p.rooms.length} rooms ·{' '}
                  {Math.round(p.rooms.reduce((s, r) => s + r.width * r.height, 0)).toLocaleString()} sq ft
                </div>
              </div>
            ))}
          </div>
        )}

        {screen === 'editor' && (
          <button className="back-btn" onClick={() => setScreen('gallery')}>
            ← Back to Gallery
          </button>
        )}
      </aside>

      <main className="main">
        {screen === 'gallery' && (
          <FloorPlanGallery plans={plans} loading={loading} onSelect={handleSelect} />
        )}
        {screen === 'editor' && selected && (
          <FloorPlanEditor plan={selected} onUpdate={handleUpdate} />
        )}
      </main>
    </div>
  )
}
