import { useState } from 'react'
import { FloorPlan, Constraints } from './types/floorplan'
import { generatePlans, generatePlansMOE, MOEResult } from './api/client'
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
  const [moeData, setMoeData] = useState<{
    expert_weights: Record<string, number>
    confidence: number
    irc_compliant: boolean
  } | null>(null)

  async function handleGenerate(c: Constraints, useMOE?: boolean) {
    setLoading(true)
    setError(null)
    setPlans([])
    setMoeData(null)
    setScreen('gallery')
    try {
      if (useMOE) {
        const result: MOEResult = await generatePlansMOE(c)
        setPlans(result.plans)
        setMoeData({
          expert_weights: result.expert_weights,
          confidence: result.confidence,
          irc_compliant: result.irc_compliant,
        })
      } else {
        const result = await generatePlans(c)
        setPlans(result)
      }
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

        {moeData && (
          <div className="moe-status-panel">
            <div className="moe-status-header">
              <span>🧠 MOE AI</span>
              <span className="moe-confidence">{moeData.confidence}%</span>
            </div>
            <div className="moe-status-irc">
              {moeData.irc_compliant ? '✅ IRC Compliant' : '⚠️ Check compliance'}
            </div>
            <div className="moe-expert-bars">
              {Object.entries(moeData.expert_weights).map(([name, weight]) => (
                <div key={name} className="expert-bar-row">
                  <span className="expert-bar-label">{name}</span>
                  <div className="expert-bar-track">
                    <div
                      className="expert-bar-fill"
                      style={{ width: `${Math.round(weight * 100)}%` }}
                    />
                  </div>
                  <span className="expert-bar-value">{Math.round(weight * 100)}%</span>
                </div>
              ))}
            </div>
          </div>
        )}

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
