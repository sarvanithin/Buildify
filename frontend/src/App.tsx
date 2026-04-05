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
      // Handle structured 422 validation errors from backend
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const axiosErr = e as any
      if (axiosErr?.response?.status === 422 && axiosErr?.response?.data?.detail?.validation_errors) {
        const issues = axiosErr.response.data.detail.validation_errors as Array<{severity: string; message: string; detail: string}>
        const lines = issues.map(i =>
          `${i.severity === 'error' ? '✕' : '⚠'} ${i.message}\n${i.detail}`
        )
        setError(lines.join('\n\n'))
      } else {
        const msg = e instanceof Error ? e.message : 'Generation failed'
        setError(msg)
      }
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
