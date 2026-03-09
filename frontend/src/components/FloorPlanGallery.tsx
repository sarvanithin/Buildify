import { FloorPlan } from '../types/floorplan'
import FloorPlanPreview from './FloorPlanPreview'

interface Props {
  plans: FloorPlan[]
  loading: boolean
  onSelect: (plan: FloorPlan) => void
}

const STEPS = [
  { n: 1, text: 'Set constraints and explore starting designs' },
  { n: 2, text: 'Explore and evolve in 2D and 3D' },
  { n: 3, text: 'View rendered exterior' },
  { n: 4, text: 'Export to CAD / DXF' },
]

export default function FloorPlanGallery({ plans, loading, onSelect }: Props) {
  if (loading) {
    return (
      <div className="gallery-loading">
        <div className="loading-spinner-large" />
        <p className="loading-title">Generating floor plans with AI...</p>
        <p className="loading-sub">3 unique layouts via Ollama · ~30-60 seconds</p>
      </div>
    )
  }

  if (plans.length === 0) {
    return (
      <div className="gallery-empty">
        <div className="empty-hex">⬡</div>
        <h2>Explore ideas fast and wide</h2>
        <p>Set your constraints in the sidebar, then click Generate Plans</p>
        <div className="steps-row">
          {STEPS.map(s => (
            <div key={s.n} className="step-card">
              <div className="step-num">{s.n}</div>
              <div className="step-text">{s.text}</div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  const totalSqft = (plan: FloorPlan) =>
    Math.round(plan.rooms.reduce((s, r) => s + r.width * r.height, 0))

  return (
    <div className="gallery">
      <div className="gallery-header">
        <h2>Choose a Starting Point</h2>
        <p>{plans.length} plans generated · click any to open in editor</p>
      </div>
      <div className="gallery-grid">
        {plans.map(plan => (
          <div key={plan.id} className="gallery-card" onClick={() => onSelect(plan)}>
            <FloorPlanPreview plan={plan} />
            <div className="card-footer">
              <span className="card-name">{plan.name}</span>
              <span className="card-meta">
                {plan.rooms.length} rooms · {totalSqft(plan).toLocaleString()} sq ft
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
