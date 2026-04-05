import { FloorPlan } from '../types/floorplan'
import FloorPlanPreview from './FloorPlanPreview'

interface Props {
  plans: FloorPlan[]
  loading: boolean
  onSelect: (plan: FloorPlan) => void
}

const CARD_LABELS = ['A', 'B', 'C', 'D', 'E', 'F']

const STEPS = [
  { n: '①', text: 'Set your room list and square footage' },
  { n: '②', text: 'AI generates multiple plan variants for you' },
  { n: '③', text: 'Pick a starting point and explore in 2D & 3D' },
  { n: '④', text: 'Refine rooms, view walkthrough, export CAD' },
]

function bedroomCount(plan: FloorPlan) {
  return plan.rooms.filter(r =>
    r.type === 'bedroom' || r.type === 'master_bedroom' ||
    r.type === 'primary_bedroom' || r.name.toLowerCase().includes('bed')
  ).length
}
function bathroomCount(plan: FloorPlan) {
  return plan.rooms.filter(r =>
    r.type === 'bathroom' || r.type === 'ensuite_bathroom' ||
    r.type === 'half_bath' || r.name.toLowerCase().includes('bath')
  ).length
}
const _UNCONDITIONED = new Set(['garage', 'patio', 'deck', 'rear_patio', 'outdoor_living', 'front_porch'])
function totalSqft(plan: FloorPlan) {
  return Math.round(plan.rooms.filter(r => !_UNCONDITIONED.has(r.type)).reduce((s, r) => s + r.width * r.height, 0))
}

export default function FloorPlanGallery({ plans, loading, onSelect }: Props) {
  if (loading) {
    return (
      <div className="charrette-loading">
        <div className="charrette-loading-icon">
          <span className="charrette-spinner" />
        </div>
        <p className="charrette-loading-title">Generating plans with MOE AI…</p>
        <p className="charrette-loading-sub">8 expert agents analyzing your constraints</p>
        <div className="charrette-loading-dots">
          <span /><span /><span />
        </div>
      </div>
    )
  }

  if (plans.length === 0) {
    return (
      <div className="charrette-empty">
        <div className="charrette-empty-logo">
          <svg viewBox="0 0 40 40" fill="none">
            <rect x="4" y="18" width="32" height="18" rx="2" stroke="currentColor" strokeWidth="1.5"/>
            <path d="M2 20 L20 6 L38 20" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round"/>
            <rect x="15" y="24" width="10" height="12" stroke="currentColor" strokeWidth="1.2"/>
            <rect x="7" y="22" width="7" height="6" stroke="currentColor" strokeWidth="1.2"/>
            <rect x="26" y="22" width="7" height="6" stroke="currentColor" strokeWidth="1.2"/>
          </svg>
        </div>
        <h2 className="charrette-empty-title">Choose a Starting Point</h2>
        <p className="charrette-empty-sub">Set your constraints and generate AI-designed floor plan variants</p>
        <div className="charrette-steps">
          {STEPS.map(s => (
            <div key={s.n} className="charrette-step">
              <div className="charrette-step-num">{s.n}</div>
              <div className="charrette-step-text">{s.text}</div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="charrette-board">
      <div className="charrette-header">
        <h2 className="charrette-title">Choose a Starting Point</h2>
        <p className="charrette-subtitle">Select one of these floor plan variations to begin evolving your design</p>
      </div>

      <div className="charrette-grid">
        {plans.map((plan, idx) => {
          const beds = bedroomCount(plan)
          const baths = bathroomCount(plan)
          const sqft = totalSqft(plan)
          const label = CARD_LABELS[idx] ?? String(idx + 1)
          const fp = `${Math.round(plan.totalWidth)}' × ${Math.round(plan.totalHeight)}'`

          return (
            <div key={plan.id} className="charrette-card" onClick={() => onSelect(plan)}>
              <div className="charrette-card-label">{label}</div>
              <div className="charrette-card-preview">
                <FloorPlanPreview plan={plan} width={280} height={200} />
              </div>
              <div className="charrette-card-footer">
                <div className="charrette-card-stats">
                  <span className="charrette-stat">{sqft.toLocaleString()} sqft</span>
                  <span className="charrette-stat-sep">·</span>
                  {beds > 0 && <span className="charrette-stat">{beds} bed</span>}
                  {beds > 0 && baths > 0 && <span className="charrette-stat-sep">·</span>}
                  {baths > 0 && <span className="charrette-stat">{baths} bath</span>}
                </div>
                <div className="charrette-card-dims">{fp} · {plan.rooms.length} rooms</div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}
