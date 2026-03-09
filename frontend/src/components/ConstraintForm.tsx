import { useState } from 'react'
import { Constraints } from '../types/floorplan'

interface Props {
  onGenerate: (c: Constraints, useMOE?: boolean) => void
  loading: boolean
}

const DEFAULT: Constraints = {
  bedrooms: 3,
  bathrooms: 2,
  sqft: 1800,
  stories: 1,
  style: 'modern',
  openPlan: false,
  primarySuite: true,
  homeOffice: false,
  formalDining: false,
  garage: '2car',
  laundry: 'room',
  outdoor: 'patio',
  ceilingHeight: 'standard',
}

function Stepper({ value, min, max, onChange }: { value: number; min: number; max: number; onChange: (v: number) => void }) {
  return (
    <div className="stepper">
      <button onClick={() => onChange(Math.max(min, value - 1))}>−</button>
      <span>{value}</span>
      <button onClick={() => onChange(Math.min(max, value + 1))}>+</button>
    </div>
  )
}

function Toggle({ on, onToggle }: { on: boolean; onToggle: () => void }) {
  return (
    <button className={`toggle-btn ${on ? 'on' : ''}`} onClick={onToggle} aria-label="toggle" />
  )
}

function OptionGroup<T extends string>({
  value,
  options,
  onChange,
}: {
  value: T
  options: { value: T; label: string }[]
  onChange: (v: T) => void
}) {
  return (
    <div className="option-group">
      {options.map(o => (
        <button
          key={o.value}
          className={`option-btn ${value === o.value ? 'active' : ''}`}
          onClick={() => onChange(o.value)}
        >
          {o.label}
        </button>
      ))}
    </div>
  )
}

export default function ConstraintForm({ onGenerate, loading }: Props) {
  const [c, setC] = useState<Constraints>(DEFAULT)
  const [useMOE, setUseMOE] = useState(true)
  const set = <K extends keyof Constraints>(field: K, value: Constraints[K]) =>
    setC(prev => ({ ...prev, [field]: value }))

  return (
    <div className="constraint-form">

      {/* ── Basics ── */}
      <div className="form-section">
        <div className="form-section-title">Basics</div>

        <div className="form-row">
          <label>Bedrooms</label>
          <Stepper value={c.bedrooms} min={1} max={8} onChange={v => set('bedrooms', v)} />
        </div>

        <div className="form-row">
          <label>Bathrooms</label>
          <Stepper value={c.bathrooms} min={1} max={6} onChange={v => set('bathrooms', v)} />
        </div>

        <div className="form-row">
          <label>Stories</label>
          <Stepper value={c.stories} min={1} max={3} onChange={v => set('stories', v)} />
        </div>

        <div className="form-group">
          <label>Square Footage</label>
          <input
            type="range" min={800} max={6000} step={100}
            value={c.sqft}
            onChange={e => set('sqft', parseInt(e.target.value))}
          />
          <span className="range-value">{c.sqft.toLocaleString()} sq ft</span>
        </div>
      </div>

      {/* ── Layout ── */}
      <div className="form-section">
        <div className="form-section-title">Layout</div>

        <div className="toggle-row">
          <span className="toggle-label">Open Kitchen / Living</span>
          <Toggle on={c.openPlan} onToggle={() => set('openPlan', !c.openPlan)} />
        </div>

        <div className="toggle-row">
          <span className="toggle-label">Primary Suite (ensuite + closet)</span>
          <Toggle on={c.primarySuite} onToggle={() => set('primarySuite', !c.primarySuite)} />
        </div>

        <div className="toggle-row">
          <span className="toggle-label">Home Office</span>
          <Toggle on={c.homeOffice} onToggle={() => set('homeOffice', !c.homeOffice)} />
        </div>

        <div className="toggle-row">
          <span className="toggle-label">Formal Dining Room</span>
          <Toggle on={c.formalDining} onToggle={() => set('formalDining', !c.formalDining)} />
        </div>
      </div>

      {/* ── Spaces ── */}
      <div className="form-section">
        <div className="form-section-title">Spaces</div>

        <div className="form-group">
          <label>Garage</label>
          <OptionGroup
            value={c.garage}
            options={[
              { value: 'none', label: 'None' },
              { value: '1car', label: '1-Car' },
              { value: '2car', label: '2-Car' },
              { value: '3car', label: '3-Car' },
            ]}
            onChange={v => set('garage', v)}
          />
        </div>

        <div className="form-group">
          <label>Laundry</label>
          <OptionGroup
            value={c.laundry}
            options={[
              { value: 'none', label: 'None' },
              { value: 'closet', label: 'Closet' },
              { value: 'room', label: 'Room' },
            ]}
            onChange={v => set('laundry', v)}
          />
        </div>

        <div className="form-group">
          <label>Outdoor Space</label>
          <OptionGroup
            value={c.outdoor}
            options={[
              { value: 'none', label: 'None' },
              { value: 'patio', label: 'Patio' },
              { value: 'deck', label: 'Deck' },
              { value: 'both', label: 'Both' },
            ]}
            onChange={v => set('outdoor', v)}
          />
        </div>
      </div>

      {/* ── Style ── */}
      <div className="form-section">
        <div className="form-section-title">Style</div>

        <div className="form-group">
          <label>Architecture</label>
          <select value={c.style} onChange={e => set('style', e.target.value)}>
            <option value="modern">Modern</option>
            <option value="traditional">Traditional</option>
            <option value="craftsman">Craftsman</option>
            <option value="ranch">Ranch</option>
            <option value="contemporary">Contemporary</option>
            <option value="colonial">Colonial</option>
            <option value="farmhouse">Farmhouse</option>
            <option value="mediterranean">Mediterranean</option>
          </select>
        </div>

        <div className="form-group">
          <label>Ceiling Height</label>
          <OptionGroup
            value={c.ceilingHeight}
            options={[
              { value: 'standard', label: '9 ft' },
              { value: 'high', label: '10 ft' },
              { value: 'vaulted', label: 'Vaulted' },
            ]}
            onChange={v => set('ceilingHeight', v)}
          />
        </div>
      </div>

      {/* ── AI Generator Toggle ── */}
      <div className="form-section">
        <div className="form-section-title">Generator</div>
        <div className="moe-toggle-container">
          <button
            className={`moe-toggle-btn ${useMOE ? 'active' : ''}`}
            onClick={() => setUseMOE(true)}
          >
            <span className="moe-icon">🧠</span>
            <span>AI-Powered</span>
            <span className="moe-badge">MOE</span>
          </button>
          <button
            className={`moe-toggle-btn ${!useMOE ? 'active' : ''}`}
            onClick={() => setUseMOE(false)}
          >
            <span className="moe-icon">⚡</span>
            <span>Classic</span>
          </button>
        </div>
        {useMOE && (
          <div className="moe-info">
            8 specialized AI experts analyze your constraints for optimal layouts.
            IRC compliant · Multi-stage refinement
          </div>
        )}
      </div>

      <div className="form-section generate-section">
        <button
          className={`generate-btn ${useMOE ? 'moe-glow' : ''}`}
          onClick={() => onGenerate(c, useMOE)}
          disabled={loading}
        >
          {loading
            ? <><span className="spinner" /> {useMOE ? 'AI Generating...' : 'Generating...'}</>
            : useMOE
              ? '🧠 Generate with MOE AI'
              : '✦ Generate Plans'
          }
        </button>
      </div>

    </div>
  )
}
