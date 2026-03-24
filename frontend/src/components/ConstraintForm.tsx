import { useState } from 'react'
import { Constraints } from '../types/floorplan'

interface Props {
  onGenerate: (c: Constraints, useMOE?: boolean) => void
  loading: boolean
}

// ─── Room Catalog Definition ──────────────────────────────────────────────────

interface RoomDef {
  key: string
  label: string
  icon: string
  constraintKey?: keyof Constraints
  isToggle?: boolean
  toggleOnValue?: string
}

interface RoomSection {
  title: string
  rooms: RoomDef[]
}

const CATALOG: RoomSection[] = [
  {
    title: 'Beds & Baths',
    rooms: [
      { key: 'primary_bed', label: 'Primary Bed', icon: '🛏', constraintKey: 'primarySuite', isToggle: true },
      { key: 'bedrooms', label: 'Bedroom', icon: '🛏' },
      { key: 'bathrooms', label: 'Bathroom', icon: '🚿' },
      // Half bath toggled via formalDining for demo; expose as optional
    ],
  },
  {
    title: 'Living Spaces',
    rooms: [
      { key: 'kitchen', label: 'Kitchen', icon: '🍳' },
      { key: 'dining', label: 'Dining', icon: '🍽' },
      { key: 'living', label: 'Living Room', icon: '🛋' },
      { key: 'office', label: 'Office', icon: '💻', constraintKey: 'homeOffice', isToggle: true },
      { key: 'formalDining', label: 'Formal Dining', icon: '🥂', constraintKey: 'formalDining', isToggle: true },
    ],
  },
  {
    title: 'Outdoor Spaces',
    rooms: [
      { key: 'outdoor', label: 'Patio / Deck', icon: '🌿' },
    ],
  },
  {
    title: 'Garage & Utility',
    rooms: [
      { key: 'garage', label: 'Garage', icon: '🚗' },
      { key: 'laundry', label: 'Laundry', icon: '🧺' },
    ],
  },
]

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
    <div className="catalog-stepper">
      <button
        className="catalog-stepper-btn"
        onClick={() => onChange(Math.max(min, value - 1))}
        disabled={value <= min}
      >−</button>
      <span className="catalog-stepper-val">{value}</span>
      <button
        className="catalog-stepper-btn"
        onClick={() => onChange(Math.min(max, value + 1))}
        disabled={value >= max}
      >+</button>
    </div>
  )
}

type GarageValue = 'none' | '1car' | '2car' | '3car'
function GarageRow({ value, onChange }: { value: string; onChange: (v: GarageValue) => void }) {
  const options: { v: GarageValue; label: string }[] = [
    { v: 'none', label: 'None' },
    { v: '1car', label: '1-Car' },
    { v: '2car', label: '2-Car' },
    { v: '3car', label: '3-Car' },
  ]
  return (
    <div className="catalog-option-chips">
      {options.map(o => (
        <button
          key={o.v}
          className={`catalog-chip ${value === o.v ? 'active' : ''}`}
          onClick={() => onChange(o.v)}
        >{o.label}</button>
      ))}
    </div>
  )
}

type LaundryValue = 'none' | 'closet' | 'room'
function LaundryRow({ value, onChange }: { value: string; onChange: (v: LaundryValue) => void }) {
  const options: { v: LaundryValue; label: string }[] = [
    { v: 'none', label: 'None' },
    { v: 'closet', label: 'Closet' },
    { v: 'room', label: 'Room' },
  ]
  return (
    <div className="catalog-option-chips">
      {options.map(o => (
        <button
          key={o.v}
          className={`catalog-chip ${value === o.v ? 'active' : ''}`}
          onClick={() => onChange(o.v)}
        >{o.label}</button>
      ))}
    </div>
  )
}

type OutdoorValue = 'none' | 'patio' | 'deck' | 'both'
function OutdoorRow({ value, onChange }: { value: string; onChange: (v: OutdoorValue) => void }) {
  const options: { v: OutdoorValue; label: string }[] = [
    { v: 'none', label: 'None' },
    { v: 'patio', label: 'Patio' },
    { v: 'deck', label: 'Deck' },
    { v: 'both', label: 'Both' },
  ]
  return (
    <div className="catalog-option-chips">
      {options.map(o => (
        <button
          key={o.v}
          className={`catalog-chip ${value === o.v ? 'active' : ''}`}
          onClick={() => onChange(o.v)}
        >{o.label}</button>
      ))}
    </div>
  )
}

export default function ConstraintForm({ onGenerate, loading }: Props) {
  const [c, setC] = useState<Constraints>(DEFAULT)
  const set = <K extends keyof Constraints>(field: K, value: Constraints[K]) =>
    setC(prev => ({ ...prev, [field]: value }))

  const sqftLabel = c.sqft >= 1000
    ? `${(c.sqft / 1000).toFixed(1).replace('.0', '')}k`
    : `${c.sqft}`

  return (
    <div className="room-catalog">

      {/* Header */}
      <div className="catalog-header">
        <h3 className="catalog-heading">My Room List</h3>
        <div className="catalog-size-target">
          <span className="catalog-size-label">Total Size Target</span>
          <div className="catalog-size-control">
            <input
              type="range" min={800} max={6000} step={100}
              value={c.sqft}
              onChange={e => set('sqft', parseInt(e.target.value))}
              className="catalog-sqft-slider"
            />
            <span className="catalog-size-value">{sqftLabel} sqft</span>
          </div>
        </div>
        <div className="catalog-meta-row">
          <span className="catalog-meta-item">
            <span className="catalog-meta-icon">⬜</span>
            {c.stories === 1 ? '1 Story' : `${c.stories} Stories`}
          </span>
          <select
            className="catalog-stories-select"
            value={c.stories}
            onChange={e => set('stories', parseInt(e.target.value))}
          >
            <option value={1}>1 Story</option>
            <option value={2}>2 Stories</option>
          </select>
        </div>
      </div>

      {/* Style */}
      <div className="catalog-section">
        <div className="catalog-section-title">Style</div>
        <div className="catalog-style-row">
          {['modern', 'traditional', 'craftsman', 'ranch', 'farmhouse', 'contemporary'].map(s => (
            <button
              key={s}
              className={`catalog-style-chip ${c.style === s ? 'active' : ''}`}
              onClick={() => set('style', s)}
            >{s.charAt(0).toUpperCase() + s.slice(1)}</button>
          ))}
        </div>
      </div>

      {/* ── Beds & Baths ── */}
      <div className="catalog-section">
        <div className="catalog-section-title">Beds &amp; Baths</div>

        <div className="catalog-row">
          <span className="catalog-row-icon">🛏</span>
          <span className="catalog-row-label">Primary Bed</span>
          <div className="catalog-row-right">
            <button
              className={`catalog-toggle ${c.primarySuite ? 'on' : ''}`}
              onClick={() => set('primarySuite', !c.primarySuite)}
            >
              {c.primarySuite ? 'L' : '–'}
            </button>
          </div>
        </div>

        <div className="catalog-row">
          <span className="catalog-row-icon">🛏</span>
          <span className="catalog-row-label">Bedroom</span>
          <div className="catalog-row-right">
            <Stepper value={c.bedrooms - (c.primarySuite ? 1 : 0)} min={0} max={7}
              onChange={v => set('bedrooms', v + (c.primarySuite ? 1 : 0))} />
          </div>
        </div>

        <div className="catalog-row">
          <span className="catalog-row-icon">🚿</span>
          <span className="catalog-row-label">Bathroom</span>
          <div className="catalog-row-right">
            <Stepper value={c.bathrooms} min={1} max={6}
              onChange={v => set('bathrooms', v)} />
          </div>
        </div>
      </div>

      {/* ── Living Spaces ── */}
      <div className="catalog-section">
        <div className="catalog-section-title">Living Spaces</div>

        <div className="catalog-row">
          <span className="catalog-row-icon">🍳</span>
          <span className="catalog-row-label">Kitchen</span>
          <span className="catalog-row-size">M</span>
        </div>
        <div className="catalog-row">
          <span className="catalog-row-icon">🛋</span>
          <span className="catalog-row-label">Living Room</span>
          <span className="catalog-row-size">M</span>
        </div>
        <div className="catalog-row">
          <span className="catalog-row-icon">🚪</span>
          <span className="catalog-row-label">Foyer</span>
          <span className="catalog-row-size">M</span>
        </div>

        <div className="catalog-row">
          <span className="catalog-row-icon">🍽</span>
          <span className="catalog-row-label">Dining</span>
          <div className="catalog-row-right">
            <button
              className={`catalog-toggle ${c.formalDining ? 'on' : ''}`}
              onClick={() => set('formalDining', !c.formalDining)}
            >
              {c.formalDining ? 'M' : '–'}
            </button>
          </div>
        </div>

        <div className="catalog-row">
          <span className="catalog-row-icon">💻</span>
          <span className="catalog-row-label">Office</span>
          <div className="catalog-row-right">
            <button
              className={`catalog-toggle ${c.homeOffice ? 'on' : ''}`}
              onClick={() => set('homeOffice', !c.homeOffice)}
            >
              {c.homeOffice ? 'M' : '–'}
            </button>
          </div>
        </div>

        <div className="catalog-row">
          <span className="catalog-row-icon">🏠</span>
          <span className="catalog-row-label">Open Plan</span>
          <div className="catalog-row-right">
            <button
              className={`catalog-toggle ${c.openPlan ? 'on' : ''}`}
              onClick={() => set('openPlan', !c.openPlan)}
            >
              {c.openPlan ? 'on' : '–'}
            </button>
          </div>
        </div>
      </div>

      {/* ── Outdoor Spaces ── */}
      <div className="catalog-section">
        <div className="catalog-section-title">Outdoor Spaces</div>
        <div className="catalog-row catalog-row-full">
          <span className="catalog-row-icon">🌿</span>
          <span className="catalog-row-label">Patio / Deck</span>
          <div className="catalog-row-right">
            <OutdoorRow value={c.outdoor} onChange={v => set('outdoor', v)} />
          </div>
        </div>
      </div>

      {/* ── Garage & Utility ── */}
      <div className="catalog-section">
        <div className="catalog-section-title">Garage &amp; Utility</div>
        <div className="catalog-row catalog-row-full">
          <span className="catalog-row-icon">🚗</span>
          <span className="catalog-row-label">Garage</span>
          <div className="catalog-row-right">
            <GarageRow value={c.garage} onChange={v => set('garage', v)} />
          </div>
        </div>
        <div className="catalog-row catalog-row-full">
          <span className="catalog-row-icon">🧺</span>
          <span className="catalog-row-label">Laundry</span>
          <div className="catalog-row-right">
            <LaundryRow value={c.laundry} onChange={v => set('laundry', v)} />
          </div>
        </div>
      </div>

      {/* ── Generate ── */}
      <div className="catalog-generate">
        <button
          className="catalog-generate-btn"
          onClick={() => onGenerate(c, true)}
          disabled={loading}
        >
          {loading
            ? <><span className="catalog-spinner" /> Generating…</>
            : <><span className="catalog-update-icon">✦</span> Update</>
          }
        </button>
      </div>

    </div>
  )
}
