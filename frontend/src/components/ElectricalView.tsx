/**
 * Electrical schematic overlay — drawn in SVG on top of the floor plan grid.
 * Auto-generates NEC-compliant outlet/switch/lighting placement from room data.
 */
import { FloorPlan, Room } from '../types/floorplan'

interface Props {
  plan: FloorPlan
  scale: number
  ox: number
  oy: number
}

interface Symbol {
  kind: 'outlet' | 'gfci' | '240v' | 'switch' | 'light' | 'smoke' | 'panel'
  x: number
  y: number
  label?: string
}

const COLORS = {
  outlet: '#2563EB',
  gfci:   '#059669',
  '240v': '#DC2626',
  switch: '#7C3AED',
  light:  '#D97706',
  smoke:  '#F97316',
  panel:  '#1E293B',
}

function generateSymbols(rooms: Room[]): Symbol[] {
  const syms: Symbol[] = []

  for (const room of rooms) {
    const rtype = room.type.toLowerCase()
    const cx = room.x + room.width / 2
    const cy = room.y + room.height / 2

    // Ceiling light in every room
    syms.push({ kind: 'light', x: cx, y: cy })

    const isGarage  = rtype.includes('garage')
    const isBath    = rtype.includes('bathroom') || rtype.includes('bath')
    const isKitchen = rtype.includes('kitchen')
    const isLaundry = rtype.includes('laundry')
    const isBedroom = rtype.includes('bedroom')

    // ── Outlets (NEC: every 12ft along wall, ≤6ft from door/corner) ──────────
    const nWide = Math.max(1, Math.floor(room.width  / 12))
    const nDeep = Math.max(1, Math.floor(room.height / 12))

    for (let i = 0; i < nWide; i++) {
      const px = room.x + room.width  / (nWide + 1) * (i + 1)
      // top and bottom wall
      syms.push({ kind: isBath || isKitchen ? 'gfci' : 'outlet', x: px, y: room.y + 0.6 })
      syms.push({ kind: isBath || isKitchen ? 'gfci' : 'outlet', x: px, y: room.y + room.height - 0.6 })
    }
    for (let i = 0; i < nDeep; i++) {
      const py = room.y + room.height / (nDeep + 1) * (i + 1)
      // left and right wall
      syms.push({ kind: isBath || isKitchen ? 'gfci' : 'outlet', x: room.x + 0.6, y: py })
      syms.push({ kind: isBath || isKitchen ? 'gfci' : 'outlet', x: room.x + room.width - 0.6, y: py })
    }

    // 240V outlets in garage and laundry
    if (isGarage || isLaundry) {
      syms.push({ kind: '240v', x: room.x + room.width - 2, y: room.y + room.height - 2, label: '240V' })
    }

    // Switch near doorway (simplified: left side of each room entry)
    syms.push({ kind: 'switch', x: room.x + 2.5, y: room.y + 2.5 })

    // Smoke detector in every bedroom + hallway/foyer
    if (isBedroom || rtype.includes('hallway') || rtype.includes('foyer')) {
      syms.push({ kind: 'smoke', x: cx, y: room.y + room.height * 0.2 })
    }
  }

  return syms
}

// ── SVG symbol renderers ──────────────────────────────────────────────────────

function OutletSymbol({ x, y, color }: { x: number; y: number; color: string }) {
  return (
    <g>
      <circle cx={x} cy={y} r={4} fill="white" stroke={color} strokeWidth={1.5} />
      <line x1={x - 1.5} y1={y - 2.5} x2={x - 1.5} y2={y + 1} stroke={color} strokeWidth={1.2} />
      <line x1={x + 1.5} y1={y - 2.5} x2={x + 1.5} y2={y + 1} stroke={color} strokeWidth={1.2} />
    </g>
  )
}

function GfciSymbol({ x, y }: { x: number; y: number }) {
  return (
    <g>
      <circle cx={x} cy={y} r={4.5} fill="white" stroke={COLORS.gfci} strokeWidth={1.5} />
      <text x={x} y={y + 3.5} textAnchor="middle" fontSize={5} fill={COLORS.gfci} fontWeight="bold">G</text>
    </g>
  )
}

function Outlet240({ x, y }: { x: number; y: number }) {
  return (
    <g>
      <rect x={x - 5} y={y - 5} width={10} height={10} fill="white" stroke={COLORS['240v']} strokeWidth={1.5} />
      <text x={x} y={y + 3} textAnchor="middle" fontSize={5} fill={COLORS['240v']} fontWeight="bold">2</text>
    </g>
  )
}

function SwitchSymbol({ x, y }: { x: number; y: number }) {
  return (
    <g>
      <circle cx={x} cy={y} r={3.5} fill="white" stroke={COLORS.switch} strokeWidth={1.5} />
      <text x={x} y={y + 3} textAnchor="middle" fontSize={5.5} fill={COLORS.switch} fontWeight="bold">S</text>
    </g>
  )
}

function LightSymbol({ x, y }: { x: number; y: number }) {
  return (
    <g>
      <circle cx={x} cy={y} r={5} fill="white" stroke={COLORS.light} strokeWidth={1.5} />
      <line x1={x - 3.5} y1={y - 3.5} x2={x + 3.5} y2={y + 3.5} stroke={COLORS.light} strokeWidth={1.2} />
      <line x1={x + 3.5} y1={y - 3.5} x2={x - 3.5} y2={y + 3.5} stroke={COLORS.light} strokeWidth={1.2} />
    </g>
  )
}

function SmokeSymbol({ x, y }: { x: number; y: number }) {
  return (
    <g>
      <circle cx={x} cy={y} r={4.5} fill="white" stroke={COLORS.smoke} strokeWidth={1.5} />
      <circle cx={x} cy={y} r={2} fill={COLORS.smoke} />
    </g>
  )
}

export default function ElectricalView({ plan, scale, ox, oy }: Props) {
  const symbols = generateSymbols(plan.rooms)

  // Count totals for legend
  const counts = symbols.reduce((acc, s) => {
    acc[s.kind] = (acc[s.kind] || 0) + 1
    return acc
  }, {} as Record<string, number>)

  return (
    <g>
      {symbols.map((sym, i) => {
        const px = ox + sym.x * scale
        const py = oy + sym.y * scale
        if (sym.kind === 'outlet')  return <OutletSymbol key={i} x={px} y={py} color={COLORS.outlet} />
        if (sym.kind === 'gfci')    return <GfciSymbol   key={i} x={px} y={py} />
        if (sym.kind === '240v')    return <Outlet240    key={i} x={px} y={py} />
        if (sym.kind === 'switch')  return <SwitchSymbol key={i} x={px} y={py} />
        if (sym.kind === 'light')   return <LightSymbol  key={i} x={px} y={py} />
        if (sym.kind === 'smoke')   return <SmokeSymbol  key={i} x={px} y={py} />
        return null
      })}
    </g>
  )
}

export function ElectricalLegend({ plan, scale, ox, oy }: Props) {
  const symbols = generateSymbols(plan.rooms)
  const counts = symbols.reduce((acc, s) => { acc[s.kind] = (acc[s.kind] || 0) + 1; return acc }, {} as Record<string, number>)

  const items = [
    { kind: 'outlet', label: `Standard outlet (${counts['outlet'] || 0})` },
    { kind: 'gfci',   label: `GFCI outlet (${counts['gfci'] || 0})` },
    { kind: '240v',   label: `240V outlet (${counts['240v'] || 0})` },
    { kind: 'switch', label: `Wall switch (${counts['switch'] || 0})` },
    { kind: 'light',  label: `Ceiling light (${counts['light'] || 0})` },
    { kind: 'smoke',  label: `Smoke detector (${counts['smoke'] || 0})` },
  ]

  return (
    <div className="elec-legend">
      {items.map(item => (
        <div key={item.kind} className="elec-legend-item">
          <span className="elec-dot" style={{ background: COLORS[item.kind as keyof typeof COLORS] }} />
          <span>{item.label}</span>
        </div>
      ))}
    </div>
  )
}
