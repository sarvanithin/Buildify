import { useState } from 'react'
import { FloorPlan, Room } from '../types/floorplan'

interface Props {
  plan: FloorPlan
}

type ElevSide = 'front' | 'rear' | 'left' | 'right'

const SIDE_LABELS: Record<ElevSide, string> = {
  front: 'Front Elevation',
  rear:  'Rear Elevation',
  left:  'Left Side Elevation',
  right: 'Right Side Elevation',
}

export default function ElevationView({ plan }: Props) {
  const [side, setSide] = useState<ElevSide>('front')

  const W = 800
  const H = 420
  const pad = 48
  const ceilH = plan.ceilingHeight ?? 9

  // Elevation width = totalWidth (front/rear) or totalHeight (left/right)
  const elevWidth  = (side === 'front' || side === 'rear')  ? plan.totalWidth  : plan.totalHeight
  const scaleX = (W - pad * 2) / elevWidth
  const scaleY = (H - pad - 80) / (ceilH + 6)   // leave room for roof
  const scale  = Math.min(scaleX, scaleY)

  const groundY = H - pad - 20
  const wallTop  = groundY - ceilH * scale
  const roofPeak = wallTop - (elevWidth * 0.18 * scale)  // ~18% pitch
  const ox = (W - elevWidth * scale) / 2

  // Rooms projected onto chosen facade
  const facadeRooms = projectRooms(plan.rooms, side, plan.totalWidth, plan.totalHeight)

  // Collect windows by room type
  function windowsForRoom(room: Room & { _elevX: number; _elevW: number }) {
    const rtype = room.type.toLowerCase()
    const isPublic  = ['living','dining','kitchen','office','family'].some(t => rtype.includes(t))
    const isBedroom = rtype.includes('bedroom')
    const isBath    = rtype.includes('bathroom') || rtype.includes('bath')
    const isGarage  = rtype.includes('garage')

    const elevW = room._elevW * scale
    if (elevW < 18 || isGarage) return []

    const wins: { cx: number; w: number; h: number; top: number }[] = []
    const winW = isPublic ? Math.min(60, elevW * 0.4) : Math.min(40, elevW * 0.3)
    const winH = isPublic ? 60 : isBedroom ? 48 : isBath ? 28 : 40
    const winTop = wallTop + (isBath ? ceilH * scale * 0.45 : ceilH * scale * 0.28)

    const n = isPublic ? Math.max(1, Math.floor(room._elevW / 10)) : 1
    for (let i = 0; i < Math.min(n, 3); i++) {
      const cx = ox + room._elevX * scale + elevW / (n + 1) * (i + 1)
      wins.push({ cx, w: winW, h: winH, top: winTop })
    }
    return wins
  }

  const doorX  = ox + elevWidth * 0.35 * scale
  const doorW  = 40, doorH = 84
  const doorTop = groundY - doorH

  const chimneyX = ox + elevWidth * 0.72 * scale
  const chimneyW = 18

  return (
    <div className="elevation-wrap">
      <div className="elev-tabs">
        {(['front','rear','left','right'] as ElevSide[]).map(s => (
          <button key={s} className={`elev-tab ${side === s ? 'active' : ''}`} onClick={() => setSide(s)}>
            {s.charAt(0).toUpperCase() + s.slice(1)}
          </button>
        ))}
      </div>

      <svg width={W} height={H} className="elevation-svg">
        {/* Sky gradient */}
        <defs>
          <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#D6E8F5" />
            <stop offset="100%" stopColor="#F0F5F9" />
          </linearGradient>
          <linearGradient id="wallGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#F5F0E8" />
            <stop offset="100%" stopColor="#EDE7D9" />
          </linearGradient>
        </defs>

        <rect width={W} height={H} fill="url(#skyGrad)" />

        {/* Ground */}
        <rect x={0} y={groundY} width={W} height={H - groundY} fill="#C8D8A8" />
        <line x1={0} y1={groundY} x2={W} y2={groundY} stroke="#8A9E6A" strokeWidth={2} />

        {/* Chimney */}
        <rect x={chimneyX} y={roofPeak - 40} width={chimneyW} height={50}
          fill="#B0897A" stroke="#8A6B5E" strokeWidth={1} />
        <rect x={chimneyX - 3} y={roofPeak - 42} width={chimneyW + 6} height={8}
          fill="#8A6B5E" />

        {/* Roof fill */}
        <polygon
          points={`${ox},${wallTop} ${ox + elevWidth * scale},${wallTop} ${ox + elevWidth * scale / 2},${roofPeak}`}
          fill="#6B4E3D"
        />
        {/* Roof overhang lines */}
        <line x1={ox - 12} y1={wallTop} x2={ox + elevWidth * scale + 12} y2={wallTop}
          stroke="#4A3428" strokeWidth={3} />
        {/* Fascia */}
        <line x1={ox - 12} y1={wallTop} x2={ox + elevWidth * scale / 2} y2={roofPeak}
          stroke="#4A3428" strokeWidth={2} />
        <line x1={ox + elevWidth * scale + 12} y1={wallTop} x2={ox + elevWidth * scale / 2} y2={roofPeak}
          stroke="#4A3428" strokeWidth={2} />

        {/* Wall body */}
        <rect x={ox} y={wallTop} width={elevWidth * scale} height={ceilH * scale}
          fill="url(#wallGrad)" stroke="#888" strokeWidth={1.5} />

        {/* Rooms projected + windows */}
        {facadeRooms.map(room => {
          const wins = windowsForRoom(room)
          return wins.map((w, i) => (
            <g key={`${room.id}-w${i}`}>
              {/* Window frame */}
              <rect x={w.cx - w.w/2} y={w.top} width={w.w} height={w.h}
                fill="#C8DCE8" stroke="#7A9CB0" strokeWidth={1.5} />
              {/* Mullion */}
              <line x1={w.cx} y1={w.top} x2={w.cx} y2={w.top + w.h} stroke="#7A9CB0" strokeWidth={1} />
              <line x1={w.cx - w.w/2} y1={w.top + w.h/2} x2={w.cx + w.w/2} y2={w.top + w.h/2}
                stroke="#7A9CB0" strokeWidth={1} />
              {/* Glare */}
              <line x1={w.cx - w.w/2 + 3} y1={w.top + 4} x2={w.cx - 4} y2={w.top + w.h/3}
                stroke="rgba(255,255,255,0.6)" strokeWidth={1.5} />
            </g>
          ))
        })}

        {/* Front door (only on front elevation) */}
        {side === 'front' && (
          <g>
            <rect x={doorX - doorW/2} y={doorTop} width={doorW} height={doorH}
              fill="#7A5C3E" stroke="#5A3C1E" strokeWidth={1.5} rx={2} />
            <rect x={doorX - doorW/2 + 3} y={doorTop + 5} width={doorW - 6} height={40}
              fill="#8B6A48" stroke="#5A3C1E" strokeWidth={0.5} />
            <circle cx={doorX + doorW/2 - 6} cy={doorTop + doorH/2} r={3} fill="#D4AF37" />
            {/* Door arch */}
            <path d={`M ${doorX-doorW/2} ${doorTop} A ${doorW/2} ${doorW/3} 0 0 1 ${doorX+doorW/2} ${doorTop}`}
              fill="#9B7A52" stroke="#5A3C1E" strokeWidth={1} />
          </g>
        )}

        {/* Steps / stoop */}
        {side === 'front' && (
          <g>
            <rect x={doorX - doorW/2 - 8} y={groundY - 12} width={doorW + 16} height={12}
              fill="#C8B89A" stroke="#A09278" strokeWidth={1} />
            <rect x={doorX - doorW/2 - 14} y={groundY - 6} width={doorW + 28} height={6}
              fill="#BCA888" stroke="#A09278" strokeWidth={1} />
          </g>
        )}

        {/* Trees */}
        {[0.12, 0.88].map((tx, i) => {
          const tx_ = ox + tx * elevWidth * scale
          return (
            <g key={`tree${i}`}>
              <polygon points={`${tx_},${groundY - 90} ${tx_ - 30},${groundY - 20} ${tx_ + 30},${groundY - 20}`}
                fill="#4A7C3F" />
              <polygon points={`${tx_},${groundY - 115} ${tx_ - 22},${groundY - 55} ${tx_ + 22},${groundY - 55}`}
                fill="#5B9250" />
              <rect x={tx_ - 5} y={groundY - 20} width={10} height={20} fill="#6B4E2E" />
            </g>
          )
        })}

        {/* Dimension line */}
        <line x1={ox} y1={H - 24} x2={ox + elevWidth * scale} y2={H - 24}
          stroke="#555" strokeWidth={1} />
        <line x1={ox} y1={H - 30} x2={ox} y2={H - 18} stroke="#555" strokeWidth={1} />
        <line x1={ox + elevWidth * scale} y1={H - 30} x2={ox + elevWidth * scale} y2={H - 18} stroke="#555" strokeWidth={1} />
        <text x={ox + elevWidth * scale / 2} y={H - 10} textAnchor="middle"
          fontSize={11} fill="#555">{`${Math.round(elevWidth)}'`}</text>

        {/* Height dimension */}
        <line x1={ox - 18} y1={wallTop} x2={ox - 18} y2={groundY}
          stroke="#555" strokeWidth={1} />
        <text x={ox - 28} y={groundY - ceilH * scale / 2}
          textAnchor="middle" fontSize={11} fill="#555"
          transform={`rotate(-90 ${ox - 28} ${groundY - ceilH * scale / 2})`}>
          {`${ceilH}'`}
        </text>

        {/* Title */}
        <text x={W / 2} y={22} textAnchor="middle" fontSize={13} fontWeight="600" fill="#333">
          {SIDE_LABELS[side]} — {plan.name}
        </text>

        {/* Scale note */}
        <text x={W - 12} y={H - 8} textAnchor="end" fontSize={9} fill="#999">
          NTS – illustrative
        </text>
      </svg>
    </div>
  )
}

function projectRooms(
  rooms: Room[],
  side: ElevSide,
  totalW: number,
  totalH: number
): (Room & { _elevX: number; _elevW: number })[] {
  return rooms.map(r => {
    let elevX: number, elevW: number
    if (side === 'front' || side === 'rear') {
      elevX = r.x
      elevW = r.width
    } else {
      elevX = r.y
      elevW = r.height
    }
    return { ...r, _elevX: elevX, _elevW: elevW }
  })
}
