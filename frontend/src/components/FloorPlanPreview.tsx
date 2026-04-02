import { FloorPlan } from '../types/floorplan'

interface Props {
  plan: FloorPlan
  width?: number
  height?: number
}

// Cool, readable fills for plan previews (distinct category hues)
const TYPE_COLORS: Record<string, string> = {
  master_bedroom:    '#e8d4dc',
  primary_bedroom:   '#e8d4dc',
  bedroom:           '#c8d4e8',
  bathroom:          '#a8cfe8',
  ensuite_bathroom:  '#a8cfe8',
  half_bath:         '#b8e0e4',
  primary_bath:      '#a8cfe8',
  closet:            '#d8dce8',
  walk_in_closet:    '#d8dce8',
  primary_closet:    '#d8dce8',

  kitchen:           '#e2d9a8',
  dining_room:       '#d4cfa0',
  living_room:       '#a8d0bc',
  great_room:        '#a8d0bc',
  family_room:       '#b8dcc4',
  foyer:             '#d8e0c8',
  entry_foyer:       '#d8e0c8',
  home_office:       '#c4c8e8',
  bar:               '#dcc8a8',
  nook:              '#d0d8c0',
  sunroom:           '#c0e0c8',

  hallway:           '#d8dce4',
  laundry_room:      '#b8d4e8',
  mudroom:           '#d0d4cc',
  pantry:            '#e0dcc8',
  utility_room:      '#ccd0d0',
  garage:            '#c4c8c4',

  patio:             '#a8d8bc',
  deck:              '#a8d8bc',
  front_porch:       '#98d0a8',
  outdoor_living:    '#98d0a8',
  rear_patio:        '#a8d8bc',
}

function getRoomColor(room: FloorPlan['rooms'][number]): string {
  const type = room.type?.toLowerCase() ?? ''
  const name = room.name?.toLowerCase() ?? ''

  // Try exact type match
  for (const [key, color] of Object.entries(TYPE_COLORS)) {
    if (type === key) return color
  }
  // Try name-based match
  if (name.includes('primary bed') || name.includes('master bed')) return TYPE_COLORS.master_bedroom
  if (name.includes('bedroom') || name.includes('bed')) return TYPE_COLORS.bedroom
  if (name.includes('bath') || name.includes('toilet')) return TYPE_COLORS.bathroom
  if (name.includes('closet')) return TYPE_COLORS.closet
  if (name.includes('kitchen')) return TYPE_COLORS.kitchen
  if (name.includes('living')) return TYPE_COLORS.living_room
  if (name.includes('dining')) return TYPE_COLORS.dining_room
  if (name.includes('garage')) return TYPE_COLORS.garage
  if (name.includes('patio') || name.includes('deck') || name.includes('porch')) return TYPE_COLORS.patio
  if (name.includes('hall')) return TYPE_COLORS.hallway
  if (name.includes('laundry')) return TYPE_COLORS.laundry_room
  if (name.includes('foyer') || name.includes('entry')) return TYPE_COLORS.foyer

  // Fall back to room.color (from backend) or default
  return room.color || '#dce0e8'
}

export default function FloorPlanPreview({ plan, width = 300, height = 210 }: Props) {
  const pad = 10
  const scaleX = (width - pad * 2) / plan.totalWidth
  const scaleY = (height - pad * 2) / plan.totalHeight
  const scale = Math.min(scaleX, scaleY)
  const ox = (width - plan.totalWidth * scale) / 2
  const oy = (height - plan.totalHeight * scale) / 2

  const WALL = Math.max(0.6, scale * 0.15)

  return (
    <svg width={width} height={height} style={{ display: 'block', background: '#e8eef5' }}>
      {/* Outer footprint shadow */}
      <rect
        x={ox + 2} y={oy + 2}
        width={plan.totalWidth * scale}
        height={plan.totalHeight * scale}
        fill="rgba(0,0,0,0.06)" rx={1}
      />
      {/* Outer footprint border */}
      <rect
        x={ox} y={oy}
        width={plan.totalWidth * scale}
        height={plan.totalHeight * scale}
        fill="#dce4ee" stroke="#94a3b8" strokeWidth={1.2} rx={1}
      />

      {/* Rooms */}
      {plan.rooms.map(room => {
        const rx = ox + room.x * scale
        const ry = oy + room.y * scale
        const rw = room.width * scale
        const rh = room.height * scale
        const color = getRoomColor(room)

        const showLabel = rw > 32 && rh > 16
        const shortName = room.name.length > 12
          ? room.name.split(' ').map(w => w[0]).join('').toUpperCase()
          : room.name.split(' ').slice(0, 2).join(' ')

        return (
          <g key={room.id}>
            <rect
              x={rx} y={ry} width={rw} height={rh}
              fill={color}
              stroke="#B0ACA4"
              strokeWidth={WALL}
            />
            {showLabel && (
              <text
                x={rx + rw / 2} y={ry + rh / 2}
                textAnchor="middle" dominantBaseline="middle"
                fontSize={Math.max(6, Math.min(9, rw / (shortName.length * 0.7)))}
                fill="#4A4540"
                fontFamily="system-ui, -apple-system, sans-serif"
                fontWeight="500"
                style={{ pointerEvents: 'none', userSelect: 'none' }}
              >
                {shortName}
              </text>
            )}
          </g>
        )
      })}
    </svg>
  )
}
