import { FloorPlan } from '../types/floorplan'

interface Props {
  plan: FloorPlan
  width?: number
  height?: number
}

// Warm, architectural pastel palette matching Drafted.ai's color coding
const TYPE_COLORS: Record<string, string> = {
  // Beds & Baths — warm pinks/purples
  master_bedroom:    '#F5C5C5',
  primary_bedroom:   '#F5C5C5',
  bedroom:           '#DCCDE8',
  bathroom:          '#B8D8E8',
  ensuite_bathroom:  '#B8D8E8',
  half_bath:         '#C8E8E8',
  primary_bath:      '#B8D8E8',
  closet:            '#F0C8A0',
  walk_in_closet:    '#F0C8A0',
  primary_closet:    '#F0C8A0',

  // Living spaces — golds/yellows/greens
  kitchen:           '#F0E088',
  dining_room:       '#E8D478',
  living_room:       '#C8E0A0',
  great_room:        '#C8E0A0',
  family_room:       '#D8E898',
  foyer:             '#F0D8A0',
  entry_foyer:       '#F0D8A0',
  home_office:       '#D0C8F0',
  bar:               '#F0D0B0',
  nook:              '#E8E0A8',
  sunroom:           '#D8F0C0',

  // Service
  hallway:           '#E0E0D8',
  laundry_room:      '#D0E8F0',
  mudroom:           '#E8D8C8',
  pantry:            '#F0E8D0',
  utility_room:      '#D8D8D0',
  garage:            '#D0D0C8',

  // Outdoor — soft greens
  patio:             '#C8E8C0',
  deck:              '#C8E8C0',
  front_porch:       '#B8E0B8',
  outdoor_living:    '#B8E0B8',
  rear_patio:        '#C8E8C0',
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
  return room.color || '#E8E4DC'
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
    <svg width={width} height={height} style={{ display: 'block', background: '#F5F2EC' }}>
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
        fill="#EDEBE5" stroke="#C0BCB4" strokeWidth={1.2} rx={1}
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
