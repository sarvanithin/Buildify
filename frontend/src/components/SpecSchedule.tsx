import { FloorPlan, Room } from '../types/floorplan'

interface Props {
  plan: FloorPlan
}

const FLOOR_FINISH: Record<string, string> = {
  master_bedroom: 'Hardwood (oak, stained)',
  bedroom:        'Hardwood or carpet',
  bathroom:       'Porcelain tile 12"×24"',
  ensuite_bathroom: 'Porcelain tile 12"×24"',
  half_bath:      'Ceramic tile',
  kitchen:        'Porcelain tile or LVP',
  living_room:    'Hardwood (oak)',
  dining_room:    'Hardwood (oak)',
  family_room:    'Hardwood or LVP',
  home_office:    'Hardwood or carpet',
  garage:         'Concrete (sealed)',
  laundry_room:   'Ceramic tile',
  foyer:          'Tile or hardwood',
  hallway:        'Hardwood or LVP',
  default:        'LVP (luxury vinyl plank)',
}

const WALL_FINISH: Record<string, string> = {
  bathroom:         'Ceramic tile to 72" + paint',
  ensuite_bathroom: 'Large-format tile to 84" + paint',
  kitchen:          'Paint + tile backsplash',
  garage:           'Drywall (unpainted)',
  default:          'Drywall, eggshell paint',
}

const CEILING_FINISH: Record<string, string> = {
  bathroom:         'Moisture-resistant paint',
  ensuite_bathroom: 'Moisture-resistant paint',
  kitchen:          'Semi-gloss paint',
  garage:           'Exposed or drywall',
  default:          'Flat white paint',
}

function getFinish(map: Record<string, string>, type: string): string {
  const key = type.toLowerCase().replace(/\s+/g, '_')
  return map[key] ?? map['default'] ?? '—'
}

function windowCount(room: Room): number {
  const rtype = room.type.toLowerCase()
  const area = room.width * room.height
  if (rtype.includes('garage')) return 1
  if (rtype.includes('hallway') || rtype.includes('closet') || rtype.includes('pantry')) return 0
  if (area > 200) return 3
  if (area > 120) return 2
  return 1
}

function doorCount(room: Room): number {
  const rtype = room.type.toLowerCase()
  if (rtype.includes('closet') || rtype.includes('pantry')) return 1
  if (rtype.includes('ensuite') || rtype.includes('garage')) return 2
  if (rtype.includes('bedroom') || rtype.includes('bathroom')) return 1
  return 1
}

export default function SpecSchedule({ plan }: Props) {
  const ceilH = plan.ceilingHeight ?? 9
  const totalSqft = Math.round(plan.rooms.reduce((s, r) => s + r.width * r.height, 0))
  const totalWindows = plan.rooms.reduce((s, r) => s + windowCount(r), 0)
  const totalDoors   = plan.rooms.reduce((s, r) => s + doorCount(r), 0)

  return (
    <div className="spec-wrap">
      <div className="spec-header">
        <h3>{plan.name} — Specification Schedule</h3>
        <div className="spec-meta">
          {plan.rooms.length} rooms · {totalSqft.toLocaleString()} sq ft ·{' '}
          {Math.round(plan.totalWidth)}' × {Math.round(plan.totalHeight)}' footprint ·{' '}
          {ceilH}ft ceilings
        </div>
      </div>

      <div className="spec-table-wrap">
        <table className="spec-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Room</th>
              <th>Type</th>
              <th>Width</th>
              <th>Depth</th>
              <th>Area</th>
              <th>Ceiling</th>
              <th>Wins</th>
              <th>Doors</th>
              <th>Floor</th>
              <th>Walls</th>
              <th>Ceiling Finish</th>
            </tr>
          </thead>
          <tbody>
            {plan.rooms.map((room, i) => (
              <tr key={room.id}>
                <td className="spec-num">{i + 1}</td>
                <td className="spec-name">{room.name}</td>
                <td className="spec-type">{room.type.replace(/_/g, ' ')}</td>
                <td>{room.width.toFixed(0)}'</td>
                <td>{room.height.toFixed(0)}'</td>
                <td className="spec-area">{Math.round(room.width * room.height)} sf</td>
                <td>{ceilH}'</td>
                <td>{windowCount(room)}</td>
                <td>{doorCount(room)}</td>
                <td className="spec-finish">{getFinish(FLOOR_FINISH, room.type)}</td>
                <td className="spec-finish">{getFinish(WALL_FINISH, room.type)}</td>
                <td className="spec-finish">{getFinish(CEILING_FINISH, room.type)}</td>
              </tr>
            ))}
          </tbody>
          <tfoot>
            <tr className="spec-total">
              <td colSpan={5}><strong>Totals</strong></td>
              <td><strong>{totalSqft.toLocaleString()} sf</strong></td>
              <td>—</td>
              <td><strong>{totalWindows}</strong></td>
              <td><strong>{totalDoors}</strong></td>
              <td colSpan={3} />
            </tr>
          </tfoot>
        </table>
      </div>

      <div className="spec-notes">
        <strong>General Notes:</strong>
        <ul>
          <li>All dimensions are nominal, verify on site before construction.</li>
          <li>Interior walls: 3.5" wood stud with ½" drywall each side (4.5" total).</li>
          <li>Exterior walls: 5.5" 2×6 stud with insulation + ½" sheathing + siding.</li>
          <li>Door heights: 6'8" standard, 8'0" for main entry and primary suite.</li>
          <li>Window sill height: 36" (bathrooms 48"), head height 7'.</li>
          <li>All bathroom GFCI outlets within 6ft of water source per NEC 210.8.</li>
        </ul>
      </div>
    </div>
  )
}
