/**
 * ArchPlan.tsx — SVG-based architectural floor plan renderer.
 * Produces drawings that resemble professional architectural documents:
 * walls, door swing symbols, window symbols, room fixtures, dimension
 * lines, north arrow, scale bar, and title block.
 */

import React, { useMemo } from 'react'
import { FloorPlan, Room } from '../types/floorplan'

// ─── Constants ───────────────────────────────────────────────────────────────

const MARGIN = 70          // px around plan for dimension lines
const WALL_W = 5           // wall stroke width px
const TITLE_H = 48         // title block height px
const BG = '#faf9f5'       // drafting paper cream

// ─── Room fill colours — very light architectural tints ───────────────────────

const ROOM_FILLS: Record<string, string> = {
  living_room:       '#f0f4fb',
  great_room:        '#eef3fa',
  kitchen:           '#f5f5ee',
  dining_room:       '#f2f5ee',
  master_bedroom:    '#f5eef5',
  bedroom:           '#f5eef5',
  bathroom:          '#eef5f5',
  ensuite_bathroom:  '#eef5f5',
  powder_bath:       '#eef5f5',
  hallway:           '#f8f8f4',
  foyer:             '#f4f4ee',
  entry_foyer:       '#f4f4ee',
  closet:            '#f8f4f0',
  laundry_room:      '#f4f0f8',
  garage:            '#f0f0ec',
  home_office:       '#eef2f8',
  patio:             '#eef5ee',
  deck:              '#eef5ee',
  mudroom:           '#f5f0ec',
  utility_room:      '#f0eeee',
}

function roomFill(type: string): string {
  return ROOM_FILLS[type.toLowerCase()] ?? '#f5f5f5'
}

// ─── Interfaces ──────────────────────────────────────────────────────────────

export interface ArchPlanProps {
  plan: FloorPlan
  selectedId: string | null
  onSelect: (id: string | null) => void
  containerWidth: number
  containerHeight: number
}

interface RoomPx {
  room: Room
  px: number   // left edge
  py: number   // top edge
  pw: number   // width in px
  ph: number   // height in px
}

// ─── Coordinate helpers ───────────────────────────────────────────────────────

function computeLayout(plan: FloorPlan, cw: number, ch: number) {
  const availW = cw - MARGIN * 2
  const availH = ch - MARGIN * 2 - TITLE_H
  const S = Math.min(availW / plan.totalWidth, availH / plan.totalHeight)
  const ox = (cw - plan.totalWidth  * S) / 2
  const oy = (ch - TITLE_H - plan.totalHeight * S) / 2
  return { S, ox, oy }
}

function toRoomPx(room: Room, ox: number, oy: number, S: number): RoomPx {
  return {
    room,
    px: ox + room.x * S,
    py: oy + room.y * S,
    pw: room.width  * S,
    ph: room.height * S,
  }
}

// ─── Adjacency detection ──────────────────────────────────────────────────────

const SNAP = 0.5  // feet tolerance for adjacency

interface SharedEdge {
  roomA: Room
  roomB: Room
  wall: 'north' | 'south' | 'east' | 'west'  // wall of roomA that is shared
  overlapStart: number
  overlapEnd: number
}

function detectAdjacencies(rooms: Room[]): SharedEdge[] {
  const edges: SharedEdge[] = []
  for (let i = 0; i < rooms.length; i++) {
    for (let j = i + 1; j < rooms.length; j++) {
      const a = rooms[i], b = rooms[j]
      // South wall of a == North wall of b
      if (Math.abs((a.y + a.height) - b.y) < SNAP) {
        const oStart = Math.max(a.x, b.x)
        const oEnd   = Math.min(a.x + a.width, b.x + b.width)
        if (oEnd - oStart > 0.5) {
          edges.push({ roomA: a, roomB: b, wall: 'south', overlapStart: oStart, overlapEnd: oEnd })
          edges.push({ roomA: b, roomB: a, wall: 'north', overlapStart: oStart, overlapEnd: oEnd })
        }
      }
      // East wall of a == West wall of b
      if (Math.abs((a.x + a.width) - b.x) < SNAP) {
        const oStart = Math.max(a.y, b.y)
        const oEnd   = Math.min(a.y + a.height, b.y + b.height)
        if (oEnd - oStart > 0.5) {
          edges.push({ roomA: a, roomB: b, wall: 'east',  overlapStart: oStart, overlapEnd: oEnd })
          edges.push({ roomA: b, roomB: a, wall: 'west',  overlapStart: oStart, overlapEnd: oEnd })
        }
      }
    }
  }
  return edges
}

// ─── Door placement ──────────────────────────────────────────────────────────

interface DoorSpec {
  roomId: string
  wall: 'north' | 'south' | 'east' | 'west'
  offsetFrac: number      // fraction along the wall where hinge sits (0..1)
  widthFt: number
  hingeRight: boolean     // hinge is at the right/bottom end of the gap
  sliding: boolean
}

const HALLWAY_TYPES = new Set([
  'hallway', 'foyer', 'entry_foyer', 'corridor', 'landing',
])

function chooseDoorWidth(type: string): number {
  const t = type.toLowerCase()
  if (t === 'garage')            return 9   // single car door width
  if (t === 'closet')            return 3
  if (t.includes('bathroom') || t === 'powder_bath') return 2.5
  if (t === 'laundry_room')      return 2.8
  if (t === 'living_room' || t === 'great_room') return 6
  if (t.includes('bedroom'))     return 2.8
  return 3
}

function planDoors(rooms: Room[], plan: FloorPlan, adjacencies: SharedEdge[]): DoorSpec[] {
  const specs: DoorSpec[] = []
  const doored = new Set<string>()   // roomId already has a door

  for (const room of rooms) {
    const t = room.type.toLowerCase()
    const key = room.id

    // Sliding door for closets
    if (t === 'closet') {
      // Place on whichever wall is longest & shares edge with another room
      const adj = adjacencies.filter(e => e.roomA.id === room.id)
      if (adj.length > 0) {
        const e = adj[0]
        const isHoriz = e.wall === 'north' || e.wall === 'south'
        const overlapFt = e.overlapEnd - e.overlapStart
        const doorWFt = Math.min(room.width > room.height ? room.width * 0.8 : room.height * 0.8, overlapFt * 0.7)
        const wallLen = isHoriz ? room.width : room.height
        const center = ((e.overlapStart + e.overlapEnd) / 2) - (isHoriz ? room.x : room.y)
        const offset = Math.max(0.1, Math.min(0.9, center / wallLen))
        specs.push({ roomId: key, wall: e.wall, offsetFrac: offset, widthFt: doorWFt, hingeRight: false, sliding: true })
        doored.add(key)
      }
      continue
    }

    // Garage door — wide, on south exterior wall
    if (t === 'garage') {
      const isExteriorSouth = Math.abs(room.y + room.height - plan.totalHeight) < SNAP
      const isExteriorNorth = room.y < SNAP
      const wall = isExteriorSouth ? 'south' : isExteriorNorth ? 'north' : 'south'
      specs.push({ roomId: key, wall, offsetFrac: 0.5, widthFt: 9, hingeRight: false, sliding: true })
      doored.add(key)
      // Also add a side door to adjacent hallway if present
      const adjHall = adjacencies.find(
        e => e.roomA.id === room.id && HALLWAY_TYPES.has(e.roomB.type.toLowerCase()) && (e.wall === 'east' || e.wall === 'west' || e.wall === 'north')
      )
      if (adjHall) {
        const isHoriz = adjHall.wall === 'north' || adjHall.wall === 'south'
        const wallLen = isHoriz ? room.width : room.height
        const overlapMid = (adjHall.overlapStart + adjHall.overlapEnd) / 2
        const pos = isHoriz ? overlapMid - room.x : overlapMid - room.y
        specs.push({ roomId: key, wall: adjHall.wall, offsetFrac: Math.max(0.1, Math.min(0.9, pos / wallLen)), widthFt: 3, hingeRight: false, sliding: false })
      }
      continue
    }

    // Entry foyer / foyer — front door on boundary wall
    if (t === 'foyer' || t === 'entry_foyer') {
      const onNorth = room.y < SNAP
      const onWest  = room.x < SNAP
      const wall = onNorth ? 'north' : onWest ? 'west' : 'south'
      specs.push({ roomId: key, wall, offsetFrac: 0.5, widthFt: 3.5, hingeRight: false, sliding: false })
      doored.add(key)
      continue
    }

    // Other rooms — try to put door on wall shared with hallway / foyer
    const adjHall = adjacencies.find(
      e => e.roomA.id === room.id && HALLWAY_TYPES.has(e.roomB.type.toLowerCase())
    )
    const adjAny = adjacencies.find(e => e.roomA.id === room.id)
    const edge = adjHall ?? adjAny
    if (!edge) continue

    const isHoriz = edge.wall === 'north' || edge.wall === 'south'
    const wallLen = isHoriz ? room.width : room.height
    const overlapMid = (edge.overlapStart + edge.overlapEnd) / 2
    const pos = isHoriz ? overlapMid - room.x : overlapMid - room.y
    const dw = chooseDoorWidth(t)

    // Slide position so door fits within overlap
    const overlapLen = edge.overlapEnd - edge.overlapStart
    const safeFrac = wallLen > 0 ? Math.max(dw / 2 / wallLen + 0.05, Math.min(1 - dw / 2 / wallLen - 0.05, pos / wallLen)) : 0.5

    // Only add if overlap is big enough for the door
    if (overlapLen >= dw + 0.5) {
      specs.push({ roomId: key, wall: edge.wall, offsetFrac: safeFrac, widthFt: dw, hingeRight: false, sliding: false })
      doored.add(key)
    }
  }

  return specs
}

// ─── Window placement ────────────────────────────────────────────────────────

interface WindowSpec {
  roomId: string
  wall: 'north' | 'south' | 'east' | 'west'
  offsetFrac: number
  widthFt: number
}

const NO_WINDOW_TYPES = new Set([
  'hallway', 'foyer', 'entry_foyer', 'closet', 'laundry_room',
  'garage', 'patio', 'deck', 'mudroom', 'corridor', 'landing',
])

function windowWidthFt(type: string): number {
  const t = type.toLowerCase()
  if (t.includes('bathroom') || t === 'powder_bath') return 2
  if (t === 'kitchen' || t === 'dining_room')         return 3
  if (t.includes('bedroom'))                          return 3
  if (t === 'living_room' || t === 'great_room')      return 5
  if (t === 'home_office')                             return 3
  return 3
}

function planWindows(rooms: Room[], plan: FloorPlan, adjacencies: SharedEdge[]): WindowSpec[] {
  const specs: WindowSpec[] = []
  const adjacentWalls = new Set<string>()  // "roomId-wall" combos that are internal
  for (const e of adjacencies) adjacentWalls.add(`${e.roomA.id}-${e.wall}`)

  for (const room of rooms) {
    if (NO_WINDOW_TYPES.has(room.type.toLowerCase())) continue
    const ww = windowWidthFt(room.type)
    const extWalls: Array<'north' | 'south' | 'east' | 'west'> = []

    if (room.y < SNAP)                                extWalls.push('north')
    if (room.y + room.height > plan.totalHeight - SNAP) extWalls.push('south')
    if (room.x < SNAP)                                extWalls.push('west')
    if (room.x + room.width  > plan.totalWidth  - SNAP) extWalls.push('east')

    for (const wall of extWalls) {
      if (adjacentWalls.has(`${room.id}-${wall}`)) continue
      const wallLen = (wall === 'north' || wall === 'south') ? room.width : room.height
      if (wallLen < ww + 2) continue
      const count = (room.type === 'living_room' || room.type === 'great_room') && wallLen > ww * 2.5 ? 2 : 1
      if (count === 1) {
        specs.push({ roomId: room.id, wall, offsetFrac: 0.5, widthFt: ww })
      } else {
        specs.push({ roomId: room.id, wall, offsetFrac: 0.3, widthFt: ww })
        specs.push({ roomId: room.id, wall, offsetFrac: 0.7, widthFt: ww })
      }
    }
  }
  return specs
}

// ─── SVG door symbol ──────────────────────────────────────────────────────────

function DoorSymbol({
  rp, spec, S,
}: { rp: RoomPx; spec: DoorSpec; S: number }): React.ReactElement {
  const { px, py, pw, ph } = rp
  const dPx = spec.widthFt * S

  if (spec.sliding) {
    // Sliding door: two parallel lines with a gap marker
    if (spec.wall === 'south' || spec.wall === 'north') {
      const gCenter = px + spec.offsetFrac * pw
      const gL = gCenter - dPx / 2
      const gR = gCenter + dPx / 2
      const wallY = spec.wall === 'south' ? py + ph : py
      const inset = spec.wall === 'south' ? -S * 0.25 : S * 0.25
      return (
        <g>
          <rect x={gL} y={wallY - 3} width={dPx} height={6} fill={BG} />
          <line x1={gL} y1={wallY - 2} x2={gR} y2={wallY - 2} stroke="#1a1a1a" strokeWidth={1.5} />
          <line x1={gL} y1={wallY + 2} x2={gR} y2={wallY + 2} stroke="#1a1a1a" strokeWidth={1.5} />
          <line x1={gL + dPx * 0.5} y1={wallY - 2} x2={gL + dPx * 0.5} y2={wallY + inset} stroke="#1a1a1a" strokeWidth={1} />
        </g>
      )
    } else {
      const gCenter = py + spec.offsetFrac * ph
      const gT = gCenter - dPx / 2
      const gB = gCenter + dPx / 2
      const wallX = spec.wall === 'east' ? px + pw : px
      const inset = spec.wall === 'east' ? -S * 0.25 : S * 0.25
      return (
        <g>
          <rect x={wallX - 3} y={gT} width={6} height={dPx} fill={BG} />
          <line x1={wallX - 2} y1={gT} x2={wallX - 2} y2={gB} stroke="#1a1a1a" strokeWidth={1.5} />
          <line x1={wallX + 2} y1={gT} x2={wallX + 2} y2={gB} stroke="#1a1a1a" strokeWidth={1.5} />
          <line x1={wallX - 2} y1={gT + dPx * 0.5} x2={wallX + inset} y2={gT + dPx * 0.5} stroke="#1a1a1a" strokeWidth={1} />
        </g>
      )
    }
  }

  // Swing door
  const hr = spec.hingeRight

  if (spec.wall === 'south') {
    const gCenter = px + spec.offsetFrac * pw
    const gL = gCenter - dPx / 2
    const gR = gCenter + dPx / 2
    const wallY = py + ph
    if (!hr) {
      return (
        <g>
          <rect x={gL} y={wallY - 3} width={dPx} height={6} fill={BG} />
          <line x1={gL} y1={wallY} x2={gL} y2={wallY - dPx} stroke="#1a1a1a" strokeWidth={1.5} />
          <path d={`M ${gL} ${wallY - dPx} A ${dPx} ${dPx} 0 0 1 ${gR} ${wallY}`}
            fill="none" stroke="#666" strokeWidth={1} strokeDasharray="4,3" />
        </g>
      )
    } else {
      return (
        <g>
          <rect x={gL} y={wallY - 3} width={dPx} height={6} fill={BG} />
          <line x1={gR} y1={wallY} x2={gR} y2={wallY - dPx} stroke="#1a1a1a" strokeWidth={1.5} />
          <path d={`M ${gR} ${wallY - dPx} A ${dPx} ${dPx} 0 0 0 ${gL} ${wallY}`}
            fill="none" stroke="#666" strokeWidth={1} strokeDasharray="4,3" />
        </g>
      )
    }
  }

  if (spec.wall === 'north') {
    const gCenter = px + spec.offsetFrac * pw
    const gL = gCenter - dPx / 2
    const gR = gCenter + dPx / 2
    const wallY = py
    if (!hr) {
      return (
        <g>
          <rect x={gL} y={wallY - 3} width={dPx} height={6} fill={BG} />
          <line x1={gL} y1={wallY} x2={gL} y2={wallY + dPx} stroke="#1a1a1a" strokeWidth={1.5} />
          <path d={`M ${gL} ${wallY + dPx} A ${dPx} ${dPx} 0 0 0 ${gR} ${wallY}`}
            fill="none" stroke="#666" strokeWidth={1} strokeDasharray="4,3" />
        </g>
      )
    } else {
      return (
        <g>
          <rect x={gL} y={wallY - 3} width={dPx} height={6} fill={BG} />
          <line x1={gR} y1={wallY} x2={gR} y2={wallY + dPx} stroke="#1a1a1a" strokeWidth={1.5} />
          <path d={`M ${gR} ${wallY + dPx} A ${dPx} ${dPx} 0 0 1 ${gL} ${wallY}`}
            fill="none" stroke="#666" strokeWidth={1} strokeDasharray="4,3" />
        </g>
      )
    }
  }

  if (spec.wall === 'east') {
    const gCenter = py + spec.offsetFrac * ph
    const gT = gCenter - dPx / 2
    const gB = gCenter + dPx / 2
    const wallX = px + pw
    return (
      <g>
        <rect x={wallX - 3} y={gT} width={6} height={dPx} fill={BG} />
        <line x1={wallX} y1={gT} x2={wallX - dPx} y2={gT} stroke="#1a1a1a" strokeWidth={1.5} />
        <path d={`M ${wallX - dPx} ${gT} A ${dPx} ${dPx} 0 0 1 ${wallX} ${gB}`}
          fill="none" stroke="#666" strokeWidth={1} strokeDasharray="4,3" />
      </g>
    )
  }

  // west
  const gCenter = py + spec.offsetFrac * ph
  const gT = gCenter - dPx / 2
  const gB = gCenter + dPx / 2
  const wallX = px
  return (
    <g>
      <rect x={wallX - 3} y={gT} width={6} height={dPx} fill={BG} />
      <line x1={wallX} y1={gT} x2={wallX + dPx} y2={gT} stroke="#1a1a1a" strokeWidth={1.5} />
      <path d={`M ${wallX + dPx} ${gT} A ${dPx} ${dPx} 0 0 0 ${wallX} ${gB}`}
        fill="none" stroke="#666" strokeWidth={1} strokeDasharray="4,3" />
    </g>
  )
}

// ─── SVG window symbol ────────────────────────────────────────────────────────

function WindowSymbol({
  rp, spec, S,
}: { rp: RoomPx; spec: WindowSpec; S: number }): React.ReactElement {
  const { px, py, pw, ph } = rp
  const wPx = spec.widthFt * S

  if (spec.wall === 'south') {
    const wCenter = px + spec.offsetFrac * pw
    const wL = wCenter - wPx / 2
    const wallY = py + ph
    return (
      <g>
        <rect x={wL} y={wallY - 3} width={wPx} height={6} fill={BG} />
        <line x1={wL} y1={wallY - 2} x2={wL + wPx} y2={wallY - 2} stroke="#1a1a1a" strokeWidth={1.5} />
        <line x1={wL} y1={wallY + 2} x2={wL + wPx} y2={wallY + 2} stroke="#1a1a1a" strokeWidth={1.5} />
        <line x1={wL} y1={wallY}     x2={wL + wPx} y2={wallY}     stroke="#1a1a1a" strokeWidth={0.8} />
      </g>
    )
  }
  if (spec.wall === 'north') {
    const wCenter = px + spec.offsetFrac * pw
    const wL = wCenter - wPx / 2
    const wallY = py
    return (
      <g>
        <rect x={wL} y={wallY - 3} width={wPx} height={6} fill={BG} />
        <line x1={wL} y1={wallY - 2} x2={wL + wPx} y2={wallY - 2} stroke="#1a1a1a" strokeWidth={1.5} />
        <line x1={wL} y1={wallY + 2} x2={wL + wPx} y2={wallY + 2} stroke="#1a1a1a" strokeWidth={1.5} />
        <line x1={wL} y1={wallY}     x2={wL + wPx} y2={wallY}     stroke="#1a1a1a" strokeWidth={0.8} />
      </g>
    )
  }
  if (spec.wall === 'east') {
    const wCenter = py + spec.offsetFrac * ph
    const wT = wCenter - wPx / 2
    const wallX = px + pw
    return (
      <g>
        <rect x={wallX - 3} y={wT} width={6} height={wPx} fill={BG} />
        <line x1={wallX - 2} y1={wT} x2={wallX - 2} y2={wT + wPx} stroke="#1a1a1a" strokeWidth={1.5} />
        <line x1={wallX + 2} y1={wT} x2={wallX + 2} y2={wT + wPx} stroke="#1a1a1a" strokeWidth={1.5} />
        <line x1={wallX}     y1={wT} x2={wallX}     y2={wT + wPx} stroke="#1a1a1a" strokeWidth={0.8} />
      </g>
    )
  }
  // west
  const wCenter = py + spec.offsetFrac * ph
  const wT = wCenter - wPx / 2
  const wallX = px
  return (
    <g>
      <rect x={wallX - 3} y={wT} width={6} height={wPx} fill={BG} />
      <line x1={wallX - 2} y1={wT} x2={wallX - 2} y2={wT + wPx} stroke="#1a1a1a" strokeWidth={1.5} />
      <line x1={wallX + 2} y1={wT} x2={wallX + 2} y2={wT + wPx} stroke="#1a1a1a" strokeWidth={1.5} />
      <line x1={wallX}     y1={wT} x2={wallX}     y2={wT + wPx} stroke="#1a1a1a" strokeWidth={0.8} />
    </g>
  )
}

// ─── Room fixtures ────────────────────────────────────────────────────────────

const FX = '#555'
const FX_W = 1.2
const FX_PROPS = { stroke: FX, strokeWidth: FX_W, fill: 'none' }

function BathroomFixtures({ rp, S }: { rp: RoomPx; S: number }): React.ReactElement | null {
  const { px, py, pw, ph } = rp
  const t = rp.room.type.toLowerCase()
  // Minimum room size check
  if (pw < S * 3 || ph < S * 3) return null

  const margin = S * 0.3
  const elements: React.ReactElement[] = []

  // Toilet — upper-right corner: tank + bowl
  const tkW = Math.min(S * 1.4, pw * 0.45)
  const tkH = S * 0.7
  const bwW = tkW
  const bwH = Math.min(S * 1.7, ph * 0.45)
  const tkX = px + pw - margin - tkW
  const tkY = py + margin
  elements.push(
    <rect key="tk" x={tkX} y={tkY} width={tkW} height={tkH} rx={2} {...FX_PROPS} />,
    <ellipse key="bw" cx={tkX + bwW / 2} cy={tkY + tkH + bwH / 2} rx={bwW / 2} ry={bwH / 2} {...FX_PROPS} />
  )

  // Sink — upper-left corner
  const skW = Math.min(S * 1.8, pw * 0.45)
  const skH = Math.min(S * 1.6, ph * 0.35)
  const skX = px + margin
  const skY = py + margin
  const skRx = skW * 0.2
  elements.push(
    <rect key="sk" x={skX} y={skY} width={skW} height={skH} rx={skRx} {...FX_PROPS} />,
    <circle key="sk-drain" cx={skX + skW / 2} cy={skY + skH / 2} r={skW * 0.08} stroke={FX} strokeWidth={0.8} fill="none" />
  )

  // Bathtub — for full bathroom / ensuite, if tall enough
  if ((t === 'bathroom' || t === 'ensuite_bathroom') && ph > S * 7) {
    const tbW = Math.min(S * 2.5, pw * 0.75)
    const tbH = Math.min(S * 4.5, ph * 0.55)
    const tbX = px + (pw - tbW) / 2
    const tbY = py + ph - margin - tbH
    const inset = S * 0.2
    elements.push(
      <rect key="tb-outer" x={tbX} y={tbY} width={tbW} height={tbH} rx={S * 0.3} {...FX_PROPS} />,
      <rect key="tb-inner" x={tbX + inset} y={tbY + inset} width={tbW - inset * 2} height={tbH - inset * 2}
        rx={S * 0.25} stroke={FX} strokeWidth={0.7} fill="none" />,
      <circle key="tb-drain" cx={tbX + tbW / 2} cy={tbY + tbH - inset * 1.5} r={S * 0.15}
        stroke={FX} strokeWidth={0.8} fill="none" />
    )
  }

  return <g>{elements}</g>
}

function KitchenFixtures({ rp, S }: { rp: RoomPx; S: number }): React.ReactElement | null {
  const { px, py, pw, ph } = rp
  if (pw < S * 4 || ph < S * 4) return null

  const ctD = Math.min(S * 2, pw * 0.3, ph * 0.3)  // counter depth
  const elements: React.ReactElement[] = []

  // Counter along north wall
  elements.push(
    <rect key="ct-n" x={px} y={py} width={pw} height={ctD} stroke={FX} strokeWidth={FX_W} fill="#f0ede5" />
  )
  // Counter along west wall
  elements.push(
    <rect key="ct-w" x={px} y={py} width={ctD} height={ph * 0.6} stroke={FX} strokeWidth={FX_W} fill="#f0ede5" />
  )

  // Stove burners on north counter
  const burnerR = ctD * 0.18
  const burnerY = py + ctD / 2
  const stoveX = px + pw * 0.55
  for (let i = 0; i < 4; i++) {
    const bx = stoveX + (i % 2) * burnerR * 2.8
    const by = burnerY + Math.floor(i / 2) * burnerR * 2.8 - burnerR * 1.4
    elements.push(
      <circle key={`b${i}`} cx={bx} cy={by} r={burnerR} stroke={FX} strokeWidth={0.8} fill="none" />
    )
  }

  // Sink in north counter
  const skW = Math.min(S * 1.5, pw * 0.2)
  const skH = ctD * 0.7
  const skX = px + pw * 0.3 - skW / 2
  const skY = py + ctD * 0.15
  elements.push(
    <rect key="sink" x={skX} y={skY} width={skW} height={skH} rx={3} stroke={FX} strokeWidth={0.8} fill="none" />,
    <line key="sink-d" x1={skX + skW / 2} y1={skY} x2={skX + skW / 2} y2={skY + skH} stroke={FX} strokeWidth={0.5} />
  )

  // Fridge on west counter
  const fridgeW = ctD * 0.9
  const fridgeH = Math.min(S * 2.5, ph * 0.25)
  const fridgeX = px + ctD * 0.05
  const fridgeY = py + ph * 0.35 - fridgeH / 2
  elements.push(
    <rect key="fridge" x={fridgeX} y={fridgeY} width={fridgeW} height={fridgeH} stroke={FX} strokeWidth={FX_W} fill="none" />,
    <line key="fridge-h" x1={fridgeX} y1={fridgeY + fridgeH / 2} x2={fridgeX + fridgeW} y2={fridgeY + fridgeH / 2}
      stroke={FX} strokeWidth={0.6} />
  )

  return <g>{elements}</g>
}

function DiningFixtures({ rp, S }: { rp: RoomPx; S: number }): React.ReactElement | null {
  const { px, py, pw, ph } = rp
  if (pw < S * 6 || ph < S * 6) return null

  const tblW = Math.min(pw * 0.55, S * 6)
  const tblH = Math.min(ph * 0.45, S * 4)
  const tblX = px + (pw - tblW) / 2
  const tblY = py + (ph - tblH) / 2
  const chairW = S * 1.5
  const chairH = S * 1.0
  const elements: React.ReactElement[] = []

  // Table
  elements.push(
    <rect key="tbl" x={tblX} y={tblY} width={tblW} height={tblH} rx={S * 0.3} stroke={FX} strokeWidth={FX_W} fill="none" />
  )

  // Chairs: top + bottom rows
  const cols = Math.max(2, Math.round(tblW / (chairW * 1.4)))
  for (let i = 0; i < cols; i++) {
    const cx = tblX + (i + 0.5) * (tblW / cols) - chairW / 2
    elements.push(
      <rect key={`ct${i}`} x={cx} y={tblY - chairH - S * 0.15} width={chairW} height={chairH}
        rx={2} stroke={FX} strokeWidth={0.8} fill="none" />,
      <rect key={`cb${i}`} x={cx} y={tblY + tblH + S * 0.15} width={chairW} height={chairH}
        rx={2} stroke={FX} strokeWidth={0.8} fill="none" />
    )
  }
  // Side chairs
  const rows = Math.max(1, Math.round(tblH / (chairH * 1.8)))
  for (let i = 0; i < rows; i++) {
    const cy = tblY + (i + 0.5) * (tblH / rows) - chairH / 2
    elements.push(
      <rect key={`cl${i}`} x={tblX - chairH - S * 0.15} y={cy} width={chairH} height={chairW}
        rx={2} stroke={FX} strokeWidth={0.8} fill="none" />,
      <rect key={`cr${i}`} x={tblX + tblW + S * 0.15} y={cy} width={chairH} height={chairW}
        rx={2} stroke={FX} strokeWidth={0.8} fill="none" />
    )
  }

  return <g>{elements}</g>
}

function BedroomFixtures({ rp, S, primary }: { rp: RoomPx; S: number; primary: boolean }): React.ReactElement | null {
  const { px, py, pw, ph } = rp
  if (pw < S * 4 || ph < S * 4) return null

  const bedW = primary ? Math.min(S * 6, pw * 0.75) : Math.min(S * 5, pw * 0.75)
  const bedH = primary ? Math.min(S * 6.5, ph * 0.65) : Math.min(S * 6, ph * 0.65)
  const bedX = px + (pw - bedW) / 2
  const bedY = py + S * 0.5
  const headH = S * 0.7
  const pillowW = bedW * 0.35
  const pillowH = headH * 0.7
  const elements: React.ReactElement[] = []

  // Headboard
  elements.push(
    <rect key="head" x={bedX} y={bedY} width={bedW} height={headH} rx={3} stroke={FX} strokeWidth={FX_W} fill="#e8e5e0" />
  )
  // Mattress
  elements.push(
    <rect key="mattress" x={bedX} y={bedY + headH} width={bedW} height={bedH - headH}
      rx={4} stroke={FX} strokeWidth={FX_W} fill="none" />
  )
  // Pillows
  elements.push(
    <rect key="p1" x={bedX + bedW * 0.1} y={bedY + headH + S * 0.2} width={pillowW} height={pillowH}
      rx={pillowH / 3} stroke={FX} strokeWidth={0.8} fill="none" />,
    <rect key="p2" x={bedX + bedW * 0.9 - pillowW} y={bedY + headH + S * 0.2} width={pillowW} height={pillowH}
      rx={pillowH / 3} stroke={FX} strokeWidth={0.8} fill="none" />
  )

  return <g>{elements}</g>
}

function LivingRoomFixtures({ rp, S }: { rp: RoomPx; S: number }): React.ReactElement | null {
  const { px, py, pw, ph } = rp
  if (pw < S * 5 || ph < S * 5) return null

  const sofaD = Math.min(S * 2.8, ph * 0.3)
  const sofaW = Math.min(S * 7, pw * 0.8)
  const sofaX = px + (pw - sofaW) / 2
  const sofaY = py + ph - sofaD - S * 0.5
  const armW  = sofaD * 0.4
  const back   = sofaD * 0.3
  const elements: React.ReactElement[] = []

  // Sofa body
  elements.push(
    <rect key="sofa" x={sofaX} y={sofaY} width={sofaW} height={sofaD} rx={S * 0.2} stroke={FX} strokeWidth={FX_W} fill="none" />
  )
  // Sofa back
  elements.push(
    <rect key="sofa-back" x={sofaX + armW} y={sofaY} width={sofaW - armW * 2} height={back}
      stroke={FX} strokeWidth={0.8} fill="none" />
  )
  // Arms
  elements.push(
    <rect key="arm-l" x={sofaX} y={sofaY} width={armW} height={sofaD} stroke={FX} strokeWidth={0.8} fill="none" />,
    <rect key="arm-r" x={sofaX + sofaW - armW} y={sofaY} width={armW} height={sofaD} stroke={FX} strokeWidth={0.8} fill="none" />
  )
  // Cushions
  const nCush = Math.max(2, Math.round(sofaW / (S * 2.2)))
  const cushW = (sofaW - armW * 2) / nCush
  for (let i = 0; i < nCush; i++) {
    elements.push(
      <rect key={`cush${i}`}
        x={sofaX + armW + i * cushW + 1} y={sofaY + back + 1}
        width={cushW - 2} height={sofaD - back - 2}
        rx={3} stroke={FX} strokeWidth={0.7} fill="none" />
    )
  }

  // Coffee table
  const ctW = sofaW * 0.5
  const ctH = sofaD * 0.5
  const ctX = px + (pw - ctW) / 2
  const ctY = sofaY - ctH - S * 0.5
  elements.push(
    <rect key="ct" x={ctX} y={ctY} width={ctW} height={ctH} rx={S * 0.1} stroke={FX} strokeWidth={FX_W} fill="none" />
  )

  return <g>{elements}</g>
}

function RoomFixtures({ rp, S }: { rp: RoomPx; S: number }): React.ReactElement | null {
  const t = rp.room.type.toLowerCase()
  if (t.includes('bathroom') || t === 'powder_bath') return <BathroomFixtures rp={rp} S={S} />
  if (t === 'kitchen')        return <KitchenFixtures   rp={rp} S={S} />
  if (t === 'dining_room')    return <DiningFixtures     rp={rp} S={S} />
  if (t === 'master_bedroom') return <BedroomFixtures    rp={rp} S={S} primary={true} />
  if (t === 'bedroom')        return <BedroomFixtures    rp={rp} S={S} primary={false} />
  if (t === 'living_room' || t === 'great_room') return <LivingRoomFixtures rp={rp} S={S} />
  return null
}

// ─── Room label ───────────────────────────────────────────────────────────────

function RoomLabel({ rp, selected }: { rp: RoomPx; selected: boolean }): React.ReactElement {
  const { px, py, pw, ph, room } = rp
  const name = room.name.toUpperCase()
  const dims = `${Math.round(room.width)}' × ${Math.round(room.height)}'`
  const area = `${Math.round(room.width * room.height)} SF`

  const fontSize = Math.max(8, Math.min(13, pw / (name.length * 0.65)))
  const dimSize  = Math.max(7, Math.min(10, fontSize * 0.78))

  const cx = px + pw / 2
  const cy = py + ph / 2

  const showDims = pw > 55 && ph > 38
  const showArea = ph > 55

  const labelBlockH = fontSize + (showDims ? dimSize + 2 : 0) + (showArea ? dimSize : 0)
  const startY = cy - labelBlockH / 2

  return (
    <g>
      <text
        x={cx} y={startY + fontSize}
        textAnchor="middle"
        fontFamily="system-ui, Arial, sans-serif"
        fontSize={fontSize}
        fontWeight="600"
        fill={selected ? '#b84000' : '#1a1a1a'}
        style={{ userSelect: 'none', pointerEvents: 'none' }}
      >
        {name}
      </text>
      {showDims && (
        <text
          x={cx} y={startY + fontSize + dimSize + 3}
          textAnchor="middle"
          fontFamily="system-ui, Arial, sans-serif"
          fontSize={dimSize}
          fill={selected ? '#b84000' : '#555'}
          style={{ userSelect: 'none', pointerEvents: 'none' }}
        >
          {dims}
        </text>
      )}
      {showArea && (
        <text
          x={cx} y={startY + fontSize + (dimSize + 3) * 2}
          textAnchor="middle"
          fontFamily="system-ui, Arial, sans-serif"
          fontSize={dimSize}
          fill={selected ? '#cc5500' : '#888'}
          style={{ userSelect: 'none', pointerEvents: 'none' }}
        >
          {area}
        </text>
      )}
    </g>
  )
}

// ─── Dimension lines ─────────────────────────────────────────────────────────

function DimensionLines({
  plan, ox, oy, S,
}: { plan: FloorPlan; ox: number; oy: number; S: number }): React.ReactElement {
  const W = plan.totalWidth  * S
  const H = plan.totalHeight * S
  const TICK = 7
  const GAP = 28

  return (
    <g stroke="#555" strokeWidth={1} fill="none">
      {/* Top dimension */}
      <line x1={ox} y1={oy - GAP} x2={ox + W} y2={oy - GAP} />
      <line x1={ox}     y1={oy - GAP - TICK} x2={ox}     y2={oy - GAP + TICK} />
      <line x1={ox + W} y1={oy - GAP - TICK} x2={ox + W} y2={oy - GAP + TICK} />
      <text
        x={ox + W / 2} y={oy - GAP - 5}
        textAnchor="middle"
        fontFamily="system-ui, Arial, sans-serif"
        fontSize={11}
        fill="#555"
        style={{ userSelect: 'none' }}
      >
        {Math.round(plan.totalWidth)}&apos;-0&quot;
      </text>

      {/* Left dimension */}
      <line x1={ox - GAP} y1={oy} x2={ox - GAP} y2={oy + H} />
      <line x1={ox - GAP - TICK} y1={oy}     x2={ox - GAP + TICK} y2={oy} />
      <line x1={ox - GAP - TICK} y1={oy + H} x2={ox - GAP + TICK} y2={oy + H} />
      <text
        x={ox - GAP - 5} y={oy + H / 2}
        textAnchor="middle"
        fontFamily="system-ui, Arial, sans-serif"
        fontSize={11}
        fill="#555"
        transform={`rotate(-90, ${ox - GAP - 5}, ${oy + H / 2})`}
        style={{ userSelect: 'none' }}
      >
        {Math.round(plan.totalHeight)}&apos;-0&quot;
      </text>
    </g>
  )
}

// ─── North arrow ──────────────────────────────────────────────────────────────

function NorthArrow({ x, y }: { x: number; y: number }): React.ReactElement {
  const r = 18
  return (
    <g transform={`translate(${x}, ${y})`}>
      <circle cx={0} cy={0} r={r} stroke="#1a1a1a" strokeWidth={1} fill="white" />
      {/* Arrow shaft */}
      <line x1={0} y1={-r + 4} x2={0} y2={r - 4} stroke="#1a1a1a" strokeWidth={1.5} />
      {/* Arrowhead pointing north */}
      <path d="M -5 -8 L 0 -14 L 5 -8 Z" fill="#1a1a1a" />
      {/* N label */}
      <text x={0} y={8} textAnchor="middle" fontFamily="system-ui, Arial" fontSize={9} fontWeight="bold" fill="#1a1a1a">N</text>
    </g>
  )
}

// ─── Scale bar ────────────────────────────────────────────────────────────────

function ScaleBar({ x, y, S }: { x: number; y: number; S: number }): React.ReactElement {
  const totalFt = 10
  const barW = totalFt * S
  const midW  = barW / 2
  const h = 4
  return (
    <g transform={`translate(${x}, ${y})`}>
      {/* Two-tone bar */}
      <rect x={0}    y={0} width={midW} height={h} fill="#1a1a1a" />
      <rect x={midW} y={0} width={midW} height={h} fill="white" stroke="#1a1a1a" strokeWidth={0.8} />
      {/* Tick marks */}
      <line x1={0}    y1={0} x2={0}    y2={h + 3} stroke="#1a1a1a" strokeWidth={0.8} />
      <line x1={midW} y1={0} x2={midW} y2={h + 3} stroke="#1a1a1a" strokeWidth={0.8} />
      <line x1={barW} y1={0} x2={barW} y2={h + 3} stroke="#1a1a1a" strokeWidth={0.8} />
      {/* Labels */}
      <text x={0}    y={h + 12} textAnchor="middle" fontFamily="system-ui,Arial" fontSize={8} fill="#1a1a1a">0</text>
      <text x={midW} y={h + 12} textAnchor="middle" fontFamily="system-ui,Arial" fontSize={8} fill="#1a1a1a">5</text>
      <text x={barW} y={h + 12} textAnchor="middle" fontFamily="system-ui,Arial" fontSize={8} fill="#1a1a1a">10</text>
      <text x={barW / 2} y={h + 22} textAnchor="middle" fontFamily="system-ui,Arial" fontSize={7} fill="#888">FEET</text>
    </g>
  )
}

// ─── Title block ──────────────────────────────────────────────────────────────

function TitleBlock({
  plan, svgW, svgH,
}: { plan: FloorPlan; svgW: number; svgH: number }): React.ReactElement {
  const y = svgH - TITLE_H
  const totalSF = Math.round(plan.rooms.reduce((s, r) => s + r.width * r.height, 0))
  return (
    <g>
      <rect x={0} y={y} width={svgW} height={TITLE_H} fill="white" stroke="#1a1a1a" strokeWidth={0.5} />
      <line x1={svgW * 0.4} y1={y} x2={svgW * 0.4} y2={svgH} stroke="#1a1a1a" strokeWidth={0.5} />
      <line x1={svgW * 0.72} y1={y} x2={svgW * 0.72} y2={svgH} stroke="#1a1a1a" strokeWidth={0.5} />
      {/* Plan name */}
      <text x={svgW * 0.2} y={y + 18} textAnchor="middle"
        fontFamily="system-ui, Arial" fontSize={12} fontWeight="700" fill="#1a1a1a">{plan.name.toUpperCase()}</text>
      <text x={svgW * 0.2} y={y + 33} textAnchor="middle"
        fontFamily="system-ui, Arial" fontSize={9} fill="#555">ARCHITECTURAL FLOOR PLAN</text>
      {/* Center label */}
      <text x={svgW * 0.56} y={y + 18} textAnchor="middle"
        fontFamily="system-ui, Arial" fontSize={11} fontWeight="600" fill="#1a1a1a">FLOOR PLAN</text>
      <text x={svgW * 0.56} y={y + 33} textAnchor="middle"
        fontFamily="system-ui, Arial" fontSize={9} fill="#555">NOT FOR CONSTRUCTION</text>
      {/* Stats */}
      <text x={svgW * 0.86} y={y + 16} textAnchor="middle"
        fontFamily="system-ui, Arial" fontSize={9} fill="#555">TOTAL AREA</text>
      <text x={svgW * 0.86} y={y + 28} textAnchor="middle"
        fontFamily="system-ui, Arial" fontSize={12} fontWeight="700" fill="#1a1a1a">{totalSF.toLocaleString()} SF</text>
      <text x={svgW * 0.86} y={y + 40} textAnchor="middle"
        fontFamily="system-ui, Arial" fontSize={8} fill="#888">{Math.round(plan.totalWidth)}' × {Math.round(plan.totalHeight)}'</text>
    </g>
  )
}

// ─── Main component ───────────────────────────────────────────────────────────

export default function ArchPlan({
  plan,
  selectedId,
  onSelect,
  containerWidth,
  containerHeight,
}: ArchPlanProps): React.ReactElement {
  const svgW = containerWidth
  const svgH = containerHeight

  const { S, ox, oy } = useMemo(
    () => computeLayout(plan, svgW, svgH),
    [plan, svgW, svgH],
  )

  const roomPxs: RoomPx[] = useMemo(
    () => plan.rooms.map(r => toRoomPx(r, ox, oy, S)),
    [plan.rooms, ox, oy, S],
  )

  const adjacencies = useMemo(() => detectAdjacencies(plan.rooms), [plan.rooms])
  const doors       = useMemo(() => planDoors(plan.rooms, plan, adjacencies), [plan, adjacencies])
  const windows     = useMemo(() => planWindows(plan.rooms, plan, adjacencies), [plan, adjacencies])

  // Build lookup maps
  const roomPxMap = useMemo(() => {
    const m = new Map<string, RoomPx>()
    roomPxs.forEach(rp => m.set(rp.room.id, rp))
    return m
  }, [roomPxs])

  const planW = plan.totalWidth  * S
  const planH = plan.totalHeight * S

  return (
    <svg
      width={svgW}
      height={svgH}
      style={{ display: 'block', background: BG, cursor: 'default' }}
      onClick={e => {
        if ((e.target as SVGElement).tagName === 'svg') onSelect(null)
      }}
    >
      {/* Paper background */}
      <rect x={0} y={0} width={svgW} height={svgH} fill={BG} />

      {/* Subtle grid */}
      <g opacity={0.35}>
        {Array.from({ length: Math.ceil(plan.totalWidth / 5) + 1 }, (_, i) => i * 5).map(x => {
          const px = ox + x * S
          return (
            <line key={`gx${x}`}
              x1={px} y1={oy} x2={px} y2={oy + planH}
              stroke={x % 10 === 0 ? '#c8c4bc' : '#dedad5'}
              strokeWidth={x % 10 === 0 ? 0.8 : 0.4}
            />
          )
        })}
        {Array.from({ length: Math.ceil(plan.totalHeight / 5) + 1 }, (_, i) => i * 5).map(y => {
          const py = oy + y * S
          return (
            <line key={`gy${y}`}
              x1={ox} y1={py} x2={ox + planW} y2={py}
              stroke={y % 10 === 0 ? '#c8c4bc' : '#dedad5'}
              strokeWidth={y % 10 === 0 ? 0.8 : 0.4}
            />
          )
        })}
      </g>

      {/* ── Room fills ── */}
      <g>
        {roomPxs.map(rp => (
          <rect
            key={`fill-${rp.room.id}`}
            x={rp.px} y={rp.py}
            width={rp.pw} height={rp.ph}
            fill={roomFill(rp.room.type)}
            stroke="none"
            onClick={() => onSelect(rp.room.id === selectedId ? null : rp.room.id)}
            style={{ cursor: 'pointer' }}
          />
        ))}
      </g>

      {/* ── Room fixtures ── */}
      <g opacity={0.85}>
        {roomPxs.map(rp => (
          <RoomFixtures key={`fx-${rp.room.id}`} rp={rp} S={S} />
        ))}
      </g>

      {/* ── Window symbols (rendered below walls so walls overlap them) ── */}
      <g>
        {windows.map((ws, i) => {
          const rp = roomPxMap.get(ws.roomId)
          if (!rp) return null
          return <WindowSymbol key={`win-${i}`} rp={rp} spec={ws} S={S} />
        })}
      </g>

      {/* ── Walls (room outlines — drawn thick) ── */}
      <g>
        {roomPxs.map(rp => (
          <rect
            key={`wall-${rp.room.id}`}
            x={rp.px} y={rp.py}
            width={rp.pw} height={rp.ph}
            fill="none"
            stroke={rp.room.id === selectedId ? '#d45000' : '#1a1a1a'}
            strokeWidth={rp.room.id === selectedId ? WALL_W + 1 : WALL_W}
            style={{ pointerEvents: 'none' }}
          />
        ))}
      </g>

      {/* ── Outer plan boundary (extra-thick) ── */}
      <rect
        x={ox} y={oy}
        width={planW} height={planH}
        fill="none"
        stroke="#1a1a1a"
        strokeWidth={WALL_W + 2}
        style={{ pointerEvents: 'none' }}
      />

      {/* ── Door symbols (rendered on top of walls) ── */}
      <g>
        {doors.map((ds, i) => {
          const rp = roomPxMap.get(ds.roomId)
          if (!rp) return null
          return <DoorSymbol key={`door-${i}`} rp={rp} spec={ds} S={S} />
        })}
      </g>

      {/* ── Room labels ── */}
      <g>
        {roomPxs.map(rp => (
          <RoomLabel
            key={`lbl-${rp.room.id}`}
            rp={rp}
            selected={rp.room.id === selectedId}
          />
        ))}
      </g>

      {/* ── Dimension lines ── */}
      <DimensionLines plan={plan} ox={ox} oy={oy} S={S} />

      {/* ── North arrow (top-right of plan area) ── */}
      <NorthArrow x={ox + planW + 38} y={oy + 28} />

      {/* ── Scale bar (bottom-right of plan area) ── */}
      <ScaleBar x={ox + planW - 10 * S - 5} y={oy + planH + 18} S={S} />

      {/* ── Title block ── */}
      <TitleBlock plan={plan} svgW={svgW} svgH={svgH} />
    </svg>
  )
}
