import { useRef, useEffect, useMemo } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Sky, PerspectiveCamera, Text, Environment, Html } from '@react-three/drei'
import * as THREE from 'three'
import { FloorPlan } from '../types/floorplan'

const S = 0.09   // 1 ft = 0.09 THREE units
type Mode = 'exterior' | 'dollhouse' | 'walkthrough' | 'topview'
const WT = 0.022  // wall thickness

// ─── Style palette ────────────────────────────────────────────────────────────
const STYLE_PALETTE: Record<string, { wall: string; trim: string; roof: string; pitch: number }> = {
  modern:       { wall: '#F2F3F5', trim: '#C8CDD6', roof: '#D0D4DC', pitch: 2 },
  contemporary: { wall: '#EAEDF0', trim: '#B0B8C4', roof: '#C0C8D4', pitch: 2 },
  craftsman:    { wall: '#D4C4A8', trim: '#6B4E2E', roof: '#5C4030', pitch: 6 },
  farmhouse:    { wall: '#F5F5F2', trim: '#2C2C2C', roof: '#3A3A3A', pitch: 5 },
  traditional:  { wall: '#EDE7D9', trim: '#8A7256', roof: '#6B4E3D', pitch: 5 },
  ranch:        { wall: '#C8B89A', trim: '#7A6248', roof: '#5A4030', pitch: 3 },
}
function palette(style?: string) {
  return STYLE_PALETTE[(style ?? 'modern').toLowerCase()] ?? STYLE_PALETTE.modern
}

// ─── Floor materials by room type ─────────────────────────────────────────────
const FLOOR_MAT: Record<string, { color: string; roughness: number; metalness: number }> = {
  living_room:      { color: '#C8A87A', roughness: 0.68, metalness: 0.0 },
  family_room:      { color: '#BFA070', roughness: 0.70, metalness: 0.0 },
  dining_room:      { color: '#B8966A', roughness: 0.65, metalness: 0.0 },
  kitchen:          { color: '#D8D0C0', roughness: 0.38, metalness: 0.04 },
  master_bedroom:   { color: '#C8B8A0', roughness: 0.92, metalness: 0.0 },
  bedroom:          { color: '#C0B098', roughness: 0.92, metalness: 0.0 },
  bathroom:         { color: '#E0E4E8', roughness: 0.28, metalness: 0.06 },
  ensuite_bathroom: { color: '#D8DCE0', roughness: 0.28, metalness: 0.06 },
  half_bath:        { color: '#D8DCE0', roughness: 0.28, metalness: 0.06 },
  hallway:          { color: '#C4B898', roughness: 0.55, metalness: 0.0 },
  foyer:            { color: '#C0B090', roughness: 0.50, metalness: 0.0 },
  home_office:      { color: '#B8A880', roughness: 0.72, metalness: 0.0 },
  garage:           { color: '#A8A8A4', roughness: 0.90, metalness: 0.0 },
  laundry_room:     { color: '#C8C8C4', roughness: 0.60, metalness: 0.0 },
  walk_in_closet:   { color: '#C0B498', roughness: 0.80, metalness: 0.0 },
  mudroom:          { color: '#B0A890', roughness: 0.78, metalness: 0.0 },
  pantry:           { color: '#C0B898', roughness: 0.75, metalness: 0.0 },
}
function floorMat(type: string) {
  return FLOOR_MAT[type] ?? { color: '#C0B898', roughness: 0.75, metalness: 0.0 }
}

// ─── Hip Roof Geometry ────────────────────────────────────────────────────────
function makeHipRoofGeometry(W: number, D: number, pitch: number, overhang: number) {
  // All in THREE units. Centered at (0, 0, 0) = eave level center
  const HW = W / 2 + overhang
  const HD = D / 2 + overhang
  const run = Math.min(HW, HD)
  const pH = run * (pitch / 12)

  // Eave corners
  const FL = [-HW, 0, -HD] as [number, number, number]
  const FR = [ HW, 0, -HD] as [number, number, number]
  const BR = [ HW, 0,  HD] as [number, number, number]
  const BL = [-HW, 0,  HD] as [number, number, number]

  let verts: number[]

  if (W > D) {
    // Ridge runs along X
    const RL: [number, number, number] = [-HW + run, pH, 0]
    const RR: [number, number, number] = [ HW - run, pH, 0]
    verts = [
      ...FL, ...FR, ...RR,  // front (quad tri 1)
      ...FL, ...RR, ...RL,  // front (quad tri 2)
      ...BR, ...BL, ...RL,  // back  (quad tri 1)
      ...BR, ...RL, ...RR,  // back  (quad tri 2)
      ...FL, ...RL, ...BL,  // left  (triangle)
      ...FR, ...BR, ...RR,  // right (triangle)
    ]
  } else if (D > W) {
    // Ridge runs along Z
    const RF: [number, number, number] = [0, pH, -HD + run]
    const RB: [number, number, number] = [0, pH,  HD - run]
    verts = [
      ...FL, ...BL, ...RB,  // left  (quad tri 1)
      ...FL, ...RB, ...RF,  // left  (quad tri 2)
      ...FR, ...BR, ...RB,  // right (quad tri 1) — fix winding
      ...FR, ...RB, ...RF,  // right (quad tri 2)
      ...FL, ...FR, ...RF,  // front (triangle)
      ...BL, ...BR, ...RB,  // back  (triangle) — fix winding
    ]
  } else {
    // Square — single pyramid apex
    const APEX: [number, number, number] = [0, pH, 0]
    verts = [
      ...FL, ...FR, ...APEX,  // front
      ...FR, ...BR, ...APEX,  // right
      ...BR, ...BL, ...APEX,  // back
      ...BL, ...FL, ...APEX,  // left
    ]
  }

  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3))
  geo.computeVertexNormals()
  return geo
}

// ─── Furniture primitives ─────────────────────────────────────────────────────
function Sofa({ rw, rd }: { rw: number; rd: number }) {
  const sw = Math.min(rw * 0.7, 0.72)
  const sd = Math.min(rd * 0.25, 0.22)
  const sh = 0.08
  const armH = 0.16
  return (
    <group position={[0, 0, rd * 0.22]}>
      {/* Seat */}
      <mesh position={[0, sh / 2, 0]}>
        <boxGeometry args={[sw, sh, sd]} />
        <meshStandardMaterial color="#8B7460" roughness={0.9} />
      </mesh>
      {/* Back */}
      <mesh position={[0, sh + armH / 2, -sd / 2 + 0.02]}>
        <boxGeometry args={[sw, armH, 0.06]} />
        <meshStandardMaterial color="#7A6350" roughness={0.88} />
      </mesh>
      {/* Arms */}
      {[-1, 1].map(side => (
        <mesh key={side} position={[(sw / 2 + 0.02) * side, sh / 2 + armH / 4, 0]}>
          <boxGeometry args={[0.05, armH * 0.6, sd]} />
          <meshStandardMaterial color="#7A6350" roughness={0.88} />
        </mesh>
      ))}
    </group>
  )
}

function CoffeeTable({ rw, rd }: { rw: number; rd: number }) {
  const tw = Math.min(rw * 0.3, 0.34)
  const td = Math.min(rd * 0.18, 0.18)
  return (
    <group position={[0, 0.04, -rd * 0.08]}>
      <mesh position={[0, 0.025, 0]}>
        <boxGeometry args={[tw, 0.025, td]} />
        <meshStandardMaterial color="#6B5240" roughness={0.65} metalness={0.05} />
      </mesh>
      {[[-1, -1], [-1, 1], [1, -1], [1, 1]].map(([sx, sz], i) => (
        <mesh key={i} position={[sx * (tw / 2 - 0.02), 0.01, sz * (td / 2 - 0.02)]}>
          <boxGeometry args={[0.02, 0.025, 0.02]} />
          <meshStandardMaterial color="#5A4030" roughness={0.75} />
        </mesh>
      ))}
    </group>
  )
}

function Bed({ rw, rd }: { rw: number; rd: number }) {
  const bw = Math.min(rw * 0.62, 0.56)
  const bd = Math.min(rd * 0.65, 0.60)
  return (
    <group position={[0, 0, rd * 0.08]}>
      {/* Mattress */}
      <mesh position={[0, 0.07, 0]}>
        <boxGeometry args={[bw, 0.07, bd]} />
        <meshStandardMaterial color="#E8E0D4" roughness={0.95} />
      </mesh>
      {/* Pillow */}
      <mesh position={[0, 0.12, -bd * 0.35]}>
        <boxGeometry args={[bw * 0.8, 0.04, bd * 0.2]} />
        <meshStandardMaterial color="#F5F0E8" roughness={0.98} />
      </mesh>
      {/* Headboard */}
      <mesh position={[0, 0.18, -bd / 2 - 0.02]}>
        <boxGeometry args={[bw + 0.04, 0.3, 0.05]} />
        <meshStandardMaterial color="#7A5C3E" roughness={0.75} />
      </mesh>
      {/* Bed frame */}
      <mesh position={[0, 0.02, 0]}>
        <boxGeometry args={[bw + 0.04, 0.04, bd + 0.04]} />
        <meshStandardMaterial color="#6B4C2E" roughness={0.78} />
      </mesh>
    </group>
  )
}

function KitchenCounter({ rw, rd }: { rw: number; rd: number }) {
  const cd = 0.055  // counter depth
  const ch = 0.20
  return (
    <>
      {/* Back counter */}
      <mesh position={[0, ch / 2, -rd / 2 + cd / 2 + WT]}>
        <boxGeometry args={[rw - WT * 2, ch, cd]} />
        <meshStandardMaterial color="#C8C4BE" roughness={0.45} metalness={0.06} />
      </mesh>
      {/* Right counter */}
      <mesh position={[rw / 2 - cd / 2 - WT, ch / 2, 0]}>
        <boxGeometry args={[cd, ch, rd - WT * 2]} />
        <meshStandardMaterial color="#C8C4BE" roughness={0.45} metalness={0.06} />
      </mesh>
      {/* Island (only if room is large enough) */}
      {rw > 0.9 && rd > 0.9 && (
        <mesh position={[0, ch * 0.7, 0]}>
          <boxGeometry args={[rw * 0.35, ch * 0.7, rd * 0.3]} />
          <meshStandardMaterial color="#B8B4AE" roughness={0.42} metalness={0.08} />
        </mesh>
      )}
    </>
  )
}

function DiningSet({ rw, rd }: { rw: number; rd: number }) {
  const tw = Math.min(rw * 0.45, 0.45)
  const td = Math.min(rd * 0.35, 0.35)
  return (
    <group position={[0, 0, 0]}>
      {/* Table */}
      <mesh position={[0, 0.115, 0]}>
        <boxGeometry args={[tw, 0.025, td]} />
        <meshStandardMaterial color="#8B6848" roughness={0.62} metalness={0.03} />
      </mesh>
      {/* Table legs */}
      {[[-1, -1], [-1, 1], [1, -1], [1, 1]].map(([sx, sz], i) => (
        <mesh key={i} position={[sx * (tw / 2 - 0.03), 0.06, sz * (td / 2 - 0.03)]}>
          <boxGeometry args={[0.025, 0.115, 0.025]} />
          <meshStandardMaterial color="#7A5838" roughness={0.70} />
        </mesh>
      ))}
      {/* Chairs */}
      {[[-1, 0], [1, 0], [0, -1], [0, 1]].map(([sx, sz], i) => (
        <group key={i} position={[sx * (tw / 2 + 0.07), 0, sz * (td / 2 + 0.07)]}>
          <mesh position={[0, 0.065, 0]}>
            <boxGeometry args={[0.09, 0.02, 0.09]} />
            <meshStandardMaterial color="#9A7858" roughness={0.80} />
          </mesh>
          <mesh position={[0, 0.12, -0.04]}>
            <boxGeometry args={[0.09, 0.11, 0.02]} />
            <meshStandardMaterial color="#9A7858" roughness={0.80} />
          </mesh>
        </group>
      ))}
    </group>
  )
}

function Desk({ rw, rd }: { rw: number; rd: number }) {
  const dw = Math.min(rw * 0.55, 0.50)
  return (
    <group position={[0, 0, rd * 0.3]}>
      <mesh position={[0, 0.12, 0]}>
        <boxGeometry args={[dw, 0.02, 0.22]} />
        <meshStandardMaterial color="#8B7050" roughness={0.65} />
      </mesh>
      {[-1, 1].map(side => (
        <mesh key={side} position={[(dw / 2 - 0.03) * side, 0.06, 0]}>
          <boxGeometry args={[0.04, 0.12, 0.20]} />
          <meshStandardMaterial color="#7A6040" roughness={0.70} />
        </mesh>
      ))}
    </group>
  )
}

function Bathtub({ rw, rd }: { rw: number; rd: number }) {
  const tw = Math.min(rw * 0.55, 0.40)
  const td = Math.min(rd * 0.38, 0.28)
  return (
    <group position={[rw * 0.18, 0, -rd * 0.28]}>
      <mesh position={[0, 0.06, 0]}>
        <boxGeometry args={[tw, 0.12, td]} />
        <meshStandardMaterial color="#F0F0EE" roughness={0.22} metalness={0.08} />
      </mesh>
    </group>
  )
}

function RoomFurniture({ type, rw, rd }: { type: string; rw: number; rd: number }) {
  const t = type.toLowerCase()
  if (t === 'living_room' || t === 'family_room') return (
    <>
      <Sofa rw={rw} rd={rd} />
      <CoffeeTable rw={rw} rd={rd} />
    </>
  )
  if (t === 'master_bedroom' || t === 'bedroom') return <Bed rw={rw} rd={rd} />
  if (t === 'kitchen') return <KitchenCounter rw={rw} rd={rd} />
  if (t === 'dining_room') return <DiningSet rw={rw} rd={rd} />
  if (t === 'home_office') return <Desk rw={rw} rd={rd} />
  if (t === 'bathroom' || t === 'ensuite_bathroom') return <Bathtub rw={rw} rd={rd} />
  return null
}

// ─── Exterior Scene ───────────────────────────────────────────────────────────
function ExteriorScene({ plan }: { plan: FloorPlan }) {
  const pal = palette(plan.style)
  const wallH = (plan.ceilingHeight ?? 9) * S
  const pitch = pal.pitch

  const mainRooms = plan.rooms.filter(r =>
    !['patio', 'deck', 'rear_patio', 'outdoor_living', 'front_porch'].includes(r.type)
  )
  const outdoorRooms = plan.rooms.filter(r =>
    ['patio', 'deck', 'rear_patio', 'outdoor_living', 'front_porch'].includes(r.type)
  )

  // House bounding box
  const minX = Math.min(...mainRooms.map(r => r.x))
  const maxX = Math.max(...mainRooms.map(r => r.x + r.width))
  const minZ = Math.min(...mainRooms.map(r => r.y))
  const maxZ = Math.max(...mainRooms.map(r => r.y + r.height))
  const houseW = (maxX - minX) * S
  const houseD = (maxZ - minZ) * S
  const cx = ((minX + maxX) / 2) * S
  const cz = ((minZ + maxZ) / 2) * S

  const roofGeo = useMemo(
    () => makeHipRoofGeometry(houseW, houseD, pitch, 0.22),
    [houseW, houseD, pitch]
  )

  // Group garage rooms separately
  const garageRooms = mainRooms.filter(r => r.type === 'garage')
  const bodyRooms = mainRooms.filter(r => r.type !== 'garage')

  return (
    <>
      {/* Ground plane */}
      <mesh position={[cx, -0.01, cz]} receiveShadow>
        <boxGeometry args={[houseW + 5, 0.02, houseD + 5]} />
        <meshStandardMaterial color="#7A9E5A" roughness={0.94} />
      </mesh>
      {/* Driveway from garage */}
      {garageRooms.map(g => {
        const gx = (g.x + g.width / 2) * S
        return (
          <mesh key={g.id} position={[gx, -0.005, cz - houseD / 2 - 0.8]} receiveShadow>
            <boxGeometry args={[g.width * S, 0.01, 1.6]} />
            <meshStandardMaterial color="#B0B0B0" roughness={0.88} />
          </mesh>
        )
      })}

      {/* Body rooms */}
      {bodyRooms.map(room => {
        const rw = room.width * S
        const rd = room.height * S
        const px = (room.x + room.width / 2) * S
        const pz = (room.y + room.height / 2) * S
        return (
          <mesh key={room.id} position={[px, wallH / 2, pz]} castShadow receiveShadow>
            <boxGeometry args={[rw, wallH, rd]} />
            <meshStandardMaterial color={pal.wall} roughness={0.82} />
          </mesh>
        )
      })}

      {/* Garage — slightly lower */}
      {garageRooms.map(room => {
        const rw = room.width * S
        const rd = room.height * S
        const px = (room.x + room.width / 2) * S
        const pz = (room.y + room.height / 2) * S
        const gh = wallH * 0.92
        return (
          <mesh key={room.id} position={[px, gh / 2, pz]} castShadow receiveShadow>
            <boxGeometry args={[rw, gh, rd]} />
            <meshStandardMaterial color={pal.wall} roughness={0.85} />
          </mesh>
        )
      })}

      {/* Outdoor slabs */}
      {outdoorRooms.map(room => {
        const rw = room.width * S
        const rd = room.height * S
        const px = (room.x + room.width / 2) * S
        const pz = (room.y + room.height / 2) * S
        return (
          <mesh key={room.id} position={[px, 0.015, pz]} receiveShadow>
            <boxGeometry args={[rw, 0.03, rd]} />
            <meshStandardMaterial color="#C8C4BC" roughness={0.88} />
          </mesh>
        )
      })}

      {/* Hip roof */}
      <mesh
        geometry={roofGeo}
        position={[cx, wallH, cz]}
        castShadow
      >
        <meshStandardMaterial color={pal.roof} roughness={0.88} side={THREE.DoubleSide} />
      </mesh>

      {/* Roof trim / fascia */}
      <mesh position={[cx, wallH - 0.01, cz]}>
        <boxGeometry args={[houseW + 0.44 + 0.012, 0.018, houseD + 0.44 + 0.012]} />
        <meshStandardMaterial color={pal.trim} roughness={0.80} />
      </mesh>

      {/* Windows on front face — derived from room widths */}
      {bodyRooms
        .filter(r => r.y === minZ && !['hallway', 'foyer', 'mudroom'].includes(r.type))
        .map(room => {
          const px = (room.x + room.width / 2) * S
          const winW = Math.min(room.width * S * 0.45, 0.28)
          const winH = wallH * 0.32
          return (
            <group key={`win-${room.id}`} position={[px, wallH * 0.55, minZ * S - 0.005]}>
              {/* Frame */}
              <mesh>
                <boxGeometry args={[winW + 0.02, winH + 0.02, 0.018]} />
                <meshStandardMaterial color={pal.trim} roughness={0.72} />
              </mesh>
              {/* Glass */}
              <mesh>
                <boxGeometry args={[winW, winH, 0.022]} />
                <meshStandardMaterial color="#B8D4E8" roughness={0.06} metalness={0.15} transparent opacity={0.55} />
              </mesh>
              {/* Mullion */}
              <mesh position={[0, 0, 0.015]}>
                <boxGeometry args={[winW, 0.008, 0.004]} />
                <meshStandardMaterial color={pal.trim} roughness={0.75} />
              </mesh>
            </group>
          )
        })}

      {/* Front door */}
      <group position={[cx, wallH * 0.3, minZ * S - 0.005]}>
        <mesh>
          <boxGeometry args={[0.12, wallH * 0.58, 0.02]} />
          <meshStandardMaterial color={pal.trim} roughness={0.68} />
        </mesh>
        <mesh>
          <boxGeometry args={[0.09, wallH * 0.52, 0.024]} />
          <meshStandardMaterial color={pal.trim} roughness={0.6} metalness={0.05} />
        </mesh>
        {/* Door glass lite */}
        <mesh position={[0, wallH * 0.1, 0.013]}>
          <boxGeometry args={[0.06, wallH * 0.14, 0.01]} />
          <meshStandardMaterial color="#C8DCE8" roughness={0.1} transparent opacity={0.6} />
        </mesh>
        {/* Stoop */}
        <mesh position={[0, -wallH * 0.32, -0.12]}>
          <boxGeometry args={[0.22, 0.025, 0.24]} />
          <meshStandardMaterial color="#B8B4AC" roughness={0.88} />
        </mesh>
      </group>

      {/* Trees */}
      {[0.1, 0.9].map((tx, i) => {
        const tpx = minX * S + tx * houseW
        const tpz = minZ * S - 0.8
        return (
          <group key={i} position={[tpx, 0, tpz]}>
            <mesh position={[0, 0.08, 0]}>
              <cylinderGeometry args={[0.025, 0.04, 0.16, 8]} />
              <meshStandardMaterial color="#5A3C1E" roughness={0.95} />
            </mesh>
            <mesh position={[0, 0.32, 0]} castShadow>
              <sphereGeometry args={[0.22, 8, 6]} />
              <meshStandardMaterial color="#3A7028" roughness={0.92} />
            </mesh>
            <mesh position={[0, 0.44, 0]} castShadow>
              <sphereGeometry args={[0.16, 8, 6]} />
              <meshStandardMaterial color="#48882E" roughness={0.90} />
            </mesh>
          </group>
        )
      })}
    </>
  )
}

// ─── Dollhouse Scene ──────────────────────────────────────────────────────────
function DollhouseScene({ plan }: { plan: FloorPlan }) {
  const wallH = (plan.ceilingHeight ?? 9) * S
  const WALL_COL = '#E8ECF2'

  return (
    <>
      {plan.rooms.map(room => {
        const rw = room.width * S
        const rd = room.height * S
        const px = (room.x + room.width / 2) * S
        const pz = (room.y + room.height / 2) * S
        const mat = floorMat(room.type)
        const isOutdoor = ['patio', 'deck', 'rear_patio', 'outdoor_living', 'front_porch'].includes(room.type)

        return (
          <group key={room.id} position={[px, 0, pz]}>
            {/* Floor slab */}
            <mesh position={[0, 0.005, 0]} receiveShadow>
              <boxGeometry args={[rw, 0.01, rd]} />
              <meshStandardMaterial color={mat.color} roughness={mat.roughness} metalness={mat.metalness} />
            </mesh>

            {!isOutdoor && (
              <>
                {/* Walls */}
                <mesh position={[0, wallH / 2, -rd / 2 + WT / 2]} castShadow>
                  <boxGeometry args={[rw, wallH, WT]} />
                  <meshStandardMaterial color={WALL_COL} roughness={0.78} />
                </mesh>
                <mesh position={[0, wallH / 2, rd / 2 - WT / 2]} castShadow>
                  <boxGeometry args={[rw, wallH, WT]} />
                  <meshStandardMaterial color={WALL_COL} roughness={0.78} />
                </mesh>
                <mesh position={[-rw / 2 + WT / 2, wallH / 2, 0]} castShadow>
                  <boxGeometry args={[WT, wallH, rd]} />
                  <meshStandardMaterial color={WALL_COL} roughness={0.78} />
                </mesh>
                <mesh position={[rw / 2 - WT / 2, wallH / 2, 0]} castShadow>
                  <boxGeometry args={[WT, wallH, rd]} />
                  <meshStandardMaterial color={WALL_COL} roughness={0.78} />
                </mesh>

                {/* Baseboard */}
                <mesh position={[0, 0.015, -rd / 2 + WT / 2]}>
                  <boxGeometry args={[rw, 0.028, WT + 0.004]} />
                  <meshStandardMaterial color="#D8D0C0" roughness={0.75} />
                </mesh>

                {/* Ceiling — translucent so room is visible from above */}
                <mesh position={[0, wallH - 0.003, 0]}>
                  <boxGeometry args={[rw, 0.006, rd]} />
                  <meshStandardMaterial color="#F5F4F2" roughness={0.9} transparent opacity={0.18} />
                </mesh>

                {/* Room label */}
                <Text
                  position={[0, 0.015, 0]}
                  rotation={[-Math.PI / 2, 0, 0]}
                  fontSize={Math.min(rw, rd) * 0.14}
                  color="#3A3020"
                  anchorX="center"
                  anchorY="middle"
                >
                  {room.name.toUpperCase()}
                </Text>

                {/* Furniture */}
                <RoomFurniture type={room.type} rw={rw} rd={rd} />
              </>
            )}
          </group>
        )
      })}
    </>
  )
}

// ─── Top-view Scene ───────────────────────────────────────────────────────────
function TopViewScene({ plan }: { plan: FloorPlan }) {
  const WT2 = 0.016
  return (
    <>
      {plan.rooms.map(room => {
        const rw = room.width * S
        const rd = room.height * S
        const px = (room.x + room.width / 2) * S
        const pz = (room.y + room.height / 2) * S
        const mat = floorMat(room.type)
        return (
          <group key={room.id} position={[px, 0, pz]}>
            <mesh position={[0, 0.002, 0]}>
              <boxGeometry args={[rw, 0.004, rd]} />
              <meshStandardMaterial color={mat.color} roughness={mat.roughness} />
            </mesh>
            {/* Walls as flat borders */}
            <mesh position={[0, 0.006, -rd / 2]}>
              <boxGeometry args={[rw, 0.008, WT2]} />
              <meshStandardMaterial color="#404040" roughness={0.8} />
            </mesh>
            <mesh position={[0, 0.006, rd / 2]}>
              <boxGeometry args={[rw, 0.008, WT2]} />
              <meshStandardMaterial color="#404040" roughness={0.8} />
            </mesh>
            <mesh position={[-rw / 2, 0.006, 0]}>
              <boxGeometry args={[WT2, 0.008, rd]} />
              <meshStandardMaterial color="#404040" roughness={0.8} />
            </mesh>
            <mesh position={[rw / 2, 0.006, 0]}>
              <boxGeometry args={[WT2, 0.008, rd]} />
              <meshStandardMaterial color="#404040" roughness={0.8} />
            </mesh>
            <Text
              position={[0, 0.012, 0]}
              rotation={[-Math.PI / 2, 0, 0]}
              fontSize={Math.min(rw, rd) * 0.13}
              color="#1a1a1a"
              anchorX="center"
              anchorY="middle"
            >
              {room.name}
            </Text>
          </group>
        )
      })}
    </>
  )
}

// ─── Walkthrough ─────────────────────────────────────────────────────────────
function WalkthroughScene({ plan }: { plan: FloorPlan }) {
  const wallH = (plan.ceilingHeight ?? 9) * S
  return (
    <>
      {plan.rooms.map(room => {
        const rw = room.width * S
        const rd = room.height * S
        const px = (room.x + room.width / 2) * S
        const pz = (room.y + room.height / 2) * S
        const mat = floorMat(room.type)
        return (
          <group key={room.id} position={[px, 0, pz]}>
            <mesh position={[0, 0.005, 0]} receiveShadow>
              <boxGeometry args={[rw, 0.01, rd]} />
              <meshStandardMaterial color={mat.color} roughness={mat.roughness} metalness={mat.metalness} />
            </mesh>
            {['N', 'S', 'W', 'E'].map(side => {
              const isNS = side === 'N' || side === 'S'
              const sign = (side === 'S' || side === 'E') ? 1 : -1
              const pos: [number, number, number] = isNS
                ? [0, wallH / 2, sign * (rd / 2 - WT / 2)]
                : [sign * (rw / 2 - WT / 2), wallH / 2, 0]
              const size: [number, number, number] = isNS
                ? [rw, wallH, WT]
                : [WT, wallH, rd]
              return (
                <mesh key={side} position={pos} castShadow>
                  <boxGeometry args={size} />
                  <meshStandardMaterial color="#ECEEF2" roughness={0.72} />
                </mesh>
              )
            })}
            {/* Ceiling */}
            <mesh position={[0, wallH, 0]}>
              <boxGeometry args={[rw, 0.008, rd]} />
              <meshStandardMaterial color="#F8F8F6" roughness={0.88} />
            </mesh>
            {/* Baseboard */}
            {['N', 'S'].map(side => {
              const sign = side === 'S' ? 1 : -1
              return (
                <mesh key={side} position={[0, 0.014, sign * (rd / 2 - WT / 2)]}>
                  <boxGeometry args={[rw, 0.028, WT + 0.005]} />
                  <meshStandardMaterial color="#D4CEC4" roughness={0.78} />
                </mesh>
              )
            })}
            <Text
              position={[0, wallH * 0.62, -rd / 2 + WT + 0.01]}
              fontSize={wallH * 0.11}
              color="#60708A"
              anchorX="center"
              anchorY="middle"
            >
              {room.name}
            </Text>
            <RoomFurniture type={room.type} rw={rw} rd={rd} />
          </group>
        )
      })}
    </>
  )
}

// ─── First-person controller ──────────────────────────────────────────────────
function FirstPersonController({ plan, wallH }: { plan: FloorPlan; wallH: number }) {
  const { camera, gl } = useThree()
  const move = useRef({ f: false, b: false, l: false, r: false })
  const yaw = useRef(0)
  const pitch = useRef(0)
  const locked = useRef(false)
  const SPEED = 0.022

  const startRoom = plan.rooms[0]
  const startPos = useMemo(() => new THREE.Vector3(
    (startRoom.x + startRoom.width / 2) * S,
    wallH * 0.62,
    (startRoom.y + startRoom.height / 2) * S
  ), [startRoom, wallH])

  useEffect(() => { camera.position.copy(startPos) }, [camera, startPos])

  useEffect(() => {
    const canvas = gl.domElement
    const onDown = (e: KeyboardEvent) => {
      if (e.code === 'KeyW' || e.code === 'ArrowUp')    move.current.f = true
      if (e.code === 'KeyS' || e.code === 'ArrowDown')  move.current.b = true
      if (e.code === 'KeyA' || e.code === 'ArrowLeft')  move.current.l = true
      if (e.code === 'KeyD' || e.code === 'ArrowRight') move.current.r = true
    }
    const onUp = (e: KeyboardEvent) => {
      if (e.code === 'KeyW' || e.code === 'ArrowUp')    move.current.f = false
      if (e.code === 'KeyS' || e.code === 'ArrowDown')  move.current.b = false
      if (e.code === 'KeyA' || e.code === 'ArrowLeft')  move.current.l = false
      if (e.code === 'KeyD' || e.code === 'ArrowRight') move.current.r = false
    }
    const onClick = () => canvas.requestPointerLock()
    const onLock = () => { locked.current = document.pointerLockElement === canvas }
    const onMove = (e: MouseEvent) => {
      if (!locked.current) return
      yaw.current   -= e.movementX * 0.002
      pitch.current  = Math.max(-1.0, Math.min(0.6, pitch.current - e.movementY * 0.002))
    }
    canvas.addEventListener('click', onClick)
    canvas.addEventListener('mousemove', onMove)
    document.addEventListener('pointerlockchange', onLock)
    window.addEventListener('keydown', onDown)
    window.addEventListener('keyup', onUp)
    return () => {
      canvas.removeEventListener('click', onClick)
      canvas.removeEventListener('mousemove', onMove)
      document.removeEventListener('pointerlockchange', onLock)
      window.removeEventListener('keydown', onDown)
      window.removeEventListener('keyup', onUp)
    }
  }, [gl])

  useFrame(() => {
    const dir = new THREE.Vector3(Math.sin(yaw.current), 0, Math.cos(yaw.current))
    const right = new THREE.Vector3(Math.cos(yaw.current), 0, -Math.sin(yaw.current))
    if (move.current.f) camera.position.addScaledVector(dir, -SPEED)
    if (move.current.b) camera.position.addScaledVector(dir,  SPEED)
    if (move.current.l) camera.position.addScaledVector(right, -SPEED)
    if (move.current.r) camera.position.addScaledVector(right,  SPEED)
    camera.position.y = wallH * 0.62
    camera.rotation.order = 'YXZ'
    camera.rotation.y = yaw.current
    camera.rotation.x = pitch.current
  })

  return null
}

// ─── Camera presets ───────────────────────────────────────────────────────────
function CameraRig({ plan, mode }: { plan: FloorPlan; mode: Mode }) {
  const { camera } = useThree()
  const cx = (plan.totalWidth / 2) * S
  const cz = (plan.totalHeight / 2) * S
  const diag = Math.sqrt(plan.totalWidth ** 2 + plan.totalHeight ** 2) * S

  useEffect(() => {
    if (mode === 'exterior') {
      camera.position.set(cx - diag * 0.55, diag * 0.5, cz - diag * 0.85)
      camera.lookAt(cx, 0, cz)
    } else if (mode === 'dollhouse') {
      camera.position.set(cx - diag * 0.45, diag * 0.85, cz - diag * 0.6)
      camera.lookAt(cx, 0, cz)
    } else if (mode === 'topview') {
      camera.position.set(cx, diag * 1.1, cz + 0.001)
      camera.lookAt(cx, 0, cz)
    }
  }, [mode, cx, cz, diag, camera])

  return null
}

// ─── Main export ──────────────────────────────────────────────────────────────
export default function View3D({ plan, initialMode }: { plan: FloorPlan; initialMode?: Mode }) {
  const mode: Mode = (initialMode as Mode) ?? 'exterior'
  const wallH = (plan.ceilingHeight ?? 9) * S
  const cx = (plan.totalWidth / 2) * S
  const cz = (plan.totalHeight / 2) * S

  return (
    <Canvas shadows style={{ width: '100%', height: '100%', background: '#D6E8F5' }}>
      <PerspectiveCamera makeDefault fov={60} near={0.01} far={80} />

      {/* Lighting */}
      <ambientLight intensity={0.55} />
      <directionalLight
        position={[cx + 4, 6, cz - 6]}
        intensity={1.4}
        castShadow
        shadow-mapSize-width={1024}
        shadow-mapSize-height={1024}
        shadow-camera-near={0.5}
        shadow-camera-far={40}
        shadow-camera-left={-8}
        shadow-camera-right={8}
        shadow-camera-top={8}
        shadow-camera-bottom={-8}
      />
      <directionalLight position={[cx - 3, 3, cz + 4]} intensity={0.35} />

      <Sky sunPosition={[100, 60, -80]} turbidity={4} rayleigh={0.6} />

      {/* Scene content */}
      {mode === 'exterior'    && <ExteriorScene plan={plan} />}
      {mode === 'dollhouse'   && <DollhouseScene plan={plan} />}
      {mode === 'topview'     && <TopViewScene plan={plan} />}
      {mode === 'walkthrough' && (
        <>
          <WalkthroughScene plan={plan} />
          <FirstPersonController plan={plan} wallH={wallH} />
        </>
      )}

      {mode !== 'walkthrough' && (
        <>
          <CameraRig plan={plan} mode={mode} />
          <OrbitControls
            target={[cx, 0, cz]}
            enableDamping
            dampingFactor={0.08}
            minDistance={0.5}
            maxDistance={20}
          />
        </>
      )}

      {mode === 'walkthrough' && (
        <Html position={[cx, wallH * 0.1, cz]} center>
          <div style={{
            background: 'rgba(0,0,0,0.55)', color: 'white',
            padding: '5px 12px', borderRadius: 6, fontSize: 11,
            pointerEvents: 'none', whiteSpace: 'nowrap',
          }}>
            Click to capture mouse · WASD / Arrows to move
          </div>
        </Html>
      )}
    </Canvas>
  )
}
