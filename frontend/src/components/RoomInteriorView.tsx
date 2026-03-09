import { useRef } from 'react'
import { Canvas, useFrame } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera, Environment } from '@react-three/drei'
import * as THREE from 'three'
import { Room } from '../types/floorplan'

// US-realistic interior materials per room type
const ROOM_MATS: Record<string, {
  floor: string; floorRoughness: number
  walls: string; wallRoughness: number
  ceiling: string
  trim: string
  accent?: string
}> = {
  default:         { floor: '#C4955A', floorRoughness: 0.35, walls: '#C8C4BC', wallRoughness: 0.92, ceiling: '#F8F6F2', trim: '#F2EDE8' },
  living:          { floor: '#C4955A', floorRoughness: 0.32, walls: '#C8C4BC', wallRoughness: 0.9,  ceiling: '#F8F6F2', trim: '#F2EDE8', accent: '#9E8870' },
  great_room:      { floor: '#C4955A', floorRoughness: 0.30, walls: '#CAC6BE', wallRoughness: 0.9,  ceiling: '#F8F6F2', trim: '#F2EDE8', accent: '#9E8870' },
  family:          { floor: '#C4955A', floorRoughness: 0.35, walls: '#C8C0B4', wallRoughness: 0.9,  ceiling: '#F8F6F2', trim: '#F2EDE8' },
  kitchen:         { floor: '#E2DDD4', floorRoughness: 0.18, walls: '#F0EDE8', wallRoughness: 0.88, ceiling: '#F8F6F2', trim: '#F2EDE8', accent: '#2E3B50' },
  dining:          { floor: '#B88C5C', floorRoughness: 0.30, walls: '#CCC8BE', wallRoughness: 0.92, ceiling: '#F8F6F2', trim: '#F2EDE8' },
  master_bedroom:  { floor: '#C8BCB4', floorRoughness: 0.98, walls: '#B8C4CC', wallRoughness: 0.94, ceiling: '#F8F6F2', trim: '#F2EDE8', accent: '#6A7A8A' },
  bedroom:         { floor: '#C8BCB4', floorRoughness: 0.98, walls: '#C8CCBC', wallRoughness: 0.94, ceiling: '#F8F6F2', trim: '#F2EDE8' },
  bathroom:        { floor: '#ECEAE4', floorRoughness: 0.12, walls: '#F0EEEA', wallRoughness: 0.10, ceiling: '#F8F6F2', trim: '#F2EDE8' },
  ensuite_bathroom:{ floor: '#ECEAE4', floorRoughness: 0.10, walls: '#F0EEEA', wallRoughness: 0.08, ceiling: '#F8F6F2', trim: '#F2EDE8', accent: '#8C7860' },
  office:          { floor: '#8C6840', floorRoughness: 0.28, walls: '#2C3C44', wallRoughness: 0.94, ceiling: '#F0EDE8', trim: '#DDD8CE' },
  hallway:         { floor: '#C4955A', floorRoughness: 0.30, walls: '#D0CCC4', wallRoughness: 0.92, ceiling: '#F8F6F2', trim: '#F2EDE8' },
  foyer:           { floor: '#D0C8B0', floorRoughness: 0.18, walls: '#CCC8C0', wallRoughness: 0.90, ceiling: '#F8F6F2', trim: '#F2EDE8' },
  laundry:         { floor: '#E0DCD4', floorRoughness: 0.15, walls: '#E8E4DC', wallRoughness: 0.88, ceiling: '#F8F6F2', trim: '#F2EDE8' },
  garage:          { floor: '#B0ACA4', floorRoughness: 0.95, walls: '#C8C4BC', wallRoughness: 0.92, ceiling: '#D8D4CC', trim: '#C8C4BC' },
  closet:          { floor: '#C8BCB4', floorRoughness: 0.98, walls: '#F0EDE8', wallRoughness: 0.92, ceiling: '#F8F6F2', trim: '#F2EDE8' },
}

function getMat(roomType: string) {
  const t = roomType.toLowerCase().replace(/-/g, '_')
  for (const key of Object.keys(ROOM_MATS)) {
    if (t.includes(key)) return ROOM_MATS[key]
  }
  return ROOM_MATS.default
}

// ─── Furniture ───────────────────────────────────────────────────────────────

function Sofa({ rw, rd }: { rw: number; rd: number }) {
  const l = Math.min(rw * 0.55, 3.2)
  const x = -rw / 2 + l / 2 + 0.15
  const z = -rd / 2 + 0.45
  return (
    <group position={[x, 0, z]}>
      {/* Seat */}
      <mesh position={[0, 0.22, 0]} castShadow>
        <boxGeometry args={[l, 0.44, 0.85]} />
        <meshStandardMaterial color="#B0A898" roughness={0.9} />
      </mesh>
      {/* Back */}
      <mesh position={[0, 0.55, -0.36]} castShadow>
        <boxGeometry args={[l, 0.65, 0.12]} />
        <meshStandardMaterial color="#A8A098" roughness={0.9} />
      </mesh>
      {/* Left arm */}
      <mesh position={[-l / 2 + 0.07, 0.38, 0]} castShadow>
        <boxGeometry args={[0.14, 0.32, 0.85]} />
        <meshStandardMaterial color="#A8A098" roughness={0.9} />
      </mesh>
      {/* Right arm */}
      <mesh position={[l / 2 - 0.07, 0.38, 0]} castShadow>
        <boxGeometry args={[0.14, 0.32, 0.85]} />
        <meshStandardMaterial color="#A8A098" roughness={0.9} />
      </mesh>
      {/* Coffee table */}
      <mesh position={[0, 0.2, 0.75]} castShadow>
        <boxGeometry args={[l * 0.6, 0.06, 0.55]} />
        <meshStandardMaterial color="#7A5C3A" roughness={0.45} metalness={0.05} />
      </mesh>
      <mesh position={[0, 0.08, 0.75]} receiveShadow>
        <boxGeometry args={[l * 0.55, 0.16, 0.5]} />
        <meshStandardMaterial color="#6A4E2E" roughness={0.55} />
      </mesh>
    </group>
  )
}

function Bed({ rw, rd, isMaster }: { rw: number; rd: number; isMaster: boolean }) {
  const bw = isMaster ? 1.93 : 1.52  // king vs full
  const bl = isMaster ? 2.03 : 1.91
  const bh = 0.55
  const headH = 0.9
  const x = 0
  const z = -rd / 2 + bl / 2 + 0.15

  return (
    <group position={[x, 0, z]}>
      {/* Frame */}
      <mesh position={[0, bh * 0.4, 0]} castShadow receiveShadow>
        <boxGeometry args={[bw + 0.06, bh * 0.8, bl + 0.06]} />
        <meshStandardMaterial color="#5C4030" roughness={0.7} />
      </mesh>
      {/* Mattress */}
      <mesh position={[0, bh, 0]} castShadow>
        <boxGeometry args={[bw, bh * 0.6, bl]} />
        <meshStandardMaterial color="#EDE8E0" roughness={0.95} />
      </mesh>
      {/* Comforter */}
      <mesh position={[0, bh + 0.08, bl * 0.1]} castShadow>
        <boxGeometry args={[bw - 0.05, 0.14, bl * 0.72]} />
        <meshStandardMaterial color="#C8D4DC" roughness={0.98} />
      </mesh>
      {/* Pillow L */}
      <mesh position={[-bw * 0.22, bh + 0.18, -bl * 0.32]} castShadow>
        <boxGeometry args={[bw * 0.38, 0.14, 0.5]} />
        <meshStandardMaterial color="#F0EDE8" roughness={0.95} />
      </mesh>
      {/* Pillow R */}
      <mesh position={[bw * 0.22, bh + 0.18, -bl * 0.32]} castShadow>
        <boxGeometry args={[bw * 0.38, 0.14, 0.5]} />
        <meshStandardMaterial color="#F0EDE8" roughness={0.95} />
      </mesh>
      {/* Headboard */}
      <mesh position={[0, bh + headH * 0.4, -bl / 2 + 0.05]} castShadow>
        <boxGeometry args={[bw + 0.1, headH, 0.14]} />
        <meshStandardMaterial color="#4A3020" roughness={0.65} />
      </mesh>
      {/* Nightstand L */}
      <mesh position={[-bw / 2 - 0.28, 0.34, -bl * 0.28]} castShadow>
        <boxGeometry args={[0.5, 0.68, 0.4]} />
        <meshStandardMaterial color="#5C4030" roughness={0.65} />
      </mesh>
      {/* Nightstand R */}
      <mesh position={[bw / 2 + 0.28, 0.34, -bl * 0.28]} castShadow>
        <boxGeometry args={[0.5, 0.68, 0.4]} />
        <meshStandardMaterial color="#5C4030" roughness={0.65} />
      </mesh>
    </group>
  )
}

function KitchenCounters({ rw, rd }: { rw: number; rd: number }) {
  const counterH = 0.9
  const counterD = 0.64
  const cabH = 0.8
  const cabStart = counterH + 0.45

  return (
    <group>
      {/* Counter along back wall */}
      <mesh position={[0, counterH / 2, -rd / 2 + counterD / 2]} castShadow receiveShadow>
        <boxGeometry args={[rw * 0.9, counterH, counterD]} />
        <meshStandardMaterial color="#F0EDE8" roughness={0.85} />
      </mesh>
      {/* Countertop */}
      <mesh position={[0, counterH + 0.025, -rd / 2 + counterD / 2]} castShadow>
        <boxGeometry args={[rw * 0.9, 0.05, counterD + 0.02]} />
        <meshStandardMaterial color="#D8D4CC" roughness={0.22} metalness={0.05} />
      </mesh>
      {/* Upper cabinets */}
      <mesh position={[0, cabStart + cabH / 2, -rd / 2 + 0.18]} castShadow>
        <boxGeometry args={[rw * 0.85, cabH, 0.36]} />
        <meshStandardMaterial color="#F2EFEA" roughness={0.82} />
      </mesh>

      {/* Right-side counter (L-shape) */}
      <mesh position={[rw / 2 - counterD / 2, counterH / 2, 0]} castShadow receiveShadow>
        <boxGeometry args={[counterD, counterH, rd * 0.55]} />
        <meshStandardMaterial color="#F0EDE8" roughness={0.85} />
      </mesh>
      <mesh position={[rw / 2 - counterD / 2, counterH + 0.025, 0]} castShadow>
        <boxGeometry args={[counterD + 0.02, 0.05, rd * 0.55]} />
        <meshStandardMaterial color="#D8D4CC" roughness={0.22} metalness={0.05} />
      </mesh>

      {/* Island */}
      {rw > 3.8 && (
        <group position={[0, 0, rd * 0.18]}>
          <mesh position={[0, counterH / 2, 0]} castShadow receiveShadow>
            <boxGeometry args={[Math.min(rw * 0.45, 2.4), counterH, 1.1]} />
            <meshStandardMaterial color="#2E3B50" roughness={0.72} />
          </mesh>
          <mesh position={[0, counterH + 0.025, 0]}>
            <boxGeometry args={[Math.min(rw * 0.45, 2.4) + 0.02, 0.06, 1.12]} />
            <meshStandardMaterial color="#E8E4DC" roughness={0.2} metalness={0.05} />
          </mesh>
        </group>
      )}
    </group>
  )
}

function Toilet({ rw, rd }: { rw: number; rd: number }) {
  const x = rw / 2 - 0.3
  const z = rd / 2 - 0.5
  return (
    <group position={[-x + 0.15, 0, -z + 0.1]}>
      <mesh position={[0, 0.22, 0]} castShadow>
        <cylinderGeometry args={[0.2, 0.22, 0.44, 12]} />
        <meshStandardMaterial color="#F2EEE8" roughness={0.4} />
      </mesh>
      <mesh position={[0, 0.44, -0.12]}>
        <boxGeometry args={[0.38, 0.12, 0.52]} />
        <meshStandardMaterial color="#F0ECE6" roughness={0.4} />
      </mesh>
      <mesh position={[0, 0.52, -0.12]}>
        <boxGeometry args={[0.36, 0.06, 0.5]} />
        <meshStandardMaterial color="#EDEAE4" roughness={0.35} />
      </mesh>
    </group>
  )
}

function BathroomVanity({ rw }: { rw: number }) {
  const vw = Math.min(rw * 0.65, 1.52)
  return (
    <group position={[0, 0, 0]}>
      <mesh position={[0, 0.4, -0.3]} castShadow>
        <boxGeometry args={[vw, 0.8, 0.55]} />
        <meshStandardMaterial color="#F0EDE8" roughness={0.82} />
      </mesh>
      <mesh position={[0, 0.82, -0.28]} castShadow>
        <boxGeometry args={[vw + 0.02, 0.04, 0.57]} />
        <meshStandardMaterial color="#D8D4CC" roughness={0.18} metalness={0.08} />
      </mesh>
      {/* Mirror */}
      <mesh position={[0, 1.2, -0.48]}>
        <boxGeometry args={[vw - 0.1, 0.6, 0.02]} />
        <meshStandardMaterial color="#C8D4DC" roughness={0.02} metalness={0.9} />
      </mesh>
    </group>
  )
}

function Desk({ rw, rd }: { rw: number; rd: number }) {
  const dw = Math.min(rw * 0.6, 1.8)
  return (
    <group position={[0, 0, -rd / 2 + 0.42]}>
      <mesh position={[0, 0.74, 0]} castShadow>
        <boxGeometry args={[dw, 0.05, 0.75]} />
        <meshStandardMaterial color="#7A5C3A" roughness={0.4} />
      </mesh>
      <mesh position={[-dw / 2 + 0.04, 0.37, 0]} castShadow>
        <boxGeometry args={[0.06, 0.74, 0.65]} />
        <meshStandardMaterial color="#7A5C3A" roughness={0.4} />
      </mesh>
      <mesh position={[dw / 2 - 0.04, 0.37, 0]} castShadow>
        <boxGeometry args={[0.06, 0.74, 0.65]} />
        <meshStandardMaterial color="#7A5C3A" roughness={0.4} />
      </mesh>
      {/* Monitor */}
      <mesh position={[0, 1.1, -0.18]} castShadow>
        <boxGeometry args={[0.65, 0.38, 0.02]} />
        <meshStandardMaterial color="#1A1A1A" roughness={0.4} />
      </mesh>
    </group>
  )
}

// ─── Room interior scene ──────────────────────────────────────────────────────

function RoomScene({ room, wallH }: { room: Room; wallH: number }) {
  const rw = (room.width * 0.3048)   // feet → meters approx for scale
  const rd = (room.height * 0.3048)
  const mat = getMat(room.type)
  const t = room.type.toLowerCase().replace(/-/g, '_')

  const hasWindow = !t.includes('garage') && !t.includes('closet') && !t.includes('laundry')

  // Ceiling lights ref for animation
  const lightRef = useRef<THREE.PointLight>(null)

  return (
    <group>
      {/* Floor */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} receiveShadow>
        <planeGeometry args={[rw, rd]} />
        <meshStandardMaterial color={mat.floor} roughness={mat.floorRoughness} metalness={0.02} />
      </mesh>

      {/* Ceiling */}
      <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, wallH, 0]}>
        <planeGeometry args={[rw, rd]} />
        <meshStandardMaterial color={mat.ceiling} roughness={0.95} />
      </mesh>

      {/* Back wall */}
      <mesh position={[0, wallH / 2, -rd / 2]} receiveShadow>
        <planeGeometry args={[rw, wallH]} />
        <meshStandardMaterial color={mat.walls} roughness={mat.wallRoughness} />
      </mesh>

      {/* Front wall */}
      <mesh position={[0, wallH / 2, rd / 2]} rotation={[0, Math.PI, 0]}>
        <planeGeometry args={[rw, wallH]} />
        <meshStandardMaterial color={mat.walls} roughness={mat.wallRoughness} />
      </mesh>

      {/* Left wall */}
      <mesh position={[-rw / 2, wallH / 2, 0]} rotation={[0, Math.PI / 2, 0]} receiveShadow>
        <planeGeometry args={[rd, wallH]} />
        <meshStandardMaterial color={mat.walls} roughness={mat.wallRoughness} />
      </mesh>

      {/* Right wall */}
      <mesh position={[rw / 2, wallH / 2, 0]} rotation={[0, -Math.PI / 2, 0]} receiveShadow>
        <planeGeometry args={[rd, wallH]} />
        <meshStandardMaterial color={mat.walls} roughness={mat.wallRoughness} />
      </mesh>

      {/* Baseboard trim */}
      {[
        { pos: [0, 0.05, -rd / 2 + 0.025] as [number,number,number], args: [rw, 0.1, 0.05] as [number,number,number] },
        { pos: [0, 0.05, rd / 2 - 0.025] as [number,number,number],  args: [rw, 0.1, 0.05] as [number,number,number] },
        { pos: [-rw / 2 + 0.025, 0.05, 0] as [number,number,number], args: [0.05, 0.1, rd] as [number,number,number] },
        { pos: [rw / 2 - 0.025, 0.05, 0] as [number,number,number],  args: [0.05, 0.1, rd] as [number,number,number] },
      ].map((b, i) => (
        <mesh key={i} position={b.pos} castShadow>
          <boxGeometry args={b.args} />
          <meshStandardMaterial color={mat.trim} roughness={0.82} />
        </mesh>
      ))}

      {/* Window on back wall */}
      {hasWindow && (
        <group position={[0, wallH * 0.55, -rd / 2 + 0.005]}>
          {/* Frame */}
          <mesh>
            <boxGeometry args={[rw * 0.32, wallH * 0.32, 0.02]} />
            <meshStandardMaterial color={mat.trim} roughness={0.8} />
          </mesh>
          {/* Glass */}
          <mesh position={[0, 0, 0.005]}>
            <boxGeometry args={[rw * 0.28, wallH * 0.28, 0.01]} />
            <meshStandardMaterial color="#90C8E0" roughness={0.02} metalness={0.4} transparent opacity={0.4} />
          </mesh>
          {/* Window light shaft */}
          <pointLight position={[0, 0, 1]} intensity={0.8} color="#FFF8F0" distance={5} />
        </group>
      )}

      {/* Ceiling light */}
      <pointLight ref={lightRef} position={[0, wallH - 0.1, 0]} intensity={1.2} color="#FFF5E8" distance={rw * 3} castShadow />
      <mesh position={[0, wallH - 0.04, 0]}>
        <cylinderGeometry args={[0.12, 0.14, 0.06, 16]} />
        <meshStandardMaterial color="#E8E4DC" roughness={0.7} emissive="#FFEECC" emissiveIntensity={0.3} />
      </mesh>

      {/* Furniture based on room type */}
      {(t.includes('living') || t.includes('great')) && <Sofa rw={rw} rd={rd} />}
      {(t.includes('master') || t.includes('bedroom')) && (
        <Bed rw={rw} rd={rd} isMaster={t.includes('master')} />
      )}
      {t.includes('kitchen') && <KitchenCounters rw={rw} rd={rd} />}
      {(t.includes('bathroom') || t.includes('bath')) && (
        <>
          <Toilet rw={rw} rd={rd} />
          <BathroomVanity rw={rw} />
        </>
      )}
      {t.includes('office') && <Desk rw={rw} rd={rd} />}
    </group>
  )
}

// ─── Main exported component ──────────────────────────────────────────────────

interface Props {
  room: Room
  ceilingHeight: number
  onClose: () => void
}

export default function RoomInteriorView({ room, ceilingHeight, onClose }: Props) {
  const wallH = ceilingHeight * 0.3048
  const rw = room.width * 0.3048
  const rd = room.height * 0.3048

  const camX = 0
  const camY = wallH * 0.42
  const camZ = rd * 0.42

  return (
    <div className="room-interior-overlay">
      <div className="room-interior-header">
        <div className="room-interior-title">
          <div className="room-interior-swatch" style={{ background: room.color }} />
          <span>{room.name}</span>
          <span className="room-interior-dims">
            {room.width.toFixed(0)}' × {room.height.toFixed(0)}' · {Math.round(room.width * room.height)} sq ft · {ceilingHeight}ft ceiling
          </span>
        </div>
        <button className="room-interior-close" onClick={onClose}>✕ Close</button>
      </div>

      <div className="room-interior-canvas">
        <Canvas shadows gl={{ antialias: true }}>
          <PerspectiveCamera makeDefault position={[camX, camY, camZ]} fov={72} />
          <color attach="background" args={['#F0ECE4']} />

          <ambientLight intensity={0.5} color="#FFF8F0" />
          <directionalLight position={[rw * 0.5, wallH * 1.5, rd]} intensity={0.6} color="#FFF4E0" castShadow />

          <RoomScene room={room} wallH={wallH} />

          <OrbitControls
            target={[0, wallH * 0.4, 0]}
            minDistance={0.5}
            maxDistance={Math.max(rw, rd) * 1.2}
            maxPolarAngle={Math.PI / 2 + 0.1}
            enablePan={false}
          />
        </Canvas>
      </div>

      <div className="room-interior-info">
        <div className="ri-stat"><span>Width</span><strong>{room.width.toFixed(0)}'</strong></div>
        <div className="ri-stat"><span>Depth</span><strong>{room.height.toFixed(0)}'</strong></div>
        <div className="ri-stat"><span>Area</span><strong>{Math.round(room.width * room.height)} sq ft</strong></div>
        <div className="ri-stat"><span>Ceiling</span><strong>{ceilingHeight} ft</strong></div>
        <div className="ri-stat"><span>Perimeter</span><strong>{Math.round(2 * (room.width + room.height))} ft</strong></div>
      </div>
    </div>
  )
}
