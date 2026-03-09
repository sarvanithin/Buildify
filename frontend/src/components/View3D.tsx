import { useState, useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Sky, PerspectiveCamera } from '@react-three/drei'
import * as THREE from 'three'
import { FloorPlan } from '../types/floorplan'

const S = 0.09   // 1 ft = 0.09 THREE units (keeps scene at comfortable scale)

type Mode = 'exterior' | 'dollhouse'

// ─────────────────────────────────────────────────────────────────────────────
// EXTERIOR SCENE
// Renders each room as a cream-colored box — this forms the EXACT house
// footprint from the floor plan, not a generic bounding-box rectangle.
// ─────────────────────────────────────────────────────────────────────────────

function ExteriorWalls({ plan, wallH }: { plan: FloorPlan; wallH: number }) {
  return (
    <>
      {plan.rooms.map(room => {
        const rw = room.width * S
        const rd = room.height * S
        const px = (room.x + room.width / 2) * S
        const pz = (room.y + room.height / 2) * S
        return (
          <mesh key={room.id} position={[px, wallH / 2, pz]} castShadow receiveShadow>
            <boxGeometry args={[rw, wallH, rd]} />
            <meshStandardMaterial color="#EDE8DF" roughness={0.76} metalness={0.02} />
          </mesh>
        )
      })}
    </>
  )
}

function GableRoof({ plan, wallH }: { plan: FloorPlan; wallH: number }) {
  const tw = plan.totalWidth * S
  const td = plan.totalHeight * S
  const overhang = 0.3
  const peakH = Math.min(tw, td) * 0.3

  const shape = useMemo(() => {
    const s = new THREE.Shape()
    s.moveTo(-(tw / 2 + overhang), 0)
    s.lineTo(tw / 2 + overhang, 0)
    s.lineTo(0, peakH)
    s.closePath()
    return s
  }, [tw, overhang, peakH])

  const cx = plan.totalWidth * S / 2

  return (
    <group>
      {/* Main gable body */}
      <mesh position={[cx, wallH, -overhang]} castShadow>
        <extrudeGeometry args={[shape, { depth: td + overhang * 2, bevelEnabled: false }]} />
        <meshStandardMaterial color="#7A5535" roughness={0.88} />
      </mesh>
      {/* Ridge beam */}
      <mesh position={[cx, wallH + peakH - 0.035, plan.totalHeight * S / 2]} castShadow>
        <boxGeometry args={[0.055, 0.055, td + overhang * 2 + 0.1]} />
        <meshStandardMaterial color="#5C3D20" roughness={0.92} />
      </mesh>
      {/* Fascia boards (eave trim) */}
      <mesh position={[cx, wallH + 0.025, -overhang]}>
        <boxGeometry args={[tw + overhang * 2, 0.05, 0.04]} />
        <meshStandardMaterial color="#D4C8B8" roughness={0.85} />
      </mesh>
      <mesh position={[cx, wallH + 0.025, td + overhang]}>
        <boxGeometry args={[tw + overhang * 2, 0.05, 0.04]} />
        <meshStandardMaterial color="#D4C8B8" roughness={0.85} />
      </mesh>
    </group>
  )
}

function ExteriorDetails({ plan, wallH }: { plan: FloorPlan; wallH: number }) {
  const tw = plan.totalWidth * S
  const cx = plan.totalWidth * S / 2

  // Distribute windows across front face
  const winCount = Math.min(5, Math.max(2, Math.floor(plan.rooms.length * 0.6)))
  const winPositions = Array.from({ length: winCount }, (_, i) => (i + 1) / (winCount + 1))

  return (
    <group>
      {/* Concrete foundation */}
      <mesh position={[cx, 0.03, plan.totalHeight * S / 2]} receiveShadow>
        <boxGeometry args={[plan.totalWidth * S + 0.32, 0.06, plan.totalHeight * S + 0.32]} />
        <meshStandardMaterial color="#C2BBB2" roughness={0.9} />
      </mesh>

      {/* Front windows */}
      {winPositions.map((xf, i) => {
        const wx = cx + (xf - 0.5) * tw * 0.82
        return (
          <group key={i} position={[wx, wallH * 0.6, 0.006]}>
            {/* Frame */}
            <mesh>
              <boxGeometry args={[tw * 0.115, wallH * 0.28, 0.012]} />
              <meshStandardMaterial color="#DDD8CE" roughness={0.82} />
            </mesh>
            {/* Glass */}
            <mesh>
              <boxGeometry args={[tw * 0.1, wallH * 0.25, 0.015]} />
              <meshStandardMaterial color="#90BBD8" roughness={0.05} metalness={0.55} transparent opacity={0.72} />
            </mesh>
            {/* Mullion */}
            <mesh>
              <boxGeometry args={[0.006, wallH * 0.25, 0.018]} />
              <meshStandardMaterial color="#D8D2C8" roughness={0.8} />
            </mesh>
          </group>
        )
      })}

      {/* Front door */}
      <group position={[cx, wallH * 0.29, 0.006]}>
        <mesh>
          <boxGeometry args={[tw * 0.085, wallH * 0.54, 0.012]} />
          <meshStandardMaterial color="#DDD8CE" roughness={0.82} />
        </mesh>
        <mesh>
          <boxGeometry args={[tw * 0.07, wallH * 0.52, 0.016]} />
          <meshStandardMaterial color="#7A5535" roughness={0.72} />
        </mesh>
        {/* Door knob */}
        <mesh position={[tw * 0.024, -wallH * 0.04, 0.012]}>
          <sphereGeometry args={[0.012, 8, 8]} />
          <meshStandardMaterial color="#C8A850" metalness={0.85} roughness={0.2} />
        </mesh>
      </group>

      {/* Side windows (right face) */}
      {[0.3, 0.65].map((zf, i) => (
        <group key={`sw${i}`} position={[plan.totalWidth * S - 0.006, wallH * 0.58, plan.totalHeight * S * zf]}>
          <mesh>
            <boxGeometry args={[0.012, wallH * 0.25, plan.totalHeight * S * 0.1]} />
            <meshStandardMaterial color="#90BBD8" roughness={0.05} metalness={0.55} transparent opacity={0.72} />
          </mesh>
        </group>
      ))}

      {/* Chimney */}
      <mesh
        position={[plan.totalWidth * S * 0.7, wallH + plan.totalWidth * S * 0.3 * 0.55, plan.totalHeight * S * 0.35]}
        castShadow
      >
        <boxGeometry args={[0.14, plan.totalWidth * S * 0.3 * 0.7, 0.14]} />
        <meshStandardMaterial color="#9E7B5A" roughness={0.92} />
      </mesh>
    </group>
  )
}

function Trees({ plan }: { plan: FloorPlan }) {
  const positions: [number, number][] = [
    [-0.7, -0.65],
    [plan.totalWidth * S + 0.6, -0.5],
    [-0.65, plan.totalHeight * S + 0.5],
    [plan.totalWidth * S + 0.55, plan.totalHeight * S * 0.65],
  ]
  return (
    <>
      {positions.map(([tx, tz], i) => (
        <group key={i} position={[tx, 0, tz]}>
          <mesh position={[0, 0.14, 0]}>
            <cylinderGeometry args={[0.036, 0.055, 0.28, 7]} />
            <meshStandardMaterial color="#6B4226" roughness={0.9} />
          </mesh>
          <mesh position={[0, 0.48, 0]}>
            <coneGeometry args={[0.26, 0.65, 8]} />
            <meshStandardMaterial color="#2D6838" roughness={0.86} />
          </mesh>
        </group>
      ))}
      {/* Driveway */}
      <mesh position={[plan.totalWidth * S * 0.18, 0.008, -0.32]} receiveShadow>
        <boxGeometry args={[plan.totalWidth * S * 0.28, 0.016, 0.64]} />
        <meshStandardMaterial color="#AEA79E" roughness={0.94} />
      </mesh>
    </>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// DOLLHOUSE SCENE
// Each room rendered as colored floor slab + 4 cream walls with no ceiling,
// so you can see inside all rooms from the overhead camera.
// ─────────────────────────────────────────────────────────────────────────────

const WALL_COLOR = '#EDE8E0'
const WT = 0.022  // wall thickness in THREE units

function RoomDollhouse({ room, wallH }: { room: FloorPlan['rooms'][number]; wallH: number }) {
  const rw = room.width * S
  const rd = room.height * S
  const px = (room.x + room.width / 2) * S
  const pz = (room.y + room.height / 2) * S

  return (
    <group position={[px, 0, pz]}>
      {/* Colored floor */}
      <mesh position={[0, WT / 2, 0]} receiveShadow>
        <boxGeometry args={[rw, WT, rd]} />
        <meshStandardMaterial color={room.color} roughness={0.88} />
      </mesh>

      {/* North wall */}
      <mesh position={[0, wallH / 2, -rd / 2 + WT / 2]} castShadow>
        <boxGeometry args={[rw, wallH, WT]} />
        <meshStandardMaterial color={WALL_COLOR} roughness={0.78} />
      </mesh>
      {/* South wall */}
      <mesh position={[0, wallH / 2, rd / 2 - WT / 2]} castShadow>
        <boxGeometry args={[rw, wallH, WT]} />
        <meshStandardMaterial color={WALL_COLOR} roughness={0.78} />
      </mesh>
      {/* West wall */}
      <mesh position={[-rw / 2 + WT / 2, wallH / 2, 0]} castShadow>
        <boxGeometry args={[WT, wallH, rd]} />
        <meshStandardMaterial color={WALL_COLOR} roughness={0.78} />
      </mesh>
      {/* East wall */}
      <mesh position={[rw / 2 - WT / 2, wallH / 2, 0]} castShadow>
        <boxGeometry args={[WT, wallH, rd]} />
        <meshStandardMaterial color={WALL_COLOR} roughness={0.78} />
      </mesh>
    </group>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN COMPONENT
// ─────────────────────────────────────────────────────────────────────────────

interface Props {
  plan: FloorPlan
}

export default function View3D({ plan }: Props) {
  const [mode, setMode] = useState<Mode>('exterior')

  const wallH = (plan.ceilingHeight ?? 9) * S
  const cx = plan.totalWidth * S / 2
  const cz = plan.totalHeight * S / 2

  // Exterior camera: eye-level, front-right
  const extCam: [number, number, number] = [
    cx + plan.totalWidth * S * 0.65,
    wallH * 1.2,
    cz + plan.totalHeight * S * 2.2,
  ]

  // Dollhouse camera: high overhead, tilted
  const dhCam: [number, number, number] = [
    cx + plan.totalWidth * S * 0.55,
    wallH * 5.5,
    cz + plan.totalHeight * S * 1.4,
  ]

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>

      {/* ── Mode toggle overlay ── */}
      <div className="view3d-mode-toggle">
        <button className={mode === 'exterior' ? 'active' : ''} onClick={() => setMode('exterior')}>
          Exterior
        </button>
        <button className={mode === 'dollhouse' ? 'active' : ''} onClick={() => setMode('dollhouse')}>
          Dollhouse
        </button>
      </div>

      <Canvas
        key={mode}   /* remount on mode change → reset camera */
        shadows
        style={{ width: '100%', height: '100%' }}
        gl={{ antialias: true }}
      >
        {mode === 'exterior' ? (
          <>
            <PerspectiveCamera makeDefault position={extCam} fov={44} />
            <color attach="background" args={['#BDD8EE']} />
            <Sky sunPosition={[55, 16, 22]} turbidity={4.5} rayleigh={0.65} />

            <ambientLight intensity={0.65} color="#FFF5EC" />
            <directionalLight
              position={[10, 18, 9]} intensity={1.6} color="#FFF4E4"
              castShadow
              shadow-mapSize-width={2048} shadow-mapSize-height={2048}
              shadow-camera-left={-16} shadow-camera-right={16}
              shadow-camera-top={16} shadow-camera-bottom={-16}
              shadow-camera-far={60}
            />
            <directionalLight position={[-7, 7, -5]} intensity={0.28} color="#B8D4FF" />

            {/* Grass ground */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow position={[cx, 0, cz]}>
              <planeGeometry args={[plan.totalWidth * S * 7, plan.totalHeight * S * 7]} />
              <meshStandardMaterial color="#497850" roughness={0.93} />
            </mesh>

            <ExteriorWalls plan={plan} wallH={wallH} />
            <ExteriorDetails plan={plan} wallH={wallH} />
            <GableRoof plan={plan} wallH={wallH} />
            <Trees plan={plan} />

            <OrbitControls
              target={[cx, wallH * 0.44, cz]}
              minDistance={1} maxDistance={30}
              maxPolarAngle={Math.PI / 2 - 0.03}
            />
          </>
        ) : (
          <>
            <PerspectiveCamera makeDefault position={dhCam} fov={52} />
            <color attach="background" args={['#EDEAE4']} />

            <ambientLight intensity={1.3} color="#FFFFFF" />
            <directionalLight
              position={[cx, wallH * 7, cz + plan.totalHeight * S * 0.8]}
              intensity={0.7} color="#FFF8F2"
              castShadow
              shadow-mapSize-width={2048} shadow-mapSize-height={2048}
              shadow-camera-left={-20} shadow-camera-right={20}
              shadow-camera-top={20} shadow-camera-bottom={-20}
            />
            {/* Fill light from front-low so wall faces are visible */}
            <directionalLight position={[cx, wallH * 0.5, cz + plan.totalHeight * S * 3]} intensity={0.4} color="#FFFFFF" />

            {/* Floor base */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow position={[cx, 0, cz]}>
              <planeGeometry args={[plan.totalWidth * S * 2.5, plan.totalHeight * S * 2.5]} />
              <meshStandardMaterial color="#D8D3C8" roughness={0.96} />
            </mesh>

            {plan.rooms.map(room => (
              <RoomDollhouse key={room.id} room={room} wallH={wallH} />
            ))}

            <OrbitControls
              target={[cx, 0, cz]}
              minDistance={0.5} maxDistance={22}
              maxPolarAngle={Math.PI / 2 - 0.01}
            />
          </>
        )}
      </Canvas>
    </div>
  )
}
