import { useState, useRef, useEffect, useMemo, useCallback } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Sky, PerspectiveCamera, Text, Html } from '@react-three/drei'
import * as THREE from 'three'
import { FloorPlan } from '../types/floorplan'

const S = 0.09   // 1 ft = 0.09 THREE units
type Mode = 'exterior' | 'dollhouse' | 'walkthrough' | 'topview'

const WALL_COLOR = '#EDE8E0'
const WT = 0.022  // wall thickness

// ─────────────────────────────────────────────────────────────────────────────
// SHARED UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

function roomCenter(room: FloorPlan['rooms'][number]): [number, number, number] {
  return [(room.x + room.width / 2) * S, 0, (room.y + room.height / 2) * S]
}

/** Detect door openings: shared wall segments between rooms */
function findDoors(rooms: FloorPlan['rooms']): { x: number; z: number; isVertical: boolean; width: number }[] {
  const doors: { x: number; z: number; isVertical: boolean; width: number }[] = []
  const TOLERANCE = 1 // ft tolerance for shared walls

  for (let i = 0; i < rooms.length; i++) {
    for (let j = i + 1; j < rooms.length; j++) {
      const a = rooms[i], b = rooms[j]

      // Check horizontal adjacency (shared vertical wall)
      if (Math.abs((a.x + a.width) - b.x) <= TOLERANCE || Math.abs((b.x + b.width) - a.x) <= TOLERANCE) {
        const overlapStart = Math.max(a.y, b.y)
        const overlapEnd = Math.min(a.y + a.height, b.y + b.height)
        if (overlapEnd - overlapStart > 3) {
          const doorY = (overlapStart + overlapEnd) / 2
          const doorX = Math.abs((a.x + a.width) - b.x) <= TOLERANCE ? a.x + a.width : b.x + b.width
          doors.push({ x: doorX * S, z: doorY * S, isVertical: true, width: 3 * S })
        }
      }

      // Check vertical adjacency (shared horizontal wall)
      if (Math.abs((a.y + a.height) - b.y) <= TOLERANCE || Math.abs((b.y + b.height) - a.y) <= TOLERANCE) {
        const overlapStart = Math.max(a.x, b.x)
        const overlapEnd = Math.min(a.x + a.width, b.x + b.width)
        if (overlapEnd - overlapStart > 3) {
          const doorX = (overlapStart + overlapEnd) / 2
          const doorZ = Math.abs((a.y + a.height) - b.y) <= TOLERANCE ? a.y + a.height : b.y + b.height
          doors.push({ x: doorX * S, z: doorZ * S, isVertical: false, width: 3 * S })
        }
      }
    }
  }
  return doors
}

// ─────────────────────────────────────────────────────────────────────────────
// EXTERIOR SCENE (existing, refined)
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
      <mesh position={[cx, wallH, -overhang]} castShadow>
        <extrudeGeometry args={[shape, { depth: td + overhang * 2, bevelEnabled: false }]} />
        <meshStandardMaterial color="#7A5535" roughness={0.88} />
      </mesh>
      <mesh position={[cx, wallH + peakH - 0.035, plan.totalHeight * S / 2]} castShadow>
        <boxGeometry args={[0.055, 0.055, td + overhang * 2 + 0.1]} />
        <meshStandardMaterial color="#5C3D20" roughness={0.92} />
      </mesh>
    </group>
  )
}

function ExteriorDetails({ plan, wallH }: { plan: FloorPlan; wallH: number }) {
  const tw = plan.totalWidth * S
  const cx = plan.totalWidth * S / 2
  const winCount = Math.min(5, Math.max(2, Math.floor(plan.rooms.length * 0.6)))
  const winPositions = Array.from({ length: winCount }, (_, i) => (i + 1) / (winCount + 1))

  return (
    <group>
      <mesh position={[cx, 0.03, plan.totalHeight * S / 2]} receiveShadow>
        <boxGeometry args={[plan.totalWidth * S + 0.32, 0.06, plan.totalHeight * S + 0.32]} />
        <meshStandardMaterial color="#C2BBB2" roughness={0.9} />
      </mesh>
      {winPositions.map((xf, i) => {
        const wx = cx + (xf - 0.5) * tw * 0.82
        return (
          <group key={i} position={[wx, wallH * 0.6, 0.006]}>
            <mesh>
              <boxGeometry args={[tw * 0.115, wallH * 0.28, 0.012]} />
              <meshStandardMaterial color="#DDD8CE" roughness={0.82} />
            </mesh>
            <mesh>
              <boxGeometry args={[tw * 0.1, wallH * 0.25, 0.015]} />
              <meshStandardMaterial color="#90BBD8" roughness={0.05} metalness={0.55} transparent opacity={0.72} />
            </mesh>
          </group>
        )
      })}
      <group position={[cx, wallH * 0.29, 0.006]}>
        <mesh>
          <boxGeometry args={[tw * 0.07, wallH * 0.52, 0.016]} />
          <meshStandardMaterial color="#7A5535" roughness={0.72} />
        </mesh>
        <mesh position={[tw * 0.024, -wallH * 0.04, 0.012]}>
          <sphereGeometry args={[0.012, 8, 8]} />
          <meshStandardMaterial color="#C8A850" metalness={0.85} roughness={0.2} />
        </mesh>
      </group>
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
      <mesh position={[plan.totalWidth * S * 0.18, 0.008, -0.32]} receiveShadow>
        <boxGeometry args={[plan.totalWidth * S * 0.28, 0.016, 0.64]} />
        <meshStandardMaterial color="#AEA79E" roughness={0.94} />
      </mesh>
    </>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// DOLLHOUSE SCENE (existing)
// ─────────────────────────────────────────────────────────────────────────────

function RoomDollhouse({ room, wallH }: { room: FloorPlan['rooms'][number]; wallH: number }) {
  const rw = room.width * S
  const rd = room.height * S
  const px = (room.x + room.width / 2) * S
  const pz = (room.y + room.height / 2) * S

  return (
    <group position={[px, 0, pz]}>
      <mesh position={[0, WT / 2, 0]} receiveShadow>
        <boxGeometry args={[rw, WT, rd]} />
        <meshStandardMaterial color={room.color} roughness={0.88} />
      </mesh>
      <mesh position={[0, wallH / 2, -rd / 2 + WT / 2]} castShadow>
        <boxGeometry args={[rw, wallH, WT]} />
        <meshStandardMaterial color={WALL_COLOR} roughness={0.78} />
      </mesh>
      <mesh position={[0, wallH / 2, rd / 2 - WT / 2]} castShadow>
        <boxGeometry args={[rw, wallH, WT]} />
        <meshStandardMaterial color={WALL_COLOR} roughness={0.78} />
      </mesh>
      <mesh position={[-rw / 2 + WT / 2, wallH / 2, 0]} castShadow>
        <boxGeometry args={[WT, wallH, rd]} />
        <meshStandardMaterial color={WALL_COLOR} roughness={0.78} />
      </mesh>
      <mesh position={[rw / 2 - WT / 2, wallH / 2, 0]} castShadow>
        <boxGeometry args={[WT, wallH, rd]} />
        <meshStandardMaterial color={WALL_COLOR} roughness={0.78} />
      </mesh>
    </group>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// WALKTHROUGH MODE — First-person WASD walking
// ─────────────────────────────────────────────────────────────────────────────

function WalkthroughRoom({ room, wallH }: { room: FloorPlan['rooms'][number]; wallH: number }) {
  const rw = room.width * S
  const rd = room.height * S
  const px = (room.x + room.width / 2) * S
  const pz = (room.y + room.height / 2) * S

  return (
    <group position={[px, 0, pz]}>
      {/* Floor */}
      <mesh position={[0, 0.005, 0]} receiveShadow>
        <boxGeometry args={[rw, 0.01, rd]} />
        <meshStandardMaterial color={room.color} roughness={0.85} />
      </mesh>
      {/* Walls — slightly transparent for visibility */}
      <mesh position={[0, wallH / 2, -rd / 2 + WT / 2]} castShadow>
        <boxGeometry args={[rw, wallH, WT]} />
        <meshStandardMaterial color="#F5F0E8" roughness={0.7} />
      </mesh>
      <mesh position={[0, wallH / 2, rd / 2 - WT / 2]} castShadow>
        <boxGeometry args={[rw, wallH, WT]} />
        <meshStandardMaterial color="#F5F0E8" roughness={0.7} />
      </mesh>
      <mesh position={[-rw / 2 + WT / 2, wallH / 2, 0]} castShadow>
        <boxGeometry args={[WT, wallH, rd]} />
        <meshStandardMaterial color="#F5F0E8" roughness={0.7} />
      </mesh>
      <mesh position={[rw / 2 - WT / 2, wallH / 2, 0]} castShadow>
        <boxGeometry args={[WT, wallH, rd]} />
        <meshStandardMaterial color="#F5F0E8" roughness={0.7} />
      </mesh>
      {/* Room label on north wall */}
      <Text
        position={[0, wallH * 0.6, -rd / 2 + WT + 0.01]}
        fontSize={wallH * 0.12}
        color="#6B5E51"
        anchorX="center"
        anchorY="middle"
      >
        {room.name}
      </Text>
      {/* Baseboard trim */}
      <mesh position={[0, 0.015, -rd / 2 + WT / 2]}>
        <boxGeometry args={[rw, 0.03, WT + 0.005]} />
        <meshStandardMaterial color="#D8D2C8" roughness={0.8} />
      </mesh>
      <mesh position={[0, 0.015, rd / 2 - WT / 2]}>
        <boxGeometry args={[rw, 0.03, WT + 0.005]} />
        <meshStandardMaterial color="#D8D2C8" roughness={0.8} />
      </mesh>
    </group>
  )
}

/** Door opening markers for the walkthrough */
function DoorOpenings({ doors, wallH }: { doors: ReturnType<typeof findDoors>; wallH: number }) {
  return (
    <>
      {doors.map((door, i) => (
        <group key={i} position={[door.x, 0, door.z]}>
          {/* Door frame */}
          <mesh position={[0, wallH * 0.38, 0]}>
            <boxGeometry args={[
              door.isVertical ? 0.04 : door.width,
              wallH * 0.76,
              door.isVertical ? door.width : 0.04
            ]} />
            <meshStandardMaterial color="#8B7355" roughness={0.7} />
          </mesh>
          {/* Door threshold */}
          <mesh position={[0, 0.01, 0]}>
            <boxGeometry args={[
              door.isVertical ? 0.06 : door.width + 0.02,
              0.02,
              door.isVertical ? door.width + 0.02 : 0.06
            ]} />
            <meshStandardMaterial color="#A89070" roughness={0.85} />
          </mesh>
        </group>
      ))}
    </>
  )
}

/** First-person camera controller with WASD movement */
function FirstPersonController({ plan, wallH }: { plan: FloorPlan; wallH: number }) {
  const { camera, gl } = useThree()
  const moveState = useRef({ forward: false, backward: false, left: false, right: false })
  const yaw = useRef(0)
  const pitch = useRef(0)
  const isLocked = useRef(false)
  const speed = 0.025

  // Start in the center of the first room
  const startRoom = plan.rooms[0]
  const startPos = useRef(new THREE.Vector3(
    (startRoom.x + startRoom.width / 2) * S,
    wallH * 0.62,  // eye level ~5.5ft
    (startRoom.y + startRoom.height / 2) * S
  ))

  // Set initial camera position
  useEffect(() => {
    camera.position.copy(startPos.current)
    camera.rotation.set(0, 0, 0)
  }, [camera])

  // Pointer lock
  useEffect(() => {
    const canvas = gl.domElement

    const onClick = () => {
      canvas.requestPointerLock()
    }

    const onLockChange = () => {
      isLocked.current = document.pointerLockElement === canvas
    }

    const onMouseMove = (e: MouseEvent) => {
      if (!isLocked.current) return
      yaw.current -= e.movementX * 0.002
      pitch.current -= e.movementY * 0.002
      pitch.current = Math.max(-Math.PI / 3, Math.min(Math.PI / 3, pitch.current))
    }

    const onKeyDown = (e: KeyboardEvent) => {
      switch (e.code) {
        case 'KeyW': case 'ArrowUp': moveState.current.forward = true; break
        case 'KeyS': case 'ArrowDown': moveState.current.backward = true; break
        case 'KeyA': case 'ArrowLeft': moveState.current.left = true; break
        case 'KeyD': case 'ArrowRight': moveState.current.right = true; break
      }
    }

    const onKeyUp = (e: KeyboardEvent) => {
      switch (e.code) {
        case 'KeyW': case 'ArrowUp': moveState.current.forward = false; break
        case 'KeyS': case 'ArrowDown': moveState.current.backward = false; break
        case 'KeyA': case 'ArrowLeft': moveState.current.left = false; break
        case 'KeyD': case 'ArrowRight': moveState.current.right = false; break
      }
    }

    canvas.addEventListener('click', onClick)
    document.addEventListener('pointerlockchange', onLockChange)
    document.addEventListener('mousemove', onMouseMove)
    document.addEventListener('keydown', onKeyDown)
    document.addEventListener('keyup', onKeyUp)

    return () => {
      canvas.removeEventListener('click', onClick)
      document.removeEventListener('pointerlockchange', onLockChange)
      document.removeEventListener('mousemove', onMouseMove)
      document.removeEventListener('keydown', onKeyDown)
      document.removeEventListener('keyup', onKeyUp)
      if (document.pointerLockElement === canvas) {
        document.exitPointerLock()
      }
    }
  }, [gl])

  // Movement + camera rotation per frame
  useFrame(() => {
    // Apply yaw/pitch to camera rotation
    const euler = new THREE.Euler(pitch.current, yaw.current, 0, 'YXZ')
    camera.quaternion.setFromEuler(euler)

    // Calculate movement direction
    const direction = new THREE.Vector3()
    const m = moveState.current

    if (m.forward) direction.z -= 1
    if (m.backward) direction.z += 1
    if (m.left) direction.x -= 1
    if (m.right) direction.x += 1

    if (direction.lengthSq() > 0) {
      direction.normalize().multiplyScalar(speed)
      direction.applyAxisAngle(new THREE.Vector3(0, 1, 0), yaw.current)

      const newPos = camera.position.clone().add(direction)

      // Simple boundary check: keep within the plan bounds
      const margin = 0.1
      const minX = -margin
      const maxX = plan.totalWidth * S + margin
      const minZ = -margin
      const maxZ = plan.totalHeight * S + margin

      newPos.x = Math.max(minX, Math.min(maxX, newPos.x))
      newPos.z = Math.max(minZ, Math.min(maxZ, newPos.z))
      newPos.y = wallH * 0.62 // Keep at eye level

      camera.position.copy(newPos)
    }
  })

  return null
}

// ─────────────────────────────────────────────────────────────────────────────
// TOP VIEW MODE — Bird's-eye colored blocks with labels
// ─────────────────────────────────────────────────────────────────────────────

const ZONE_COLORS: Record<string, string> = {
  living_room: '#E8D5B7', kitchen: '#B7D5E8', dining_room: '#D5E8B7',
  family_room: '#E8E0B7', master_bedroom: '#D8B7E8', bedroom: '#C8B7E8',
  ensuite_bathroom: '#B7E8E0', bathroom: '#B7E8D5', half_bath: '#D0E8E8',
  hallway: '#E0E0D3', foyer: '#EEEAE0', home_office: '#F5F0D3',
  laundry_room: '#D3F5F5', garage: '#D5D5CC', walk_in_closet: '#E8D8E8',
  closet: '#E0D8E0', pantry: '#EDE8DC', mudroom: '#E8E4D8',
  utility_room: '#E8E8D3', patio: '#C8E8C8', deck: '#E8E4D0',
}

function TopViewBlock({ room, blockH }: { room: FloorPlan['rooms'][number]; blockH: number }) {
  const rw = room.width * S
  const rd = room.height * S
  const px = (room.x + room.width / 2) * S
  const pz = (room.y + room.height / 2) * S
  const color = ZONE_COLORS[room.type] || room.color || '#E0E0E0'

  return (
    <group position={[px, 0, pz]}>
      {/* 3D block */}
      <mesh position={[0, blockH / 2, 0]} castShadow receiveShadow>
        <boxGeometry args={[rw - 0.01, blockH, rd - 0.01]} />
        <meshStandardMaterial color={color} roughness={0.75} metalness={0.05} />
      </mesh>
      {/* Top edge highlight */}
      <mesh position={[0, blockH, 0]}>
        <boxGeometry args={[rw, 0.004, rd]} />
        <meshStandardMaterial color="#FFFFFF" roughness={0.5} transparent opacity={0.3} />
      </mesh>
      {/* Block outline (wireframe) */}
      <lineSegments position={[0, blockH / 2, 0]}>
        <edgesGeometry args={[new THREE.BoxGeometry(rw - 0.005, blockH + 0.002, rd - 0.005)]} />
        <lineBasicMaterial color="#00000030" />
      </lineSegments>
      {/* Room label */}
      <Text
        position={[0, blockH + 0.03, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={Math.min(rw, rd) * 0.22}
        color="#3A3530"
        anchorX="center"
        anchorY="middle"
        maxWidth={rw * 0.9}
      >
        {room.name}
      </Text>
      {/* Size label below name */}
      <Text
        position={[0, blockH + 0.02, Math.min(rw, rd) * 0.15]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={Math.min(rw, rd) * 0.13}
        color="#8A8580"
        anchorX="center"
        anchorY="middle"
      >
        {`${room.width}'×${room.height}'`}
      </Text>
    </group>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// WALKTHROUGH HUD (overlays)
// ─────────────────────────────────────────────────────────────────────────────

function WalkthroughHUD({ plan }: { plan: FloorPlan }) {
  return (
    <>
      {/* Crosshair */}
      <div className="walkthrough-crosshair">+</div>
      {/* Controls hint */}
      <div className="walkthrough-hint">
        <div>🖱️ Click to look around</div>
        <div>⌨️ WASD or arrows to walk</div>
        <div>ESC to release cursor</div>
      </div>
      {/* Minimap */}
      <div className="walkthrough-minimap">
        <svg viewBox={`0 0 ${plan.totalWidth} ${plan.totalHeight}`} preserveAspectRatio="xMidYMid meet">
          {plan.rooms.map(room => (
            <rect
              key={room.id}
              x={room.x} y={room.y}
              width={room.width} height={room.height}
              fill={room.color} stroke="#333" strokeWidth="0.5"
            />
          ))}
        </svg>
      </div>
    </>
  )
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN COMPONENT
// ─────────────────────────────────────────────────────────────────────────────

interface Props {
  plan: FloorPlan
  initialMode?: Mode
}

export default function View3D({ plan, initialMode }: Props) {
  const [mode, setMode] = useState<Mode>(initialMode || 'exterior')

  const wallH = (plan.ceilingHeight ?? 9) * S
  const cx = plan.totalWidth * S / 2
  const cz = plan.totalHeight * S / 2

  const doors = useMemo(() => findDoors(plan.rooms), [plan.rooms])

  // Camera positions for each mode
  const extCam: [number, number, number] = [
    cx + plan.totalWidth * S * 0.65,
    wallH * 1.2,
    cz + plan.totalHeight * S * 2.2,
  ]
  const dhCam: [number, number, number] = [
    cx + plan.totalWidth * S * 0.55,
    wallH * 5.5,
    cz + plan.totalHeight * S * 1.4,
  ]
  const topCam: [number, number, number] = [
    cx, wallH * 8, cz + 0.01,
  ]

  const modeButtons: { id: Mode; label: string; icon: string }[] = [
    { id: 'exterior', label: 'Exterior', icon: '🏠' },
    { id: 'dollhouse', label: 'Dollhouse', icon: '🏘️' },
    { id: 'walkthrough', label: 'Walk', icon: '🚶' },
    { id: 'topview', label: 'Top View', icon: '⬜' },
  ]

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>

      {/* ── Mode toggle overlay ── */}
      <div className="view3d-mode-toggle">
        {modeButtons.map(btn => (
          <button
            key={btn.id}
            className={mode === btn.id ? 'active' : ''}
            onClick={() => setMode(btn.id)}
          >
            <span>{btn.icon}</span> {btn.label}
          </button>
        ))}
      </div>

      {/* ── Walkthrough HUD ── */}
      {mode === 'walkthrough' && <WalkthroughHUD plan={plan} />}

      <Canvas
        key={mode}
        shadows
        style={{ width: '100%', height: '100%', cursor: mode === 'walkthrough' ? 'crosshair' : 'grab' }}
        gl={{ antialias: true }}
      >
        {/* ── EXTERIOR MODE ── */}
        {mode === 'exterior' && (
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
        )}

        {/* ── DOLLHOUSE MODE ── */}
        {mode === 'dollhouse' && (
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
            <directionalLight position={[cx, wallH * 0.5, cz + plan.totalHeight * S * 3]} intensity={0.4} color="#FFFFFF" />

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

        {/* ── WALKTHROUGH MODE ── */}
        {mode === 'walkthrough' && (
          <>
            <color attach="background" args={['#E8E4DC']} />

            <ambientLight intensity={0.5} color="#FFF8F0" />
            <directionalLight
              position={[cx, wallH * 4, cz]}
              intensity={1.2} color="#FFF8E8"
              castShadow
              shadow-mapSize-width={2048} shadow-mapSize-height={2048}
              shadow-camera-left={-20} shadow-camera-right={20}
              shadow-camera-top={20} shadow-camera-bottom={-20}
            />
            {/* Fill lights from corners */}
            <pointLight position={[0, wallH * 0.7, 0]} intensity={0.3} color="#FFF0E0" />
            <pointLight position={[plan.totalWidth * S, wallH * 0.7, plan.totalHeight * S]} intensity={0.3} color="#FFF0E0" />

            {/* Base ground plane */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow position={[cx, -0.01, cz]}>
              <planeGeometry args={[plan.totalWidth * S * 3, plan.totalHeight * S * 3]} />
              <meshStandardMaterial color="#C5BFB5" roughness={0.95} />
            </mesh>

            {/* Rooms */}
            {plan.rooms.map(room => (
              <WalkthroughRoom key={room.id} room={room} wallH={wallH} />
            ))}

            {/* Door openings */}
            <DoorOpenings doors={doors} wallH={wallH} />

            <FirstPersonController plan={plan} wallH={wallH} />
          </>
        )}

        {/* ── TOP VIEW MODE ── */}
        {mode === 'topview' && (
          <>
            <PerspectiveCamera makeDefault position={topCam} fov={50} />
            <color attach="background" args={['#F5F3EE']} />

            <ambientLight intensity={1.0} color="#FFFFFF" />
            <directionalLight
              position={[cx + 2, wallH * 10, cz - 2]}
              intensity={0.8} color="#FFF8F0"
              castShadow
              shadow-mapSize-width={2048} shadow-mapSize-height={2048}
              shadow-camera-left={-20} shadow-camera-right={20}
              shadow-camera-top={20} shadow-camera-bottom={-20}
            />
            <directionalLight position={[cx - 3, wallH * 5, cz + 3]} intensity={0.3} color="#E0E8FF" />

            {/* Subtle ground */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow position={[cx, -0.01, cz]}>
              <planeGeometry args={[plan.totalWidth * S * 3, plan.totalHeight * S * 3]} />
              <meshStandardMaterial color="#EAE7E0" roughness={0.96} />
            </mesh>

            {/* Room blocks */}
            {plan.rooms.map(room => (
              <TopViewBlock key={room.id} room={room} blockH={wallH * 0.5} />
            ))}

            <OrbitControls
              target={[cx, 0, cz]}
              minDistance={0.5} maxDistance={25}
              maxPolarAngle={Math.PI / 4}
              minPolarAngle={0}
              enableRotate={true}
            />
          </>
        )}
      </Canvas>
    </div>
  )
}
