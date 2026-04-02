import { useState, useRef, useEffect, useMemo, useCallback } from 'react'
import { Canvas, useFrame, useThree } from '@react-three/fiber'
import { OrbitControls, Sky, PerspectiveCamera, Text, Html } from '@react-three/drei'
import * as THREE from 'three'
import { FloorPlan } from '../types/floorplan'

const S = 0.09   // 1 ft = 0.09 THREE units
type Mode = 'exterior' | 'dollhouse' | 'walkthrough' | 'topview'

const WALL_COLOR = '#e2e6ec'
const WT = 0.022  // wall thickness

// ─────────────────────────────────────────────────────────────────────────────
// SHARED UTILITIES
// ─────────────────────────────────────────────────────────────────────────────

function roomCenter(room: FloorPlan['rooms'][number]): [number, number, number] {
  return [(room.x + room.width / 2) * S, 0, (room.y + room.height / 2) * S]
}

// Removed frontend findDoors in favor of backend plan.doors data

// ─────────────────────────────────────────────────────────────────────────────
// EXTERIOR SCENE — Presentation model: light massing, soft edges
// ─────────────────────────────────────────────────────────────────────────────

/** White architectural walls — unified house mass from room bounding box */
function ArchitecturalHouse({ plan, wallH }: { plan: FloorPlan; wallH: number }) {
  // Compute overall bounding box for the main house body
  const rooms = plan.rooms.filter(r =>
    r.type !== 'patio' && r.type !== 'deck' && r.type !== 'rear_patio' && r.type !== 'outdoor_living'
  )
  const outdoor = plan.rooms.filter(r =>
    r.type === 'patio' || r.type === 'deck' || r.type === 'rear_patio' || r.type === 'outdoor_living'
  )

  const WHITE = '#f1f4f8'
  const EDGE = '#c5cdd8'

  return (
    <>
      {/* 🏠 Main house volume — one solid white box per room (they merge visually) */}
      {rooms.map(room => {
        const rw = room.width * S
        const rd = room.height * S
        const px = (room.x + room.width / 2) * S
        const pz = (room.y + room.height / 2) * S

        // Vary heights slightly by zone for realistic massing
        const isGarage = room.type === 'garage'
        const h = isGarage ? wallH * 0.88 : wallH

        return (
          <mesh key={room.id} position={[px, h / 2, pz]} castShadow receiveShadow>
            <boxGeometry args={[rw, h, rd]} />
            <meshStandardMaterial color={WHITE} roughness={0.82} metalness={0.0} />
          </mesh>
        )
      })}

      {/* 🌿 Outdoor / patio areas — flat slab, slightly different tone */}
      {outdoor.map(room => {
        const rw = room.width * S
        const rd = room.height * S
        const px = (room.x + room.width / 2) * S
        const pz = (room.y + room.height / 2) * S
        return (
          <mesh key={room.id} position={[px, 0.018, pz]} receiveShadow>
            <boxGeometry args={[rw, 0.036, rd]} />
            <meshStandardMaterial color="#d4dce8" roughness={0.92} />
          </mesh>
        )
      })}

      {/* Edge cap on top to create crisp roofline edge */}
      {rooms.map(room => {
        const rw = room.width * S
        const rd = room.height * S
        const px = (room.x + room.width / 2) * S
        const pz = (room.y + room.height / 2) * S
        const isGarage = room.type === 'garage'
        const h = isGarage ? wallH * 0.88 : wallH
        return (
          <mesh key={`cap-${room.id}`} position={[px, h - 0.005, pz]}>
            <boxGeometry args={[rw + 0.008, 0.012, rd + 0.008]} />
            <meshStandardMaterial color={EDGE} roughness={0.9} />
          </mesh>
        )
      })}
    </>
  )
}

/** Hip-style roof masses — neutral studio gray-white */
function ArchitecturalRoof({ plan, wallH }: { plan: FloorPlan; wallH: number }) {
  const rooms = plan.rooms.filter(r =>
    r.type !== 'patio' && r.type !== 'deck' && r.type !== 'rear_patio' &&
    r.type !== 'outdoor_living' && r.type !== 'garage'
  )

  const WHITE = '#e8ecf2'
  const overhang = 0.28

  // Group rooms into row-based sub-roofs for more realistic complex roofline
  const rowMap: Map<number, typeof rooms> = new Map()
  for (const room of rooms) {
    const key = Math.round(room.y / 2) * 2
    if (!rowMap.has(key)) rowMap.set(key, [])
    rowMap.get(key)!.push(room)
  }

  const roofSections: { mx: number; my: number; mw: number; mh: number }[] = []
  for (const rowRooms of rowMap.values()) {
    const minX = Math.min(...rowRooms.map(r => r.x))
    const maxX = Math.max(...rowRooms.map(r => r.x + r.width))
    const minY = Math.min(...rowRooms.map(r => r.y))
    const maxY = Math.max(...rowRooms.map(r => r.y + r.height))
    roofSections.push({ mx: minX, my: minY, mw: maxX - minX, mh: maxY - minY })
  }

  return (
    <>
      {roofSections.map((sec, idx) => {
        const tw = sec.mw * S
        const td = sec.mh * S
        const cx = (sec.mx + sec.mw / 2) * S
        const cz = (sec.my + sec.mh / 2) * S
        const peakH = Math.min(tw, td) * 0.38

        // Simple hip roof mesh via a slightly elevated pyramidal shape
        // Build as two crossed gable prisms
        return (
          <group key={idx} position={[cx, wallH, cz]}>
            {/* Main roof deck — flat with slope suggestion */}
            <mesh castShadow>
              <boxGeometry args={[tw + overhang * 2, 0.025, td + overhang * 2]} />
              <meshStandardMaterial color={WHITE} roughness={0.86} />
            </mesh>
            {/* Ridge beam along long axis */}
            <mesh position={[0, peakH / 2, 0]} castShadow>
              <boxGeometry args={[tw * 0.85, peakH, 0.04]} />
              <meshStandardMaterial color={WHITE} roughness={0.82} />
            </mesh>
            {/* Cross ridge */}
            <mesh position={[0, peakH / 2, 0]} castShadow>
              <boxGeometry args={[0.04, peakH, td * 0.85]} />
              <meshStandardMaterial color={WHITE} roughness={0.82} />
            </mesh>
            {/* Ridge cap */}
            <mesh position={[0, peakH, 0]}>
              <boxGeometry args={[tw * 0.5, 0.025, 0.06]} />
              <meshStandardMaterial color="#d0d6de" roughness={0.9} />
            </mesh>
          </group>
        )
      })}
    </>
  )
}

/** Windows and front door details — clean white frames */
function ArchitecturalDetails({ plan, wallH }: { plan: FloorPlan; wallH: number }) {
  const tw = plan.totalWidth * S
  const td = plan.totalHeight * S
  const cx = tw / 2

  const winCount = Math.min(5, Math.max(2, Math.floor(plan.rooms.length * 0.5)))
  const winPositions = Array.from({ length: winCount }, (_, i) => (i + 1) / (winCount + 1))

  const FRAME = '#c5ccd6'
  const GLASS = '#C8D8E8'

  return (
    <group>
      {/* Green lawn base */}
      <mesh position={[cx, -0.015, td / 2]} receiveShadow>
        <boxGeometry args={[tw + 2.4, 0.03, td + 2.4]} />
        <meshStandardMaterial color="#8FB87A" roughness={0.95} />
      </mesh>

      {/* Concrete foundation lip */}
      <mesh position={[cx, 0.016, td / 2]} receiveShadow>
        <boxGeometry args={[tw + 0.06, 0.032, td + 0.06]} />
        <meshStandardMaterial color="#b8c0c8" roughness={0.88} />
      </mesh>

      {/* Windows — front face (z = 0) */}
      {winPositions.map((xf, i) => {
        const wx = (xf - 0.5) * tw * 0.78 + cx
        return (
          <group key={i} position={[wx, wallH * 0.58, 0.01]}>
            {/* Frame */}
            <mesh>
              <boxGeometry args={[tw * 0.11, wallH * 0.30, 0.018]} />
              <meshStandardMaterial color={FRAME} roughness={0.7} />
            </mesh>
            {/* Glass pane */}
            <mesh>
              <boxGeometry args={[tw * 0.095, wallH * 0.265, 0.022]} />
              <meshStandardMaterial color={GLASS} roughness={0.08} metalness={0.2} transparent opacity={0.6} />
            </mesh>
            {/* Mullion */}
            <mesh position={[0, 0, 0.019]}>
              <boxGeometry args={[tw * 0.095, 0.008, 0.004]} />
              <meshStandardMaterial color={FRAME} roughness={0.8} />
            </mesh>
          </group>
        )
      })}

      {/* Front door */}
      <group position={[cx, wallH * 0.28, 0.01]}>
        <mesh>
          <boxGeometry args={[tw * 0.065, wallH * 0.52, 0.018]} />
          <meshStandardMaterial color="#d4dce6" roughness={0.72} />
        </mesh>
        {/* Door glass lite */}
        <mesh position={[0, wallH * 0.08, 0.012]}>
          <boxGeometry args={[tw * 0.042, wallH * 0.14, 0.008]} />
          <meshStandardMaterial color={GLASS} roughness={0.1} transparent opacity={0.55} />
        </mesh>
        {/* Simple stoop */}
        <mesh position={[0, -wallH * 0.28, -0.08]}>
          <boxGeometry args={[tw * 0.12, 0.02, 0.18]} />
          <meshStandardMaterial color="#b8c0cc" roughness={0.88} />
        </mesh>
      </group>
    </group>
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
      {/* Room label on floor */}
      <Text
        position={[0, WT + 0.005, 0]}
        rotation={[-Math.PI / 2, 0, 0]}
        fontSize={Math.min(rw, rd) * 0.18}
        color="#1a1a1a80" // Hex with alpha (50%)
        anchorX="center"
        anchorY="middle"
      >
        {room.name.toUpperCase()}
      </Text>
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
        <meshStandardMaterial color="#e8ecf2" roughness={0.7} />
      </mesh>
      <mesh position={[0, wallH / 2, rd / 2 - WT / 2]} castShadow>
        <boxGeometry args={[rw, wallH, WT]} />
        <meshStandardMaterial color="#e8ecf2" roughness={0.7} />
      </mesh>
      <mesh position={[-rw / 2 + WT / 2, wallH / 2, 0]} castShadow>
        <boxGeometry args={[WT, wallH, rd]} />
        <meshStandardMaterial color="#e8ecf2" roughness={0.7} />
      </mesh>
      <mesh position={[rw / 2 - WT / 2, wallH / 2, 0]} castShadow>
        <boxGeometry args={[WT, wallH, rd]} />
        <meshStandardMaterial color="#e8ecf2" roughness={0.7} />
      </mesh>
      {/* Room label on north wall */}
      <Text
        position={[0, wallH * 0.6, -rd / 2 + WT + 0.01]}
        fontSize={wallH * 0.12}
        color="#475569"
        anchorX="center"
        anchorY="middle"
      >
        {room.name}
      </Text>
      {/* Baseboard trim */}
      <mesh position={[0, 0.015, -rd / 2 + WT / 2]}>
        <boxGeometry args={[rw, 0.03, WT + 0.005]} />
        <meshStandardMaterial color="#c5cbd6" roughness={0.8} />
      </mesh>
      <mesh position={[0, 0.015, rd / 2 - WT / 2]}>
        <boxGeometry args={[rw, 0.03, WT + 0.005]} />
        <meshStandardMaterial color="#c5cbd6" roughness={0.8} />
      </mesh>
    </group>
  )
}

/** Door opening markers for the walkthrough */
function DoorOpenings({ doors, wallH }: { doors: FloorPlan['doors']; wallH: number }) {
  return (
    <>
      {doors.map((door, i) => {
        const dx = door.x * S
        const dz = door.y * S
        const dw = 3 * S // Standard 3ft door
        return (
          <group key={i} position={[dx, 0, dz]}>
            {/* Door frame */}
            <mesh position={[0, wallH * 0.38, 0]}>
              <boxGeometry args={[
                door.isVertical ? 0.04 : dw,
                wallH * 0.76,
                door.isVertical ? dw : 0.04
              ]} />
              <meshStandardMaterial color="#8B7355" roughness={0.7} />
            </mesh>
            {/* Door threshold */}
            <mesh position={[0, 0.01, 0]}>
              <boxGeometry args={[
                door.isVertical ? 0.06 : dw + 0.02,
                0.02,
                door.isVertical ? dw + 0.02 : 0.06
              ]} />
              <meshStandardMaterial color="#A89070" roughness={0.85} />
            </mesh>
          </group>
        )
      })}
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

  // Movement + collision detection
  useFrame(() => {
    const euler = new THREE.Euler(pitch.current, yaw.current, 0, 'YXZ')
    camera.quaternion.setFromEuler(euler)

    const direction = new THREE.Vector3()
    const m = moveState.current
    if (m.forward) direction.z -= 1
    if (m.backward) direction.z += 1
    if (m.left) direction.x -= 1
    if (m.right) direction.x += 1

    if (direction.lengthSq() > 0) {
      direction.normalize().multiplyScalar(speed)
      direction.applyAxisAngle(new THREE.Vector3(0, 1, 0), yaw.current)

      const nextX = camera.position.x + direction.x
      const nextZ = camera.position.z + direction.z

      // Collision helper: find if point is inside a room OR a door
      const isPassable = (tx: number, tz: number) => {
        const ftX = tx / S
        const ftY = tz / S
        
        // 1. Check if inside any room with a small 0.5ft wall margin
        const insideRoom = plan.rooms.some(r => 
          ftX >= r.x + 0.5 && ftX <= r.x + r.width - 0.5 &&
          ftY >= r.y + 0.5 && ftY <= r.y + r.height - 0.5
        )
        if (insideRoom) return true

        // 2. Check if inside a door opening
        const nearDoor = plan.doors.some(d => {
          const dx = d.x, dy = d.y
          const dSize = 1.5 // 3ft wide door = 1.5ft radius
          if (d.isVertical) {
            return Math.abs(ftX - dx) < 1.0 && Math.abs(ftY - dy) < dSize
          } else {
            return Math.abs(ftY - dy) < 1.0 && Math.abs(ftX - dx) < dSize
          }
        })
        return nearDoor
      }

      // Sliding collision (check X and Z separately)
      if (isPassable(nextX, camera.position.z)) camera.position.x = nextX
      if (isPassable(camera.position.x, nextZ)) camera.position.z = nextZ
      
      camera.position.y = wallH * 0.62
    }
  })

  return null
}

// ─────────────────────────────────────────────────────────────────────────────
// TOP VIEW MODE — Bird's-eye colored blocks with labels
// ─────────────────────────────────────────────────────────────────────────────

const ZONE_COLORS: Record<string, string> = {
  living_room: '#a8d0bc', kitchen: '#e2d9a8', dining_room: '#d4cfa0',
  family_room: '#b8dcc4', master_bedroom: '#e8d4dc', bedroom: '#c8d4e8',
  ensuite_bathroom: '#a8cfe8', bathroom: '#a8cfe8', half_bath: '#b8e0e4',
  hallway: '#d8dce4', foyer: '#d8e0c8', home_office: '#c4c8e8',
  laundry_room: '#b8d4e8', garage: '#c4c8c4', walk_in_closet: '#d8dce8',
  closet: '#d8dce8', pantry: '#e0dcc8', mudroom: '#d0d4cc',
  utility_room: '#ccd0d0', patio: '#a8d8bc', deck: '#a8d8bc',
}

function TopViewBlock({ room, blockH }: { room: FloorPlan['rooms'][number]; blockH: number }) {
  const rw = room.width * S
  const rd = room.height * S
  const px = (room.x + room.width / 2) * S
  const pz = (room.y + room.height / 2) * S
  const color = ZONE_COLORS[room.type] || room.color || '#dce0e8'

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
        color="#1e293b"
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
        color="#64748b"
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

  const doors = plan.doors || []

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

            <ambientLight intensity={0.65} color="#f4f6fb" />
            <directionalLight
              position={[10, 18, 9]} intensity={1.6} color="#eef2f9"
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

            <ArchitecturalHouse plan={plan} wallH={wallH} />
            <ArchitecturalDetails plan={plan} wallH={wallH} />
            <ArchitecturalRoof plan={plan} wallH={wallH} />

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
            <color attach="background" args={['#e8ecf2']} />

            <ambientLight intensity={1.3} color="#FFFFFF" />
            <directionalLight
              position={[cx, wallH * 7, cz + plan.totalHeight * S * 0.8]}
              intensity={0.7} color="#f1f5f9"
              castShadow
              shadow-mapSize-width={2048} shadow-mapSize-height={2048}
              shadow-camera-left={-20} shadow-camera-right={20}
              shadow-camera-top={20} shadow-camera-bottom={-20}
            />
            <directionalLight position={[cx, wallH * 0.5, cz + plan.totalHeight * S * 3]} intensity={0.4} color="#FFFFFF" />

            <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow position={[cx, 0, cz]}>
              <planeGeometry args={[plan.totalWidth * S * 2.5, plan.totalHeight * S * 2.5]} />
              <meshStandardMaterial color="#cbd5e1" roughness={0.96} />
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
            <color attach="background" args={['#e2e8f0']} />

            <ambientLight intensity={0.5} color="#f8fafc" />
            <directionalLight
              position={[cx, wallH * 4, cz]}
              intensity={1.2} color="#f1f5f9"
              castShadow
              shadow-mapSize-width={2048} shadow-mapSize-height={2048}
              shadow-camera-left={-20} shadow-camera-right={20}
              shadow-camera-top={20} shadow-camera-bottom={-20}
            />
            {/* Fill lights from corners */}
            <pointLight position={[0, wallH * 0.7, 0]} intensity={0.3} color="#e0e7ef" />
            <pointLight position={[plan.totalWidth * S, wallH * 0.7, plan.totalHeight * S]} intensity={0.3} color="#e0e7ef" />

            {/* Base ground plane */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow position={[cx, -0.01, cz]}>
              <planeGeometry args={[plan.totalWidth * S * 3, plan.totalHeight * S * 3]} />
              <meshStandardMaterial color="#94a3b8" roughness={0.95} />
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
            <color attach="background" args={['#e8eef5']} />

            <ambientLight intensity={1.0} color="#FFFFFF" />
            <directionalLight
              position={[cx + 2, wallH * 10, cz - 2]}
              intensity={0.8} color="#f8fafc"
              castShadow
              shadow-mapSize-width={2048} shadow-mapSize-height={2048}
              shadow-camera-left={-20} shadow-camera-right={20}
              shadow-camera-top={20} shadow-camera-bottom={-20}
            />
            <directionalLight position={[cx - 3, wallH * 5, cz + 3]} intensity={0.3} color="#E0E8FF" />

            {/* Subtle ground */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} receiveShadow position={[cx, -0.01, cz]}>
              <planeGeometry args={[plan.totalWidth * S * 3, plan.totalHeight * S * 3]} />
              <meshStandardMaterial color="#d8dee9" roughness={0.96} />
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
