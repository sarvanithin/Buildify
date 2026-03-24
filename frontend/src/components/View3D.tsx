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
// EXTERIOR SCENE — Clean White Architectural Line-Art Style (Drafted.ai)
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

  const WHITE = '#F8F7F4'
  const EDGE = '#D8D4CC'

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
            <meshStandardMaterial color="#E8E4DC" roughness={0.92} />
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

/** Clean white hip-style roof matching Drafted.ai */
function ArchitecturalRoof({ plan, wallH }: { plan: FloorPlan; wallH: number }) {
  const rooms = plan.rooms.filter(r =>
    r.type !== 'patio' && r.type !== 'deck' && r.type !== 'rear_patio' &&
    r.type !== 'outdoor_living' && r.type !== 'garage'
  )

  const WHITE = '#F2F0EC'
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
              <meshStandardMaterial color="#E0DDD6" roughness={0.9} />
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

  const FRAME = '#E4E0D8'
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
        <meshStandardMaterial color="#D0CCC4" roughness={0.88} />
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
          <meshStandardMaterial color="#E8E4DC" roughness={0.72} />
        </mesh>
        {/* Door glass lite */}
        <mesh position={[0, wallH * 0.08, 0.012]}>
          <boxGeometry args={[tw * 0.042, wallH * 0.14, 0.008]} />
          <meshStandardMaterial color={GLASS} roughness={0.1} transparent opacity={0.55} />
        </mesh>
        {/* Simple stoop */}
        <mesh position={[0, -wallH * 0.28, -0.08]}>
          <boxGeometry args={[tw * 0.12, 0.02, 0.18]} />
          <meshStandardMaterial color="#D4D0C8" roughness={0.88} />
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
