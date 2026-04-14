import { useState, useRef, useEffect } from 'react'
import { FloorPlan, Room, Constraints } from '../types/floorplan'
import View3D from './View3D'
import RoomInteriorView from './RoomInteriorView'
import ElevationView from './ElevationView'
import SpecSchedule from './SpecSchedule'
import CostPanel from './CostPanel'
import DesignScore from './DesignScore'
import ChatPanel from './ChatPanel'
import { exportDxf, exportPdf } from '../api/client'
import ArchPlan from './ArchPlan'

interface Props {
  plan: FloorPlan
  constraints: Constraints
  onUpdate: (plan: FloorPlan, updatedConstraints?: Constraints) => void
}

type MainTab = 'plan' | 'elevations' | 'spec' | 'cost' | 'score'
type ViewMode = '2d' | 'exterior' | 'dollhouse' | 'walkthrough' | 'topview'

const TAB_LABELS: { id: MainTab; label: string }[] = [
  { id: 'plan', label: '2D / 3D Plan' },
  { id: 'elevations', label: 'Elevations' },
  { id: 'spec', label: 'Spec Schedule' },
  { id: 'cost', label: 'Cost Estimate' },
  { id: 'score', label: 'Design Score' },
]

const VIEW_BUTTONS: { id: ViewMode; label: string; icon: string }[] = [
  { id: '2d', label: '2D Plan', icon: '📐' },
  { id: 'exterior', label: 'Exterior', icon: '🏠' },
  { id: 'dollhouse', label: 'Dollhouse', icon: '🏘️' },
  { id: 'walkthrough', label: 'Walk', icon: '🚶' },
  { id: 'topview', label: 'Top View', icon: '⬜' },
]

export default function FloorPlanEditor({ plan, constraints, onUpdate }: Props) {
  const [tab, setTab] = useState<MainTab>('plan')
  const [viewMode, setViewMode] = useState<ViewMode>('2d')
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [activeFloor, setActiveFloor] = useState<number>(1)
  const [interiorRoom, setInteriorRoom] = useState<Room | null>(null)
  const [canvasSize, setCanvasSize] = useState({ width: 800, height: 600 })
  const [showSitePlan, setShowSitePlan] = useState(false)
  const [showStructGrid, setShowStructGrid] = useState(false)
  const [chatOpen, setChatOpen] = useState(false)
  const containerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const obs = new ResizeObserver(() => {
      if (containerRef.current) {
        setCanvasSize({
          width: containerRef.current.clientWidth,
          height: containerRef.current.clientHeight,
        })
      }
    })
    if (containerRef.current) obs.observe(containerRef.current)
    return () => obs.disconnect()
  }, [])

  function updateRoomDim(roomId: string, field: 'width' | 'height', val: number) {
    onUpdate({ ...plan, rooms: plan.rooms.map(r => (r.id === roomId ? { ...r, [field]: val } : r)) })
  }

  const selectedRoom = plan.rooms.find(r => r.id === selectedId)
  const ceilH = plan.ceilingHeight ?? 9

  return (
    <div className="editor">
      {/* Interior view overlay */}
      {interiorRoom && (
        <RoomInteriorView
          room={interiorRoom}
          ceilingHeight={ceilH}
          style={plan.style ?? 'modern'}
          onClose={() => setInteriorRoom(null)}
        />
      )}

      {/* ── Top toolbar ──────────────────────────────────────────────────── */}
      <div className="editor-toolbar">
        <span className="editor-title">{plan.name}</span>
        <span className="editor-ceiling-tag">{ceilH}ft ceilings</span>

        <div className="editor-tabs">
          {TAB_LABELS.map(t => (
            <button
              key={t.id}
              className={`editor-tab ${tab === t.id ? 'active' : ''}`}
              onClick={() => setTab(t.id)}
            >
              {t.label}
            </button>
          ))}
        </div>

        <button className="export-btn" onClick={() => exportPdf(plan)}>↓ PDF</button>
        <button className="export-btn" onClick={() => exportDxf(plan)}>↓ CAD</button>
        <button
          className={`chat-toggle-btn ${chatOpen ? 'active' : ''}`}
          onClick={() => setChatOpen(v => !v)}
          title="AI Design Assistant"
        >
          ✦ AI Chat
        </button>
      </div>

      {/* ── Tab content ──────────────────────────────────────────────────── */}
      {tab === 'plan' && (
        <div className={`editor-body${chatOpen ? ' editor-body-split' : ''}`}>
          <div className="editor-canvas" ref={containerRef}>
            {/* View mode toggle — 5 modes */}
            <div className="plan-sub-controls">
              <div className="view-toggle view-toggle-expanded">
                {VIEW_BUTTONS.map(btn => (
                  <button
                    key={btn.id}
                    className={viewMode === btn.id ? 'active' : ''}
                    onClick={() => setViewMode(btn.id)}
                  >
                    <span className="view-btn-icon">{btn.icon}</span>
                    <span className="view-btn-label">{btn.label}</span>
                  </button>
                ))}
              </div>
            </div>

            {viewMode === '2d' && (
              <div className="view-overlay-toggles">
                <button
                  className={`overlay-btn ${showSitePlan ? 'active' : ''}`}
                  onClick={() => setShowSitePlan(v => !v)}
                  title="Show lot boundary and setback lines"
                >Site</button>
                <button
                  className={`overlay-btn ${showStructGrid ? 'active' : ''}`}
                  onClick={() => setShowStructGrid(v => !v)}
                  title="Show 16ft structural column grid"
                >Grid</button>
              </div>
            )}

            {viewMode === '2d' && plan.stories === 2 && (
              <div className="floor-selector">
                <button
                  className={activeFloor === 1 ? 'active' : ''}
                  onClick={() => setActiveFloor(1)}
                >Floor 1</button>
                <button
                  className={activeFloor === 2 ? 'active' : ''}
                  onClick={() => setActiveFloor(2)}
                >Floor 2</button>
              </div>
            )}

            {viewMode === '2d' ? (
              <ArchPlan
                plan={plan}
                selectedId={selectedId}
                onSelect={setSelectedId}
                containerWidth={canvasSize.width}
                containerHeight={canvasSize.height}
                activeFloor={activeFloor}
                showSitePlan={showSitePlan}
                showStructGrid={showStructGrid}
              />
            ) : (
              <View3D plan={plan} initialMode={viewMode} />
            )}
          </div>

          {/* Inspector */}
          <div className="inspector">
            <div className="inspector-section">
              <div className="inspector-title">Rooms</div>
              <div className="room-list">
                {(plan.stories === 2
                  ? plan.rooms.filter(r => (r.floor ?? 1) === activeFloor)
                  : plan.rooms
                ).map(room => (
                  <div
                    key={room.id}
                    className={`room-item ${room.id === selectedId ? 'selected' : ''}`}
                    onClick={() => setSelectedId(room.id === selectedId ? null : room.id)}
                  >
                    <div className="room-swatch" style={{ background: room.color }} />
                    <div style={{ flex: 1 }}>
                      <div className="room-item-name">{room.name}</div>
                      <div className="room-item-size">
                        {room.width.toFixed(0)}' × {room.height.toFixed(0)}' · {Math.round(room.width * room.height)} sq ft
                      </div>
                    </div>
                    <button
                      className="room-interior-btn"
                      title="View interior"
                      onClick={e => { e.stopPropagation(); setInteriorRoom(room) }}
                    >↗</button>
                  </div>
                ))}
              </div>
            </div>

            {selectedRoom && (
              <div className="inspector-section room-edit">
                <div className="inspector-title">Edit Room</div>
                <div className="inspector-name">{selectedRoom.name}</div>
                <div className="inspector-field">
                  <label>Width (ft)</label>
                  <input type="number" min={6} max={60} value={selectedRoom.width}
                    onChange={e => { const v = parseFloat(e.target.value); if (!isNaN(v) && v >= 4) updateRoomDim(selectedRoom.id, 'width', v) }} />
                </div>
                <div className="inspector-field">
                  <label>Depth (ft)</label>
                  <input type="number" min={6} max={60} value={selectedRoom.height}
                    onChange={e => { const v = parseFloat(e.target.value); if (!isNaN(v) && v >= 4) updateRoomDim(selectedRoom.id, 'height', v) }} />
                </div>
                <div className="inspector-area">{Math.round(selectedRoom.width * selectedRoom.height).toLocaleString()} sq ft</div>
                <button className="view-interior-btn" onClick={() => setInteriorRoom(selectedRoom)}>
                  View Interior →
                </button>
              </div>
            )}

            <div className="inspector-section stats">
              <div className="inspector-title">Summary</div>
              <div className="stat-row"><span>Rooms</span><span>{plan.rooms.length}</span></div>
              <div className="stat-row">
                <span>Living area</span>
                <span>{Math.round(plan.rooms.filter(r => !['garage','patio','deck','rear_patio','outdoor_living','front_porch'].includes(r.type)).reduce((s, r) => s + r.width * r.height, 0)).toLocaleString()} sq ft</span>
              </div>
              {plan.stories === 2 && (
                <div className="stat-row">
                  <span>Stories</span><span>2-Story</span>
                </div>
              )}
              <div className="stat-row">
                <span>Footprint</span>
                <span>
                  {Math.round(plan.totalWidth)}' × {Math.round(plan.totalHeight)}'
                  {plan.stories === 2 && plan.floor2Height ? ` + ${Math.round(plan.floor2Height)}'` : ''}
                </span>
              </div>
              <div className="stat-row"><span>Ceiling</span><span>{ceilH} ft</span></div>
            </div>
          </div>

          {/* ── AI Chat side panel ── */}
          {chatOpen && (
            <div className="editor-chat-panel">
              <ChatPanel
                plan={plan}
                constraints={constraints}
                onPlanUpdate={(updatedPlan, updatedC) => onUpdate(updatedPlan, updatedC)}
              />
            </div>
          )}
        </div>
      )}

      {tab === 'elevations' && (
        <div className="tab-content-scroll">
          <ElevationView plan={plan} />
        </div>
      )}

      {tab === 'spec' && (
        <div className="tab-content-scroll">
          <SpecSchedule plan={plan} />
        </div>
      )}

      {tab === 'cost' && (
        <div className="tab-content-scroll">
          <CostPanel plan={plan} />
        </div>
      )}

      {tab === 'score' && (
        <div className="tab-content-scroll">
          <DesignScore plan={plan} />
        </div>
      )}

    </div>
  )
}
