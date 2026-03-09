import { FloorPlan } from '../types/floorplan'

interface Props {
  plan: FloorPlan
  width?: number
  height?: number
}

export default function FloorPlanPreview({ plan, width = 300, height = 210 }: Props) {
  const pad = 12
  const scaleX = (width - pad * 2) / plan.totalWidth
  const scaleY = (height - pad * 2) / plan.totalHeight
  const scale = Math.min(scaleX, scaleY)
  const ox = (width - plan.totalWidth * scale) / 2
  const oy = (height - plan.totalHeight * scale) / 2

  return (
    <svg width={width} height={height} style={{ display: 'block', background: '#F7F4EF' }}>
      <rect
        x={ox} y={oy}
        width={plan.totalWidth * scale}
        height={plan.totalHeight * scale}
        fill="none" stroke="#BBBBBB" strokeWidth={1.5}
      />
      {plan.rooms.map(room => {
        const rx = ox + room.x * scale
        const ry = oy + room.y * scale
        const rw = room.width * scale
        const rh = room.height * scale
        const showLabel = rw > 38 && rh > 18
        const shortName = room.name.length > 10 ? room.name.split(' ')[0] : room.name
        return (
          <g key={room.id}>
            <rect x={rx} y={ry} width={rw} height={rh}
              fill={room.color} stroke="#999" strokeWidth={0.8} />
            {showLabel && (
              <text
                x={rx + rw / 2} y={ry + rh / 2}
                textAnchor="middle" dominantBaseline="middle"
                fontSize={Math.min(10, rw / 5)}
                fill="#555" fontFamily="system-ui"
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
