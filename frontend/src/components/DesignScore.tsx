import { useState, useEffect } from 'react'
import { FloorPlan } from '../types/floorplan'
import { fetchDesignScore, ScoreResult } from '../api/client'

interface Props {
  plan: FloorPlan
}

const SCORE_META: Record<string, { label: string; icon: string; desc: string }> = {
  adjacency:     { label: 'Room Adjacency',   icon: '🔗', desc: 'Ideal room-to-room proximity (kitchen↔dining, bed↔bath…)' },
  natural_light: { label: 'Natural Light',    icon: '☀️', desc: 'Percentage of living spaces on exterior walls' },
  circulation:   { label: 'Circulation',      icon: '🚶', desc: 'Foyer, hallway adequacy and traffic flow' },
  privacy:       { label: 'Privacy Zoning',   icon: '🔒', desc: 'Bedroom separation from public living areas' },
  efficiency:    { label: 'Space Efficiency', icon: '📐', desc: 'How well rooms fill the building footprint' },
}

function ScoreBar({ score }: { score: number }) {
  const color = score >= 85 ? '#22C55E' : score >= 70 ? '#F59E0B' : '#EF4444'
  return (
    <div className="score-bar-bg">
      <div
        className="score-bar-fill"
        style={{ width: `${score}%`, background: color }}
      />
    </div>
  )
}

function GradeCircle({ grade, overall }: { grade: string; overall: number }) {
  const color = overall >= 85 ? '#22C55E' : overall >= 70 ? '#F59E0B' : '#EF4444'
  return (
    <div className="grade-circle" style={{ borderColor: color, color }}>
      <span className="grade-letter">{grade}</span>
      <span className="grade-score">{Math.round(overall)}</span>
    </div>
  )
}

export default function DesignScore({ plan }: Props) {
  const [result, setResult] = useState<ScoreResult | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    run()
  }, [plan.id])

  async function run() {
    setLoading(true)
    try {
      const r = await fetchDesignScore(plan)
      setResult(r)
    } catch { /* noop */ }
    finally { setLoading(false) }
  }

  return (
    <div className="score-wrap">
      <div className="score-header">
        <h3>Design Quality Score</h3>
        <p className="score-subhead">Automated analysis against US residential architectural standards</p>
      </div>

      {loading && <div className="score-loading">Analyzing layout...</div>}

      {result && !loading && (
        <div className="score-body">
          <div className="score-left">
            <GradeCircle grade={result.grade} overall={result.overall} />
            <div className="score-overall-label">Overall Score</div>

            <div className="score-insights">
              <div className="score-insights-title">AI Insights</div>
              {result.insights.map((ins, i) => (
                <div key={i} className="score-insight-item">
                  {ins.startsWith('Excellent') || ins.startsWith('Great')
                    ? '✅' : '⚠️'} {ins}
                </div>
              ))}
            </div>
          </div>

          <div className="score-right">
            {Object.entries(result.scores).map(([key, val]) => {
              const meta = SCORE_META[key] || { label: key, icon: '📊', desc: '' }
              return (
                <div key={key} className="score-row">
                  <div className="score-row-top">
                    <span className="score-icon">{meta.icon}</span>
                    <span className="score-label">{meta.label}</span>
                    <span className="score-val">{Math.round(val)}</span>
                  </div>
                  <ScoreBar score={val} />
                  <div className="score-desc">{meta.desc}</div>
                </div>
              )
            })}
          </div>
        </div>
      )}
    </div>
  )
}
