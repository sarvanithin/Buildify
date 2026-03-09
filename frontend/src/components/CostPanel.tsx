import { useState, useEffect } from 'react'
import { FloorPlan } from '../types/floorplan'
import { fetchCostEstimate, fetchCostRegions, CostResult } from '../api/client'

interface Props {
  plan: FloorPlan
}

const fmt = (n: number) => '$' + n.toLocaleString()

export default function CostPanel({ plan }: Props) {
  const [regions, setRegions] = useState<string[]>([])
  const [region, setRegion] = useState('National Average')
  const [result, setResult] = useState<CostResult | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchCostRegions().then(setRegions).catch(() => {})
  }, [])

  useEffect(() => {
    run()
  }, [plan.id, region])

  async function run() {
    setLoading(true)
    try {
      const r = await fetchCostEstimate(plan, region)
      setResult(r)
    } catch { /* noop */ }
    finally { setLoading(false) }
  }

  return (
    <div className="cost-wrap">
      <div className="cost-header">
        <h3>Construction Cost Estimate</h3>
        <select
          className="region-select"
          value={region}
          onChange={e => setRegion(e.target.value)}
        >
          {regions.map(r => <option key={r} value={r}>{r}</option>)}
        </select>
      </div>

      {loading && <div className="cost-loading">Calculating...</div>}

      {result && !loading && (
        <>
          {/* Summary cards */}
          <div className="cost-summary-row">
            {(['low','mid','high'] as const).map(tier => (
              <div key={tier} className={`cost-card cost-card-${tier}`}>
                <div className="cost-card-label">
                  {tier === 'low' ? '🏠 Economy' : tier === 'mid' ? '⭐ Mid-Range' : '✨ Premium'}
                </div>
                <div className="cost-card-total">{fmt(result.total[tier])}</div>
                <div className="cost-card-psf">{fmt(result.per_sqft[tier])} /sq ft</div>
              </div>
            ))}
          </div>

          {/* Breakdown table */}
          <div className="cost-section-title">Room-by-Room Breakdown</div>
          <div className="cost-table-wrap">
            <table className="cost-table">
              <thead>
                <tr>
                  <th>Room</th>
                  <th>Area</th>
                  <th>Economy</th>
                  <th>Mid-Range</th>
                  <th>Premium</th>
                </tr>
              </thead>
              <tbody>
                {result.rooms.map((row, i) => (
                  <tr key={i}>
                    <td>{row.room}</td>
                    <td>{row.sqft.toLocaleString()} sf</td>
                    <td>{fmt(row.low)}</td>
                    <td className="cost-mid">{fmt(row.mid)}</td>
                    <td>{fmt(row.high)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Fixed cost breakdown */}
          <div className="cost-section-title">Additional Cost Components</div>
          <div className="cost-fixed-grid">
            {[
              { label: 'Foundation', val: result.foundation },
              { label: 'Roof System', val: result.roof },
              { label: 'MEP (Mechanical / Electrical / Plumbing)', val: result.mep },
            ].map(item => (
              <div key={item.label} className="cost-fixed-row">
                <span className="cost-fixed-label">{item.label}</span>
                <span>{fmt(item.val.low)} – {fmt(item.val.high)}</span>
              </div>
            ))}
          </div>

          <div className="cost-disclaimer">
            * Estimates are illustrative only. Actual costs vary by location, site conditions,
            material selections, and contractor. Region: <strong>{result.region}</strong>.
            Consult a licensed general contractor for binding quotes.
          </div>
        </>
      )}
    </div>
  )
}
