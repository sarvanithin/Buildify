import axios from 'axios'
import { Constraints, FloorPlan } from '../types/floorplan'

const api = axios.create({ baseURL: '/api' })

export async function generatePlans(constraints: Constraints): Promise<FloorPlan[]> {
  const { data } = await api.post('/generate', constraints)
  return data.plans
}

export async function exportDxf(plan: FloorPlan): Promise<void> {
  const response = await api.post(
    '/export/dxf',
    { floor_plan: plan },
    { responseType: 'blob' }
  )
  const url = URL.createObjectURL(response.data)
  const a = document.createElement('a')
  a.href = url
  a.download = `${plan.name.replace(/\s+/g, '_')}.dxf`
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

export async function fetchCostEstimate(plan: FloorPlan, region: string): Promise<CostResult> {
  const { data } = await api.post('/cost', { floor_plan: plan, region })
  return data
}

export async function fetchDesignScore(plan: FloorPlan): Promise<ScoreResult> {
  const { data } = await api.post('/score', { floor_plan: plan })
  return data
}

export async function fetchCostRegions(): Promise<string[]> {
  const { data } = await api.get('/cost/regions')
  return data.regions
}

export async function sendChatMessage(
  plan: FloorPlan,
  messages: ChatMsg[]
): Promise<{ reply: string; updated_plan: FloorPlan | null }> {
  const { data } = await api.post('/chat', { floor_plan: plan, messages })
  return data
}

// ── types returned by API ────────────────────────────────────────────────────

export interface CostRoomRow {
  room: string
  type: string
  sqft: number
  low: number
  mid: number
  high: number
}

export interface CostResult {
  region: string
  rooms: CostRoomRow[]
  foundation: { low: number; mid: number; high: number }
  roof:       { low: number; mid: number; high: number }
  mep:        { low: number; mid: number; high: number }
  total:      { low: number; mid: number; high: number }
  per_sqft:   { low: number; mid: number; high: number }
}

export interface ScoreResult {
  scores: {
    adjacency: number
    natural_light: number
    circulation: number
    privacy: number
    efficiency: number
  }
  overall: number
  grade: string
  insights: string[]
}

export interface ChatMsg {
  role: 'user' | 'assistant'
  content: string
}

// ── MOE API ──────────────────────────────────────────────────────────────────

export interface MOEResult {
  plans: FloorPlan[]
  expert_weights: Record<string, number>
  confidence: number
  irc_compliant: boolean
}

export interface ExpertWeightsResult {
  expert_weights: Record<string, number>
  expert_names: string[]
  confidence: number
}

export interface AuthResult {
  api_key: string
  tier: string
  message: string
}

export interface UsageResult {
  tier: string
  generations_today: number
  daily_limit: number | string
  total_generations: number
  total_exports: number
  total_chats: number
  variants_per_request: number
}

export async function generatePlansMOE(
  constraints: Constraints,
  apiKey?: string
): Promise<MOEResult> {
  const headers: Record<string, string> = {}
  if (apiKey) headers['X-API-Key'] = apiKey
  const { data } = await api.post('/generate/moe', constraints, { headers })
  return data
}

export async function getExpertWeights(
  constraints: Constraints
): Promise<ExpertWeightsResult> {
  const { data } = await api.post('/moe/experts', constraints)
  return data
}

export async function registerApiKey(
  email: string = '',
  tier: string = 'free'
): Promise<AuthResult> {
  const { data } = await api.post('/auth/register', { email, tier })
  return data
}

export async function getUsage(apiKey: string): Promise<UsageResult> {
  const { data } = await api.get('/auth/usage', {
    headers: { 'X-API-Key': apiKey },
  })
  return data
}
