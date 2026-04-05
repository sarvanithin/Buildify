export interface Room {
  id: string
  name: string
  type: string
  x: number
  y: number
  width: number
  height: number
  color: string
}

export interface Door {
  x: number
  y: number
  isVertical: boolean
  roomA: string
  roomB: string
}

export interface FloorPlan {
  id: string
  name: string
  totalWidth: number
  totalHeight: number
  ceilingHeight: number   // feet
  rooms: Room[]
  doors: Door[]
}

export interface Constraints {
  // Basics
  bedrooms: number
  bathrooms: number
  sqft: number
  stories: number
  style: string

  // Layout options
  openPlan: boolean        // merge living/kitchen/dining
  primarySuite: boolean    // ensuite + walk-in closet for primary bedroom
  homeOffice: boolean
  formalDining: boolean    // separate formal dining (ignored when openPlan)

  // Spaces
  garage: 'none' | '1car' | '2car' | '3car'
  laundry: 'none' | 'closet' | 'room'
  outdoor: 'none' | 'patio' | 'deck' | 'both'

  // Style
  ceilingHeight: 'standard' | 'high' | 'vaulted'   // 9ft / 10ft / 12ft
}

export interface ValidationIssue {
  field: string
  severity: 'error' | 'warning'
  message: string
  detail: string
}
