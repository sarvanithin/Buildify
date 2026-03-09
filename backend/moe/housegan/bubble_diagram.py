"""
Bubble Diagram Builder — converts user constraints into a room adjacency graph
suitable as input for HouseGAN++.

A bubble diagram is what architects draw first: circles (rooms) connected by
lines (adjacency requirements). HouseGAN++ takes this graph and generates a
spatially valid floor plan from it.

US Residential Adjacency Rules (from architectural practice):
  - Foyer  ↔  Living Room (strong: entry sequence)
  - Foyer  ↔  Garage (moderate: interior garage access)
  - Kitchen ↔  Dining Room (strong: food service)
  - Kitchen ↔  Family Room (strong: open plan)
  - Kitchen ↔  Mudroom/Laundry (moderate: service entry)
  - Living  ↔  Dining (moderate: entertainment flow)
  - Master Bedroom ↔  Ensuite Bath (required)
  - Master Bedroom ↔  Walk-in Closet (required)
  - Bedrooms ↔  Hallway (required: circulation)
  - Bathrooms ↔  Hallway (required: circulation)
  - Hallway ↔  Laundry (moderate)
  - Living Room ↔  Patio/Deck (moderate: indoor-outdoor)
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict


# ── HouseGAN++ room type IDs ─────────────────────────────────────────────────
# These match the RPLAN dataset categories used in HouseGAN++ training.
# We add US-specific types (garage, mudroom) as extensions.
HG_TYPES = {
    "living_room":      1,
    "master_bedroom":   2,
    "kitchen":          3,
    "bathroom":         4,
    "dining_room":      5,
    "bedroom":          6,   # children/secondary room
    "home_office":      7,   # study room
    "bedroom_guest":    8,   # guest room → maps to bedroom
    "balcony":          9,   # maps to patio/deck
    "foyer":            10,  # entrance
    "closet":           11,  # storage / walk-in closet
    "laundry_room":     12,
    "hallway":          13,
    "garage":           14,  # extension (not in original RPLAN)
    "mudroom":          15,  # extension
}

# Reverse map: HG type ID → display name
HG_TYPE_NAMES = {v: k for k, v in HG_TYPES.items()}


@dataclass
class BubbleRoom:
    """A single room node in the bubble diagram."""
    id: str
    name: str
    buildify_type: str    # e.g. "master_bedroom"
    hg_type: int          # HouseGAN++ type ID
    area_sqft: float      # target area in sq ft
    min_w: float          # minimum width in feet
    min_h: float          # minimum depth in feet
    zone: str             # "public" | "private" | "service" | "outdoor"


@dataclass
class BubbleDiagram:
    """Room adjacency graph — input to HouseGAN++."""
    rooms: List[BubbleRoom]
    adj_matrix: np.ndarray   # (N, N) float32 — 1.0 = must be adjacent, 0.5 = preferred
    house_w: float           # target footprint width in feet
    house_h: float           # target footprint height in feet

    @property
    def n(self) -> int:
        return len(self.rooms)

    @property
    def hg_type_vector(self) -> np.ndarray:
        """(N,) int32 — HouseGAN++ type IDs for each room."""
        return np.array([r.hg_type for r in self.rooms], dtype=np.int32)

    @property
    def binary_adj(self) -> np.ndarray:
        """(N, N) binary adjacency (threshold at 0.4)."""
        return (self.adj_matrix >= 0.4).astype(np.float32)


# ── Area targets per room type (sq ft) ───────────────────────────────────────
# Based on US IRC standards + typical residential practice
AREA_TARGETS = {
    "living_room":      220,
    "family_room":      200,
    "great_room":       300,
    "kitchen":          150,
    "dining_room":      140,
    "master_bedroom":   210,
    "bedroom":          150,
    "ensuite_bathroom": 85,
    "bathroom":         65,
    "half_bath":        30,
    "hallway":          80,
    "foyer":            100,
    "home_office":      130,
    "laundry_room":     70,
    "garage_1car":      240,
    "garage_2car":      480,
    "garage_3car":      680,
    "walk_in_closet":   60,
    "closet":           25,
    "mudroom":          70,
    "patio":            200,
    "deck":             160,
}

MIN_DIMS = {
    # (min_w, min_h) in feet — IRC R304 minimums + good practice
    "living_room":      (12, 14),
    "kitchen":          (10, 12),
    "dining_room":      (11, 11),
    "master_bedroom":   (12, 14),
    "bedroom":          (10, 10),
    "ensuite_bathroom": (7,  9),
    "bathroom":         (5,  8),
    "half_bath":        (3,  6),
    "hallway":          (4,  8),
    "foyer":            (6,  8),
    "home_office":      (9, 10),
    "laundry_room":     (6,  7),
    "garage":           (12, 20),
    "walk_in_closet":   (5,  7),
    "closet":           (3,  4),
    "mudroom":          (6,  7),
    "patio":            (12, 14),
    "deck":             (10, 12),
}

ZONE_MAP = {
    "living_room": "public",   "family_room": "public",
    "great_room":  "public",   "kitchen":     "public",
    "dining_room": "public",   "home_office": "public",
    "foyer":       "service",  "hallway":     "service",
    "laundry_room":"service",  "mudroom":     "service",
    "garage":      "service",  "closet":      "service",
    "master_bedroom":  "private", "bedroom":         "private",
    "ensuite_bathroom":"private", "bathroom":        "private",
    "half_bath":       "private", "walk_in_closet":  "private",
    "patio":  "outdoor",  "deck": "outdoor",
}


# ── Core builder ─────────────────────────────────────────────────────────────

def build_bubble_diagram(constraints: dict) -> BubbleDiagram:
    """
    Convert Buildify user constraints into a HouseGAN++-compatible
    room adjacency graph.

    Args:
        constraints: dict with keys: sqft, bedrooms, bathrooms, garage,
                     outdoor, openPlan, primarySuite, homeOffice,
                     formalDining, laundry, ceilingHeight, style

    Returns:
        BubbleDiagram with rooms + adjacency matrix
    """
    sqft     = int(constraints.get("sqft", 1800))
    beds     = int(constraints.get("bedrooms", 3))
    baths    = int(constraints.get("bathrooms", 2))
    garage   = constraints.get("garage", "none")
    outdoor  = constraints.get("outdoor", "none")
    open_plan   = constraints.get("openPlan", False)
    pri_suite   = constraints.get("primarySuite", True)
    has_office  = constraints.get("homeOffice", False)
    formal_din  = constraints.get("formalDining", False)
    laundry     = constraints.get("laundry", "room")

    # Scale factor: rooms in larger homes are bigger
    sf = (sqft / 1800) ** 0.45

    rooms: List[BubbleRoom] = []

    def add(rid, name, btype, hg_type, base_area, zone_override=None):
        scaled_area = base_area * (sf ** 0.8)  # mild scaling
        mw, mh = MIN_DIMS.get(btype, (8, 8))
        z = zone_override or ZONE_MAP.get(btype, "public")
        rooms.append(BubbleRoom(
            id=rid, name=name, buildify_type=btype,
            hg_type=hg_type, area_sqft=scaled_area,
            min_w=mw, min_h=mh, zone=z,
        ))

    # ── Public zone ──────────────────────────────────────────────────────────
    if open_plan:
        add("great",   "Great Room",   "great_room",   HG_TYPES["living_room"],  350)
    else:
        add("living",  "Living Room",  "living_room",  HG_TYPES["living_room"],  220)
        add("kitchen", "Kitchen",      "kitchen",      HG_TYPES["kitchen"],      150)
        if formal_din:
            add("dining", "Dining Room", "dining_room", HG_TYPES["dining_room"], 160)
        else:
            add("dining", "Dining Area", "dining_room", HG_TYPES["dining_room"], 130)

    if has_office:
        add("office", "Home Office", "home_office", HG_TYPES["home_office"], 130)

    # ── Bedrooms ─────────────────────────────────────────────────────────────
    add("bed1", "Primary Bedroom", "master_bedroom", HG_TYPES["master_bedroom"], 210)
    for i in range(1, beds):
        add(f"bed{i+1}", f"Bedroom {i+1}", "bedroom", HG_TYPES["bedroom"], 150)

    # ── Bathrooms ────────────────────────────────────────────────────────────
    if pri_suite:
        add("ensuite", "Primary En-Suite",  "ensuite_bathroom", HG_TYPES["bathroom"], 90)
        add("wcloset", "Walk-in Closet",    "walk_in_closet",   HG_TYPES["closet"],   65)

    shared_baths = max(0, baths - (1 if pri_suite else 0))
    for i in range(shared_baths):
        if i == 0:
            add(f"bath{i+1}", "Full Bathroom", "bathroom", HG_TYPES["bathroom"], 65)
        else:
            add(f"bath{i+1}", "Powder Bath",   "half_bath", HG_TYPES["bathroom"], 30)

    # ── Service zone ─────────────────────────────────────────────────────────
    add("foyer",  "Entry Foyer", "foyer",   HG_TYPES["foyer"],    100)
    add("hall",   "Hallway",     "hallway", HG_TYPES["hallway"],   80)

    if laundry == "room":
        add("laundry", "Laundry Room",   "laundry_room", HG_TYPES["laundry_room"], 70)
    elif laundry == "closet":
        add("laundry", "Laundry Closet", "laundry_room", HG_TYPES["laundry_room"], 30)

    # ── Garage ───────────────────────────────────────────────────────────────
    if garage != "none":
        area = {"1car": 240, "2car": 480, "3car": 680}[garage]
        add("garage", "Garage", "garage", HG_TYPES["garage"], area)
        add("mudroom", "Mudroom", "mudroom", HG_TYPES["mudroom"], 60)

    # ── Outdoor ──────────────────────────────────────────────────────────────
    if outdoor in ("patio", "both"):
        add("patio", "Patio", "patio", HG_TYPES["balcony"], 200, "outdoor")
    if outdoor in ("deck", "both"):
        add("deck", "Deck", "deck", HG_TYPES["balcony"], 160, "outdoor")

    # ── Build adjacency matrix ────────────────────────────────────────────────
    n = len(rooms)
    adj = np.zeros((n, n), dtype=np.float32)
    id_to_idx = {r.id: i for i, r in enumerate(rooms)}

    def connect(a: str, b: str, strength: float = 1.0):
        if a in id_to_idx and b in id_to_idx:
            i, j = id_to_idx[a], id_to_idx[b]
            adj[i, j] = adj[j, i] = max(adj[i, j], strength)

    # Required adjacencies
    connect("foyer",   "living",   1.0)
    connect("foyer",   "great",    1.0)
    connect("foyer",   "hall",     1.0)
    connect("foyer",   "garage",   0.8)
    connect("foyer",   "mudroom",  0.9)

    connect("kitchen", "dining",   1.0)
    connect("kitchen", "great",    0.9)
    connect("kitchen", "living",   0.7)
    connect("kitchen", "laundry",  0.7)
    connect("kitchen", "mudroom",  0.7)
    connect("kitchen", "patio",    0.6)

    connect("living",  "dining",   0.8)
    connect("living",  "great",    0.0)   # same room if open plan
    connect("living",  "patio",    0.8)
    connect("living",  "deck",     0.8)

    connect("hall",    "bed1",     1.0)
    for i in range(1, beds):
        connect("hall", f"bed{i+1}", 1.0)
    connect("hall",    "bath1",    0.9)
    connect("hall",    "laundry",  0.6)

    connect("bed1",    "ensuite",  1.0)
    connect("bed1",    "wcloset",  1.0)

    connect("garage",  "mudroom",  1.0)
    connect("mudroom", "hall",     0.8)

    # ── Compute target footprint ───────────────────────────────────────────
    total_area = sum(r.area_sqft for r in rooms
                     if r.zone != "outdoor")
    fp = total_area / 0.82   # ~82% efficiency (walls + mechanical)
    H  = round((fp / 1.35) ** 0.5, 1)
    W  = round(H * 1.35, 1)
    if garage == "2car": W = max(W, 46.0)
    if garage == "3car": W = max(W, 58.0)
    W = max(W, 36.0)
    H = round(fp / W, 1)

    return BubbleDiagram(rooms=rooms, adj_matrix=adj, house_w=W, house_h=H)


def diagram_summary(d: BubbleDiagram) -> str:
    """Human-readable summary of a bubble diagram."""
    lines = [f"Bubble Diagram — {d.n} rooms — footprint {d.house_w}×{d.house_h}ft"]
    for r in d.rooms:
        adj_ids = [d.rooms[j].id for j in range(d.n) if d.adj_matrix[d.rooms.index(r), j] > 0]
        lines.append(f"  {r.id:12s} [{r.zone:8s}] ~{r.area_sqft:.0f}sqft  adj→ {adj_ids}")
    return "\n".join(lines)
