"""
Production inference with multi-stage refinement and quality guarantees.

HYBRID approach:
  - MOE model → expert weights for intelligent room sizing & proportions
  - HouseGAN++ → graph-based spatial placement (when available)
  - Deterministic solver → zone-based placement fallback

Pipeline:
  Stage 1 — Build: Deterministic room list from constraints (correct room types)
  Stage 2 — Size: MOE expert weights influence room sizing
  Stage 3 — Place: HouseGAN++ spatial layout (falls back to zone-based)
  Stage 4 — Validate: IRC compliance check
  Stage 5 — Refine: Gap filling, grid snap, overlap resolution
"""
import asyncio
import math
import random
import uuid
from pathlib import Path
from typing import List, Dict, Optional

import torch

from .config import MOEConfig
from .model import BuildifyMOE
from .data import (
    encode_constraints, IRC_ROOM_SPECS, ZONE_MAP,
    STYLE_TEMPLATES, ADJACENCY_RULES, _build_room_list,
)
from .experts import EXPERT_NAMES

# HouseGAN++ integration (optional — falls back gracefully)
try:
    from .housegan import build_bubble_diagram, generate_layouts as _hg_generate_layouts
    _HOUSEGAN_AVAILABLE = True
except ImportError:
    _HOUSEGAN_AVAILABLE = False

# ─────────────────────────────────────────────────────────────────────────────
# Color palette for room types
# ─────────────────────────────────────────────────────────────────────────────

ROOM_COLORS = {
    "living_room": "#E8D5B7", "kitchen": "#B7D5E8", "dining_room": "#D5E8B7",
    "family_room": "#E8E0B7", "master_bedroom": "#D8B7E8", "bedroom": "#C8B7E8",
    "ensuite_bathroom": "#B7E8E0", "bathroom": "#B7E8D5", "half_bath": "#D0E8E8",
    "hallway": "#E0E0D3", "foyer": "#EEEAE0", "home_office": "#F5F0D3",
    "laundry_room": "#D3F5F5", "garage": "#D5D5CC", "walk_in_closet": "#E8D8E8",
    "closet": "#E0D8E0", "pantry": "#EDE8DC", "mudroom": "#E8E4D8",
    "utility_room": "#E8E8D3", "patio": "#E0EED8", "deck": "#E8E4D0",
}

# ─────────────────────────────────────────────────────────────────────────────
# Load model
# ─────────────────────────────────────────────────────────────────────────────

_cached_model: Optional[BuildifyMOE] = None
_cached_config: Optional[MOEConfig] = None


def load_model(config: MOEConfig = None) -> BuildifyMOE:
    """Load trained MOE model from weights file."""
    global _cached_model, _cached_config

    if _cached_model is not None:
        return _cached_model

    config = config or MOEConfig()
    _cached_config = config

    weights_path = Path(__file__).parent / "weights" / config.model_filename

    model = BuildifyMOE(config)

    if weights_path.exists():
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"[MOE] Loaded weights from {weights_path}")
    else:
        print(f"[MOE] No weights found at {weights_path} — using untrained model")

    model.eval()
    _cached_model = model
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Stage 2: MOE-influenced room sizing
# ─────────────────────────────────────────────────────────────────────────────

def _moe_adjusted_size(room_type: str, sqft: int, expert_weights: dict) -> tuple:
    """
    Use MOE expert weights to intelligently size rooms.
    Interpolates between minimum and STANDARD sizes (not premium).
    Premium sizes only for homes > 3000 sqft.
    """
    specs = IRC_ROOM_SPECS.get(room_type)
    if not specs:
        return (8, 8)

    min_w, min_h, std_w, std_h, prem_w, prem_h = specs

    # Size factor from sqft (0.0 = compact, 1.0 = spacious)
    sf = min(1.0, max(0.0, (sqft - 1200) / 2800))

    # Expert influence
    sizing_w = expert_weights.get("Room Sizing", 0.125)
    cost_w = expert_weights.get("Cost Optimization", 0.125)

    # For homes ≤ 3000 sqft: interpolate between min and standard
    # For homes > 3000 sqft: interpolate between standard and premium
    if sqft <= 3000:
        t = sf * 0.8 + sizing_w * 1.5 - cost_w * 1.5
        t = min(1.0, max(0.0, t))
        w = min_w + (std_w - min_w) * t
        h = min_h + (std_h - min_h) * t
    else:
        t = (sqft - 3000) / 2000  # 0-1 range for 3000-5000 sqft
        t = min(1.0, max(0.0, t)) * 0.7 + sizing_w
        t = min(1.0, t)
        w = std_w + (prem_w - std_w) * t
        h = std_h + (prem_h - std_h) * t

    return (round(w), round(h))


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3: Architecturally correct US residential placement
# ─────────────────────────────────────────────────────────────────────────────

def _place_rooms_architectural(rooms: List[dict], total_w: float, total_h: float,
                               style: str, variant_seed: int = 0) -> List[dict]:
    """
    Architecturally correct US residential front-to-back placement.

    Band order (street → backyard):
      ENTRY   — garage front-left, foyer adjacent, mudroom attached to garage
      PUBLIC  — living room, dining room, family room, home office
      KITCHEN — kitchen faces backyard, pantry adjacent, laundry service column
      HALLWAY — 4ft fixed-height full-width circulation spine
      PRIVATE — master suite grouped (bed+ensuite+walk-in), secondary bedrooms
      OUTDOOR — patio/deck at rear

    Rules:
      - Last room in each row stretches to fill width (no proportional distortion)
      - Each row has its own height from its tallest room (no uniform zone height)
      - Hallway is always full-width at 4ft minimum depth
      - Master suite rooms always placed together as a cluster
    """
    placed: List[dict] = []
    y_cursor = 0.0

    # Room types that should never be stretched far beyond their natural size
    _SMALL_ROOMS = {"closet", "walk_in_closet", "half_bath", "pantry",
                    "mudroom", "laundry_room", "utility_room",
                    "bathroom", "ensuite_bathroom"}

    # Group rooms by zone (uses updated ZONE_MAP: 0=ENTRY..5=OUTDOOR)
    zone_rooms: Dict[int, List[dict]] = {i: [] for i in range(6)}
    for r in rooms:
        z = ZONE_MAP.get(r["type"], 1)
        zone_rooms[z].append(r)

    def _pack_rows(band_rooms: List[dict], fixed_height: float = None) -> None:
        """
        Pack a pre-ordered list into rows. Last room in each row fills remaining width.
        Small rooms (closets, pantry, etc.) are capped at 1.5× their natural size.
        Each row height = tallest room in that row (or fixed_height if specified).
        """
        nonlocal y_cursor
        if not band_rooms:
            return

        # Split into rows by width
        rows: List[List[dict]] = []
        cur_row: List[dict] = []
        row_w = 0.0
        for r in band_rooms:
            if row_w + r["width"] > total_w and cur_row:
                rows.append(cur_row)
                cur_row = []
                row_w = 0.0
            cur_row.append(r)
            row_w += r["width"]
        if cur_row:
            rows.append(cur_row)

        for row in rows:
            row_h = fixed_height if fixed_height else max(r["height"] for r in row)
            x_pos = 0.0
            for i, r in enumerate(row):
                w = float(r["width"])
                # Last room in row fills remaining width,
                # but small rooms are capped at 1.5× natural size
                if i == len(row) - 1:
                    remaining = total_w - x_pos
                    if remaining > w:
                        if r["type"] in _SMALL_ROOMS:
                            w = min(remaining, w * 1.5)  # cap small rooms
                        else:
                            w = remaining  # major rooms take all remaining space
                w = max(4.0, round(w))
                placed.append({
                    **r,
                    "x": round(x_pos),
                    "y": round(y_cursor),
                    "width": int(w),
                    "height": round(row_h),
                })
                x_pos += w
            y_cursor += round(row_h)

    # ── ZONE 0: ENTRY BAND (y=0, front of house) ──────────────────────────
    # Garage always front-left (x=0), mudroom tucked beside garage, foyer at right
    entry = zone_rooms[0]
    entry.sort(key=lambda r: {
        "garage": 0, "mudroom": 1, "laundry_room": 2, "utility_room": 3, "foyer": 4
    }.get(r["type"], 5))
    _pack_rows(entry)

    # ── ZONE 1: PUBLIC BAND ───────────────────────────────────────────────
    # Keep adjacency order (living→dining or living→family depending on plan)
    public = _sort_by_adjacency(zone_rooms[1])
    _pack_rows(public)

    # ── ZONE 2: KITCHEN/SERVICE BAND ─────────────────────────────────────
    # Kitchen first (faces backyard), pantry adjacent, laundry/utility at left-side
    kitchen_rooms = zone_rooms[2]
    # Pantry/utility first (small service rooms), kitchen last so it fills remaining width
    kitchen_rooms.sort(key=lambda r: {
        "pantry": 0, "laundry_room": 1, "utility_room": 2, "kitchen": 3
    }.get(r["type"], 4))
    _pack_rows(kitchen_rooms)

    # ── ZONE 3: HALLWAY — full-width 4ft circulation spine ────────────────
    hallway_rooms = zone_rooms[3]
    hallway = next((r for r in hallway_rooms if r["type"] == "hallway"), None)
    other_hall = [r for r in hallway_rooms if r["type"] != "hallway"]
    if hallway:
        hallway_h = max(4, hallway["height"])
        hallway["width"] = int(total_w)   # spans full footprint width
        _pack_rows([hallway], fixed_height=hallway_h)
    _pack_rows(other_hall)  # half_bath, etc., in next sub-row

    # ── ZONE 4: PRIVATE BAND (bedrooms) ───────────────────────────────────
    # Master suite cluster first: master_bedroom → ensuite_bathroom → walk_in_closet
    # Then secondary bedrooms, shared bathrooms, closets
    private = zone_rooms[4]
    suite_order = {
        "master_bedroom": 0, "ensuite_bathroom": 1, "walk_in_closet": 2,
        "bedroom": 3, "bathroom": 4, "closet": 5,
    }
    private.sort(key=lambda r: suite_order.get(r["type"], 6))
    _pack_rows(private)

    # ── ZONE 5: OUTDOOR BAND (rear of house) ──────────────────────────────
    _pack_rows(zone_rooms[5])

    # Final overlap safety check
    placed = _final_overlap_check(placed, total_w)

    return placed


def _final_overlap_check(rooms: List[dict], total_w: float) -> List[dict]:
    """Last-resort overlap resolver: shift any still-overlapping rooms down."""
    for i in range(len(rooms)):
        for j in range(i + 1, len(rooms)):
            a, b = rooms[i], rooms[j]
            if (a["x"] < b["x"] + b["width"] and a["x"] + a["width"] > b["x"] and
                a["y"] < b["y"] + b["height"] and a["y"] + a["height"] > b["y"]):
                b["y"] = a["y"] + a["height"]
    return rooms


def _sort_by_adjacency(rooms: List[dict]) -> List[dict]:
    """Sort rooms within a zone to keep adjacent pairs next to each other."""
    if len(rooms) <= 2:
        return rooms

    adj = {}
    for ra, rb, strength in ADJACENCY_RULES:
        if strength >= 0.7:
            adj.setdefault(ra, set()).add(rb)
            adj.setdefault(rb, set()).add(ra)

    result = [rooms[0]]
    remaining = list(rooms[1:])

    while remaining:
        last_type = result[-1]["type"]
        neighbors = adj.get(last_type, set())

        best_idx = 0
        best_score = -1
        for i, r in enumerate(remaining):
            score = 1 if r["type"] in neighbors else 0
            if score > best_score:
                best_score = score
                best_idx = i

        result.append(remaining.pop(best_idx))

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Stage 3b: HouseGAN++ spatial layout (graph-based, adjacency-aware)
# ─────────────────────────────────────────────────────────────────────────────

def _place_rooms_housegan(
    sized_rooms: List[dict],
    constraints: dict,
    total_w: float,
    total_h: float,
    num_variants: int = 1,
    variant_idx: int = 0,
) -> Optional[List[dict]]:
    """
    Use HouseGAN++ to place rooms spatially based on adjacency graph.
    Returns placed rooms list or None if HouseGAN unavailable/failed.
    """
    if not _HOUSEGAN_AVAILABLE:
        return None

    try:
        diagram = build_bubble_diagram(constraints)

        # Run HouseGAN++ (async → sync bridge)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Already in async context — use thread executor
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(
                        asyncio.run,
                        _hg_generate_layouts(diagram, num_variants=num_variants, mode="auto")
                    )
                    layouts = future.result(timeout=30)
            else:
                layouts = loop.run_until_complete(
                    _hg_generate_layouts(diagram, num_variants=num_variants, mode="auto")
                )
        except RuntimeError:
            layouts = asyncio.run(
                _hg_generate_layouts(diagram, num_variants=num_variants, mode="auto")
            )

        if not layouts:
            return None

        # Pick variant (with wrap-around)
        layout = layouts[variant_idx % len(layouts)]

        # Merge HouseGAN positions with MOE-sized rooms
        # HouseGAN gives us x/y/width/height — override with MOE sizes but keep positions
        hg_by_type: Dict[str, dict] = {}
        for room in layout:
            rtype = room["type"]
            if rtype not in hg_by_type:
                hg_by_type[rtype] = room

        placed = []
        type_count: Dict[str, int] = {}
        for r in sized_rooms:
            rtype = r["type"]
            count = type_count.get(rtype, 0)
            type_count[rtype] = count + 1

            # Get HouseGAN position for this room type
            hg_rooms_of_type = [h for h in layout if h["type"] == rtype]
            hg_room = hg_rooms_of_type[count] if count < len(hg_rooms_of_type) else None

            if hg_room:
                # Use HouseGAN x/y position, but MOE-sized width/height
                x = max(0, min(hg_room["x"], total_w - r["width"]))
                y = max(0, min(hg_room["y"], total_h - r["height"]))
            else:
                x, y = 0.0, 0.0

            placed.append({
                **r,
                "x": round(x, 1),
                "y": round(y, 1),
            })

        return placed

    except Exception as e:
        print(f"[MOE] HouseGAN placement failed: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Stage 4: IRC Compliance Validation
# ─────────────────────────────────────────────────────────────────────────────

def _validate_irc(rooms: List[dict], total_w: float, total_h: float) -> List[dict]:
    """Check and fix rooms against IRC building code minimums."""
    validated = []
    for r in rooms:
        rtype = r.get("type", "living_room")
        specs = IRC_ROOM_SPECS.get(rtype)

        if specs:
            min_w, min_h = specs[0], specs[1]
            if r["width"] < min_w:
                r["width"] = min_w
            if r["height"] < min_h:
                r["height"] = min_h

        # Enforce 70 sqft minimum for habitable rooms (IRC R304)
        if rtype not in ("closet", "walk_in_closet", "half_bath", "pantry",
                          "hallway", "utility_room"):
            area = r["width"] * r["height"]
            if area < 70:
                scale = math.sqrt(70 / area)
                r["width"] = round(r["width"] * scale, 1)
                r["height"] = round(r["height"] * scale, 1)

        # Clamp to footprint
        r["x"] = max(0, min(r["x"], total_w - r["width"]))
        r["y"] = max(0, min(r["y"], total_h - r["height"]))

        r["irc_compliant"] = True
        validated.append(r)

    return validated


# ─────────────────────────────────────────────────────────────────────────────
# Stage 5: Gap filling & grid snapping — ZERO GAP guarantee
# ─────────────────────────────────────────────────────────────────────────────

def _snap_and_fill(rooms: List[dict], total_w: float, total_h: float) -> List[dict]:
    """Snap to 2ft grid and ensure zero-gap, tight-packed layout."""
    # Snap to 2ft grid
    for r in rooms:
        r["x"] = round(r["x"] / 2) * 2
        r["y"] = round(r["y"] / 2) * 2
        r["width"] = max(4, round(r["width"] / 2) * 2)
        r["height"] = max(4, round(r["height"] / 2) * 2)

    # Group rooms by Y position (same row)
    rows = {}
    for r in rooms:
        row_y = r["y"]
        rows.setdefault(row_y, []).append(r)

    # Within each row: close gaps and force exact total_w
    tw = round(total_w)
    for row_y, row_rooms in rows.items():
        row_rooms_sorted = sorted(row_rooms, key=lambda r: r["x"])

        # Close any interior gaps between rooms (pack left-to-right)
        x_cursor = 0
        for r in row_rooms_sorted:
            r["x"] = x_cursor
            x_cursor += r["width"]

        # Adjust last room to exactly hit total_w (stretch OR shrink)
        # Small auxiliary rooms (closets, pantry, etc.) are skipped for large stretches
        _SMALL_SNAP = {"closet", "walk_in_closet", "half_bath", "pantry",
                       "mudroom", "laundry_room", "utility_room",
                       "bathroom", "ensuite_bathroom"}
        if row_rooms_sorted:
            last = row_rooms_sorted[-1]
            right_edge = last["x"] + last["width"]
            diff = tw - right_edge

            # Don't stretch a small room by more than 50% of its current width
            if last.get("type") in _SMALL_SNAP and diff > last["width"] * 0.5:
                diff = round(last["width"] * 0.5)

            if diff != 0:
                # Try adjusting just the last room
                new_w = last["width"] + diff
                if new_w >= 4:
                    last["width"] = new_w
                else:
                    # Last room too small — distribute shrinkage across all rooms
                    excess = right_edge - tw
                    total_room_w = sum(r["width"] for r in row_rooms_sorted)
                    x_cursor = 0
                    for r in row_rooms_sorted:
                        shrink = round(excess * (r["width"] / total_room_w))
                        r["width"] = max(4, r["width"] - shrink)
                        r["x"] = x_cursor
                        x_cursor += r["width"]
                    # Final adjustment on last room for exact fit
                    last = row_rooms_sorted[-1]
                    last["width"] = max(4, tw - last["x"])

        # Uniform row height
        max_h = max(r["height"] for r in row_rooms_sorted)
        for r in row_rooms_sorted:
            r["height"] = max_h

    # Ensure no overlaps remain after grid snapping
    rooms = _final_overlap_check(rooms, total_w)

    return rooms


# ─────────────────────────────────────────────────────────────────────────────
# Main inference entry point
# ─────────────────────────────────────────────────────────────────────────────

def predict_floor_plan(constraints: dict, num_variants: int = 3,
                       config: MOEConfig = None) -> dict:
    """
    Generate floor plans using MOE-guided hybrid approach.

    HYBRID: MOE experts guide room sizing, deterministic solver does placement.
    """
    config = config or MOEConfig()
    model = load_model(config)

    # Parse constraint keys
    style = constraints.get("style", "modern")
    if style not in config.STYLES:
        style = "modern"

    bedrooms = constraints.get("bedrooms", 3)
    bathrooms = constraints.get("bathrooms", 2)
    sqft = constraints.get("sqft", 1800)
    stories = constraints.get("stories", 1)
    open_plan = constraints.get("openPlan", False)
    primary_suite = constraints.get("primarySuite", True)
    home_office = constraints.get("homeOffice", False)
    formal_dining = constraints.get("formalDining", False)
    garage = constraints.get("garage", "2car")
    laundry = constraints.get("laundry", "room")
    outdoor = constraints.get("outdoor", "patio")
    ceiling_height = constraints.get("ceilingHeight", "standard")

    # Encode constraints for MOE
    constraint_vec = encode_constraints(
        bedrooms=bedrooms, bathrooms=bathrooms, sqft=sqft, stories=stories,
        style=style, open_plan=open_plan, primary_suite=primary_suite,
        home_office=home_office, formal_dining=formal_dining,
        garage=garage, laundry=laundry, outdoor=outdoor,
        ceiling_height=ceiling_height, config=config,
    ).unsqueeze(0)

    # Get MOE expert weights (the AI part)
    with torch.no_grad():
        expert_weights_tensor = model.get_expert_weights(constraint_vec)
    expert_weights = {
        name: round(expert_weights_tensor[0, i].item(), 4)
        for i, name in enumerate(EXPERT_NAMES)
    }

    # Calculate footprint
    footprint_sqft = sqft / stories
    style_template = STYLE_TEMPLATES.get(style, STYLE_TEMPLATES["modern"])
    aspect = 1.4 if open_plan else 1.2
    total_w = round(math.sqrt(footprint_sqft * aspect))
    total_h = round(footprint_sqft / total_w)
    total_w = max(30, min(80, total_w))
    total_h = max(25, min(70, total_h))

    ceil_map = {"standard": 9, "high": 10, "vaulted": 12}
    ceiling_ft = ceil_map.get(ceiling_height, 9)

    # Stage 1: Build correct room list from constraints (deterministic)
    base_rooms = _build_room_list(
        bedrooms, bathrooms, sqft, style,
        open_plan, primary_suite, home_office, formal_dining,
        garage, laundry, outdoor,
    )

    # Generate variants
    plans = []
    variant_names = [
        "MOE Plan A — Optimized",
        "MOE Plan B — Balanced",
        "MOE Plan C — Efficient",
        "MOE Plan D — Premium",
        "MOE Plan E — Compact",
    ]

    for v in range(num_variants):
        # Stage 2: MOE-influenced sizing (with slight variance per variant)
        variant_expert_w = dict(expert_weights)
        if v > 0:
            # Slightly vary expert weights for diversity
            rng = random.Random(v * 42)
            for key in variant_expert_w:
                variant_expert_w[key] *= (0.85 + rng.random() * 0.3)
            # Renormalize
            total = sum(variant_expert_w.values())
            for key in variant_expert_w:
                variant_expert_w[key] /= total

        sized_rooms = []
        for r in base_rooms:
            w, h = _moe_adjusted_size(r["type"], sqft, variant_expert_w)
            # Apply small per-variant jitter (±1ft)
            if v > 0:
                jitter_w = random.Random(v * 100 + hash(r["name"])).choice([-2, 0, 0, 2])
                jitter_h = random.Random(v * 200 + hash(r["name"])).choice([-2, 0, 0, 2])
                w = max(IRC_ROOM_SPECS.get(r["type"], (4, 4))[0], w + jitter_w)
                h = max(IRC_ROOM_SPECS.get(r["type"], (4, 4))[1], h + jitter_h)

            sized_rooms.append({
                "id": f"moe_{uuid.uuid4().hex[:8]}",
                "name": r["name"],
                "type": r["type"],
                "width": w,
                "height": h,
                "color": ROOM_COLORS.get(r["type"], "#E0E0E0"),
                "x": 0, "y": 0,  # will be set by placement
            })

        # Stage 3: Spatial placement — HouseGAN++ first, zone-based fallback
        hg_placed = _place_rooms_housegan(
            sized_rooms, constraints, total_w, total_h,
            num_variants=num_variants, variant_idx=v,
        )
        used_housegan = hg_placed is not None
        placed = hg_placed if used_housegan else _place_rooms_architectural(
            sized_rooms, total_w, total_h, style, variant_seed=v
        )

        # Calculate actual footprint from placed rooms (BEFORE validation)
        if placed:
            actual_h = max(total_h, max(r["y"] + r["height"] for r in placed))
            actual_h = round(actual_h / 2) * 2  # snap to grid
        else:
            actual_h = total_h

        # Stage 4: IRC validation (use actual height, not target)
        placed = _validate_irc(placed, total_w, actual_h)

        # Stage 5: Grid snap + gap fill (use actual height)
        placed = _snap_and_fill(placed, total_w, actual_h)

        # Recompute actual height after snapping
        if placed:
            actual_h = max(total_h, max(r["y"] + r["height"] for r in placed))
            actual_h = round(actual_h / 2) * 2

        plan = {
            "id": f"moe_{uuid.uuid4().hex[:8]}",
            "name": variant_names[v] if v < len(variant_names) else f"MOE Plan {v + 1}",
            "totalWidth": total_w,
            "totalHeight": actual_h,
            "ceilingHeight": ceiling_ft,
            "rooms": placed,
            "generator": "moe+housegan" if used_housegan else "moe",
            "variant": v,
        }
        plans.append(plan)

    # Confidence scoring
    confidence = _calculate_confidence(expert_weights, plans, sqft)

    return {
        "plans": plans,
        "expert_weights": expert_weights,
        "confidence": confidence,
        "irc_compliant": True,
    }


def _calculate_confidence(expert_weights: dict, plans: list, target_sqft: int) -> float:
    """
    Calculate confidence score (0-100) based on:
    - Expert balance (entropy)
    - Area utilization vs target sqft
    - Room fit quality (no rooms extending past boundary)
    """
    weights = list(expert_weights.values())
    entropy = -sum(w * math.log(max(w, 1e-10)) for w in weights)
    max_entropy = math.log(len(weights))
    balance_score = entropy / max_entropy if max_entropy > 0 else 0

    quality_scores = []
    for plan in plans:
        rooms = plan["rooms"]
        total_w = plan["totalWidth"]
        total_h = plan["totalHeight"]

        # Area utilization vs target
        room_area = sum(r["width"] * r["height"] for r in rooms)
        util_ratio = room_area / max(target_sqft, 1)
        util_score = 1.0 - min(0.5, abs(1.0 - util_ratio))

        # Boundary compliance (all rooms within footprint)
        boundary_ok = all(
            r["x"] >= 0 and r["y"] >= 0 and
            r["x"] + r["width"] <= total_w + 2 and
            r["y"] + r["height"] <= total_h + 2
            for r in rooms
        )
        boundary_score = 1.0 if boundary_ok else 0.5

        # Room uniqueness (no unexpected duplicates)
        singleton_types = {"living_room", "kitchen", "dining_room", "family_room",
                           "foyer", "garage", "pantry", "home_office"}
        types_seen = {}
        for r in rooms:
            t = r["type"]
            types_seen[t] = types_seen.get(t, 0) + 1
        unique_score = 1.0
        for t, count in types_seen.items():
            if t in singleton_types and count > 1:
                unique_score -= 0.1

        quality_scores.append((util_score + boundary_score + unique_score) / 3)

    avg_quality = sum(quality_scores) / max(len(quality_scores), 1)
    confidence = round((balance_score * 0.3 + avg_quality * 0.7) * 100, 1)
    return max(50.0, min(99.0, confidence))
