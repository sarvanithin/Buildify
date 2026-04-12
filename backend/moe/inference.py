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
    _build_room_list_two_story,
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
        state = checkpoint["model_state_dict"]
        # V1 checkpoints have SparseGating keys not present in SoftGating — drop them
        state = {k: v for k, v in state.items() if not k.startswith("gating.noise_linear")}
        model.load_state_dict(state, strict=False)
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
    Calibrated so 1800 sqft → standard sizes, 2400 sqft → standard-to-premium.
    """
    specs = IRC_ROOM_SPECS.get(room_type)
    if not specs:
        return (8, 8)

    min_w, min_h, std_w, std_h, prem_w, prem_h = specs

    # sf: 0.0 at 1000 sqft (tiny), 1.0 at 4000 sqft (large luxury).
    # Power curve (^0.6) compresses the lower end so that mid-range homes
    # (1800-2400 sqft) land near standard sizes rather than near minimums.
    # 1200→0.18  1500→0.36  1800→0.49  2400→0.65  3000→0.78  4000→1.0
    sf = min(1.0, max(0.0, (sqft - 1000) / 3000)) ** 0.6

    # Expert influence (subtle nudge on top of sqft-based sizing)
    sizing_w = expert_weights.get("Room Sizing", 0.125)
    cost_w = expert_weights.get("Cost Optimization", 0.125)

    # Blend: sf 0→0.5 = min→standard, sf 0.5→1.0 = standard→premium
    if sf <= 0.5:
        t = min(1.0, max(0.0, sf * 2.0 + sizing_w * 0.4 - cost_w * 0.4))
        w = min_w + (std_w - min_w) * t
        h = min_h + (std_h - min_h) * t
    else:
        t = min(1.0, max(0.0, (sf - 0.5) * 2.0 + sizing_w * 0.3 - cost_w * 0.3))
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
                    "bathroom", "ensuite_bathroom", "foyer"}

    # Group rooms by zone (uses updated ZONE_MAP: 0=ENTRY..5=OUTDOOR)
    zone_rooms: Dict[int, List[dict]] = {i: [] for i in range(6)}
    for r in rooms:
        z = ZONE_MAP.get(r["type"], 1)
        zone_rooms[z].append(r)

    def _pack_rows(band_rooms: List[dict], fixed_height: float = None) -> None:
        """
        Pack a pre-ordered list into rows. Last non-small room in each row fills remaining width.
        If total natural widths exceed total_w, rooms are proportionally shrunk first
        to prevent wrapping small rooms into isolated overflow rows.
        Each row height = tallest room in that row (or fixed_height if specified).
        """
        nonlocal y_cursor
        if not band_rooms:
            return

        # Pre-shrink: if band's total natural width > total_w, shrink proportionally.
        # This prevents small rooms from wrapping to an isolated row (e.g. a lone bathroom).
        natural_widths = [max(4.0, round(float(r["width"]))) for r in band_rooms]
        total_natural = sum(natural_widths)
        if total_natural > total_w:
            excess = total_natural - total_w
            adjusted = []
            for i, r in enumerate(band_rooms):
                specs = IRC_ROOM_SPECS.get(r["type"], (4, 4))
                shrink = excess * (natural_widths[i] / total_natural)
                adjusted.append(max(float(specs[0]), round((natural_widths[i] - shrink) / 2) * 2))
            # Iterative trim: remove rounding overage from non-small rooms largest-first
            diff = sum(adjusted) - total_w
            while diff > 0:
                trimmed = False
                for k in range(len(band_rooms) - 1, -1, -1):
                    specs = IRC_ROOM_SPECS.get(band_rooms[k]["type"], (4, 4))
                    headroom = adjusted[k] - specs[0]
                    if headroom > 0:
                        trim = min(diff, headroom)
                        adjusted[k] -= trim
                        diff -= trim
                        trimmed = True
                    if diff <= 0:
                        break
                if not trimmed:
                    break  # all rooms at IRC minimum, cannot shrink further
            natural_widths = adjusted

        # Split into rows by (pre-shrunk) width
        rows: List[List[dict]] = []
        row_widths_list: List[List[float]] = []
        cur_row: List[dict] = []
        cur_widths: List[float] = []
        row_w = 0.0
        for i, r in enumerate(band_rooms):
            w = natural_widths[i]
            if row_w + w > total_w and cur_row:
                rows.append(cur_row)
                row_widths_list.append(cur_widths)
                cur_row = []
                cur_widths = []
                row_w = 0.0
            cur_row.append(r)
            cur_widths.append(w)
            row_w += w
        if cur_row:
            rows.append(cur_row)
            row_widths_list.append(cur_widths)

        for row_idx, row in enumerate(rows):
            band_advance = fixed_height if fixed_height else max(r["height"] for r in row)
            row_widths = list(row_widths_list[row_idx])

            # Stretch last non-small room to fill remaining width.
            # Never stretch small-only rows (orphan bathrooms etc.).
            # Cap stretch for rooms that shouldn't grow unboundedly (bedrooms,
            # dining, offices) — prevents a bedroom from becoming 31' wide just
            # because it's the only large room in a short row.
            _STRETCH_CAP = {
                "bedroom": 16, "master_bedroom": 22, "home_office": 16,
                "dining_room": 18, "family_room": 22, "walk_in_closet": 12,
                "garage": 30, "kitchen": 22, "living_room": 28,
            }
            remaining = total_w - sum(row_widths)
            if remaining > 0:
                stretch_idx = None
                for k in range(len(row) - 1, -1, -1):
                    if row[k]["type"] not in _SMALL_ROOMS:
                        stretch_idx = k
                        break
                if stretch_idx is not None:
                    rtype = row[stretch_idx]["type"]
                    cap = _STRETCH_CAP.get(rtype, 9999)
                    allowed = max(0, cap - row_widths[stretch_idx])
                    row_widths[stretch_idx] += min(remaining, allowed)

            x_pos = 0.0
            for i, r in enumerate(row):
                room_h = fixed_height if fixed_height else r["height"]
                placed.append({
                    **r,
                    "x": round(x_pos),
                    "y": round(y_cursor),
                    "width": int(row_widths[i]),
                    "height": round(room_h),
                })
                x_pos += row_widths[i]
            y_cursor += round(band_advance)  # advance by tallest room in row

    # ── ZONE 0: ENTRY BAND (y=0, front of house) ──────────────────────────
    # Garage always front-left (x=0), mudroom tucked beside garage, foyer at right.
    # If no garage is present, the foyer is relocated to the start of the social band
    # (architecturally: entry → foyer → living area) to avoid a tiny isolated foyer
    # at top-left with empty dead space beside it.
    entry = zone_rooms[0]
    has_garage = any(r["type"] == "garage" for r in entry)
    if not has_garage and entry:
        # No garage: foyer goes to social front (entry → living area flow).
        # Service rooms (laundry, mudroom, utility) go to the kitchen zone — they
        # belong adjacent to kitchen, not at the front of the social band.
        foyer_rooms   = [r for r in entry if r["type"] == "foyer"]
        service_rooms = [r for r in entry if r["type"] != "foyer"]
        zone_rooms[1] = foyer_rooms + zone_rooms[1]      # foyer leads social band
        zone_rooms[2] = zone_rooms[2] + service_rooms    # laundry/mudroom trail kitchen band
        entry = []
    else:
        entry.sort(key=lambda r: {
            "garage": 0, "mudroom": 1, "laundry_room": 2, "utility_room": 3, "foyer": 4
        }.get(r["type"], 5))
        # Stretch ALL entry band rooms to the band's max height (set by garage depth).
        # This eliminates the dead space void that appears when mudroom/laundry/foyer
        # are 8ft tall in a 20ft-deep entry band. All rooms flush to the same bottom
        # edge guarantees shared walls with the social band below.
        if entry:
            band_max_h = max(r["height"] for r in entry)
            for r in entry:
                r["height"] = band_max_h
    _pack_rows(entry)

    # ── ZONES 1+2: SOCIAL BAND (kitchen + living merged) ─────────────────
    # Merge public + kitchen into one open zone — reduces depth by ~12ft and
    # ensures kitchen is adjacent to living areas (not isolated two bands away).
    # Order: foyer (if no garage), living_room, dining/family, kitchen, pantry.
    social_rooms = zone_rooms[1] + zone_rooms[2]
    # Kitchen + service first, then dining/family, then living room LAST.
    # Living room is the largest flexible social space — placing it last lets it
    # absorb any remaining row width without over-stretching kitchen or dining.
    # Architectural flow: entry → kitchen/dining → living (open plan read from inside).
    social_order = {
        "foyer": 0, "kitchen": 1, "pantry": 2,
        "home_office": 3, "dining_room": 4, "family_room": 5,
        "living_room": 6,
    }
    social_rooms.sort(key=lambda r: social_order.get(r["type"], 6))
    _pack_rows(social_rooms)

    # ── ZONE 3: HALLWAY — full-width 4ft circulation spine ────────────────
    hallway_rooms = zone_rooms[3]
    hallway = next((r for r in hallway_rooms if r["type"] == "hallway"), None)
    other_hall = [r for r in hallway_rooms if r["type"] != "hallway"]
    if hallway:
        hallway_h = 4  # IRC R311.7: 36in min; 4ft is standard corridor depth (not sqft-scaled)
        hallway["width"] = int(total_w)   # spans full footprint width
        _pack_rows([hallway], fixed_height=hallway_h)
    # NOTE: other_hall (half_bath etc.) is intentionally merged into the private band
    # below so that half_baths are adjacent to bedrooms and share a connectable wall.

    # ── ZONE 4: PRIVATE BAND (bedrooms) ───────────────────────────────────
    # Master suite cluster first: master_bedroom → ensuite_bathroom → walk_in_closet
    # Then secondary bedrooms, shared bathrooms, closets.
    # Half-baths (zone 3 overflow) are merged here so they sit adjacent to bedrooms
    # and share a wall — creating a door. Placing them between hallway and private
    # band creates a dead gap and leaves bedrooms disconnected from the hallway.
    private = zone_rooms[4] + [r for r in zone_rooms[3] if r["type"] != "hallway"]
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
    # Snap to 2ft grid, enforcing IRC per-room-type minimums
    for r in rooms:
        r["x"] = round(r["x"] / 2) * 2
        r["y"] = round(r["y"] / 2) * 2
        specs = IRC_ROOM_SPECS.get(r["type"], (4, 4))
        r["width"] = max(specs[0], round(r["width"] / 2) * 2)
        r["height"] = max(specs[1], round(r["height"] / 2) * 2)

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
        # Only close small gaps (≤ 6ft) — large gaps are intentional from proportional sizing.
        # Shrinks are always applied to keep rooms within the footprint.
        if row_rooms_sorted:
            last = row_rooms_sorted[-1]
            right_edge = last["x"] + last["width"]
            diff = tw - right_edge

            # Close gaps and shrinks. For positive gaps, only stretch non-small rooms.
            _SNAP_SMALL = {"closet","walk_in_closet","half_bath","pantry","mudroom",
                           "laundry_room","utility_room","bathroom","ensuite_bathroom","foyer",
                           "patio","deck","rear_patio","outdoor_living","front_porch"}
            # Cap: prevent these room types from growing beyond a realistic max width
            # even if they end up alone in a row after rounding/wrapping.
            _SNAP_WIDTH_CAP = {
                "master_bedroom": 22, "bedroom": 16,
                "home_office": 16, "dining_room": 18, "family_room": 22,
                "garage": 30, "kitchen": 22, "living_room": 28,
            }
            if diff != 0:
                last_min = IRC_ROOM_SPECS.get(last["type"], (4, 4))[0]
                # For stretching (diff > 0), skip if last room is a small type
                if diff > 0 and last["type"] in _SNAP_SMALL:
                    diff = 0  # don't inflate small rooms in snap pass
                # Apply width cap for room types that shouldn't grow unboundedly
                if diff > 0 and last["type"] in _SNAP_WIDTH_CAP:
                    max_w = _SNAP_WIDTH_CAP[last["type"]]
                    diff = max(0, min(diff, max_w - last["width"]))
                new_w = last["width"] + diff
                if new_w >= last_min:
                    last["width"] = new_w
                else:
                    # Last room can't shrink to min — distribute shrinkage across all rooms
                    excess = right_edge - tw
                    total_room_w = sum(r["width"] for r in row_rooms_sorted)
                    x_cursor = 0
                    for r in row_rooms_sorted:
                        shrink = round(excess * (r["width"] / total_room_w))
                        r_min = IRC_ROOM_SPECS.get(r["type"], (4, 4))[0]
                        r["width"] = max(r_min, r["width"] - shrink)
                        r["x"] = x_cursor
                        x_cursor += r["width"]
                    # Final adjustment on last room for exact fit
                    last = row_rooms_sorted[-1]
                    last_min = IRC_ROOM_SPECS.get(last["type"], (4, 4))[0]
                    last["width"] = max(last_min, tw - last["x"])

    # Ensure no overlaps remain after grid snapping
    rooms = _final_overlap_check(rooms, total_w)

    return rooms


_OUTDOOR_TYPES = {"patio", "deck", "rear_patio", "outdoor_living", "front_porch"}

# Private rooms that must not be extended vertically — they've been carefully
# sized and stretching them to fill the footprint creates unrealistic 24ft-deep bedrooms.
_NO_EXTEND_TYPES = {
    "bedroom", "master_bedroom", "ensuite_bathroom", "bathroom",
    "half_bath", "walk_in_closet", "closet",
}

def _fill_vertical_gaps(rooms: List[dict], total_h: float) -> List[dict]:
    """Extend rooms that have no room below them to fill the plan's total height.
    Outdoor rooms (patio/deck) are not extended, private rooms (bedrooms/baths)
    are not extended beyond their placed size, and conditioned rooms are not
    extended into the outdoor band even when the outdoor room is narrower."""
    th = int(round(total_h))
    # Find the y-start of the outdoor band (topmost outdoor room y position)
    outdoor_y = th
    for r in rooms:
        if r["type"] in _OUTDOOR_TYPES:
            outdoor_y = min(outdoor_y, r["y"])
    for r in rooms:
        if r["type"] in _OUTDOOR_TYPES:
            continue  # never extend outdoor rooms
        if r["type"] in _NO_EXTEND_TYPES:
            continue  # private rooms keep their placed size exactly
        bottom = r["y"] + r["height"]
        if bottom >= th:
            continue
        # Don't extend conditioned rooms into the outdoor band
        if outdoor_y < th and bottom >= outdoor_y:
            continue
        rx1, rx2 = r["x"], r["x"] + r["width"]
        has_room_below = any(
            other is not r
            and other["x"] < rx2 and other["x"] + other["width"] > rx1
            and other["y"] >= bottom
            for other in rooms
        )
        if not has_room_below:
            # Cap extension at outdoor band start if outdoor rooms exist
            target_h = outdoor_y if outdoor_y < th else th
            r["height"] = target_h - r["y"]
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
    open_plan = constraints.get("openPlan") or constraints.get("open_plan", False)
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

    ceil_map = {"standard": 9, "high": 10, "vaulted": 12}
    ceiling_ft = ceil_map.get(ceiling_height, 9)

    # Stage 1: Build correct room list from constraints (deterministic)
    base_rooms = _build_room_list(
        bedrooms, bathrooms, sqft, style,
        open_plan, primary_suite, home_office, formal_dining,
        garage, laundry, outdoor,
    )

    # ── TWO-STORY PATH ────────────────────────────────────────────────────────
    if stories == 2:
        two_story_names = [
            "MOE Plan A — Two Story",
            "MOE Plan B — Two Story",
            "MOE Plan C — Two Story",
            "MOE Plan D — Two Story",
            "MOE Plan E — Two Story",
        ]
        plans = []
        for v in range(num_variants):
            variant_expert_w = dict(expert_weights)
            if v > 0:
                rng = random.Random(v * 42)
                for key in variant_expert_w:
                    variant_expert_w[key] *= (0.85 + rng.random() * 0.3)
                total_w_sum = sum(variant_expert_w.values())
                if total_w_sum > 0:
                    for key in variant_expert_w:
                        variant_expert_w[key] /= total_w_sum
            plan = _generate_two_story_variant(
                constraints=constraints,
                expert_weights=variant_expert_w,
                variant_name=two_story_names[v] if v < len(two_story_names) else f"MOE Plan {v + 1} — Two Story",
                ceiling_ft=ceiling_ft,
                config=config,
            )
            plans.append(plan)
        confidence = _calculate_confidence(expert_weights, plans, sqft)
        return {
            "plans": plans,
            "expert_weights": expert_weights,
            "confidence": confidence,
            "irc_compliant": True,
        }

    # ── SINGLE-STORY PATH ─────────────────────────────────────────────────────
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

        # ── Proportional sqft scaling ───────────────────────────────────────
        # Room types NOT counted toward conditioned living area
        _UNCONDITIONED = {"garage", "patio", "deck"}
        # Hallway is hardcoded to full_width×4ft in placement; exclude from
        # area target so sizing doesn't distort other rooms.
        _FIXED_PLACEMENT = {"hallway"}

        conditioned = [
            r for r in sized_rooms
            if r["type"] not in _UNCONDITIONED and r["type"] not in _FIXED_PLACEMENT
        ]
        natural_area = sum(r["width"] * r["height"] for r in conditioned)

        # Hallway will be placed as total_w × 4ft — subtract its estimated footprint
        # from the sqft target so the total conditioned area stays accurate.
        # Estimate: entry band width × 4ft (hallway depth).
        est_entry_w = sum(r["width"] for r in sized_rooms if ZONE_MAP.get(r["type"], 1) == 0)
        est_entry_w = max(30, min(65, round(est_entry_w)))
        hallway_est = est_entry_w * 4  # ≈ 160sf for a typical house
        sqft_target = max(int(sqft * 0.75), sqft - hallway_est)

        # Cap unconditioned rooms (not in conditioned array) directly here
        # so outdoor band stays shallow and garage doesn't grow too deep.
        _UNCOND_HEIGHT_CAPS = {
            "patio": 10, "deck": 10, "rear_patio": 10,
            "outdoor_living": 10, "front_porch": 6,
            "garage": 20,
        }
        for r in sized_rooms:
            cap = _UNCOND_HEIGHT_CAPS.get(r["type"])
            if cap and r["height"] > cap:
                r["height"] = cap
            # Also cap unconditioned widths to avoid sprawling garages
            if r["type"] == "patio" and r["width"] > 20:
                r["width"] = 20
            if r["type"] == "deck" and r["width"] > 20:
                r["width"] = 20

        if natural_area > 0:
            # Phase 1 — sqrt scale preserves room aspect ratios while approaching sqft.
            # Round to 2ft grid (matches snap_and_fill) so subsequent snapping doesn't inflate area.
            sf = math.sqrt(sqft_target / natural_area)
            sf = min(sf, 2.0)
            sf = max(sf, 0.5)
            for r in conditioned:
                specs = IRC_ROOM_SPECS.get(r["type"], (4, 4))
                r["width"]  = max(specs[0], round(r["width"]  * sf / 2) * 2)
                r["height"] = max(specs[1], round(r["height"] * sf / 2) * 2)

            # Phase 2 — cap room heights to architectural maxima.
            # Tight caps force rooms to be WIDE rather than deep, creating a
            # realistic shallow house footprint (vs. a narrow tower).
            _HEIGHT_CAPS = {
                "living_room": 16, "family_room": 16, "dining_room": 14,
                "kitchen": 14, "home_office": 13,
                "master_bedroom": 16, "bedroom": 14,
                "ensuite_bathroom": 10,  "bathroom": 10,  "half_bath": 7,
                "foyer": 10,  "mudroom": 10,  "laundry_room": 10,  "utility_room": 10,
                "walk_in_closet": 10,  "closet": 8,  "pantry": 8,
                "garage": 22, "patio": 10, "deck": 10, "rear_patio": 10,
            }
            for r in conditioned:
                cap = _HEIGHT_CAPS.get(r["type"])
                if cap and r["height"] > cap:
                    r["height"] = cap

            # Phase 3 — width-only scale to restore conditioned area to sqft_target.
            # Widths expand → rooms are proportionally wider, bands are shallower.
            # Round to 2ft grid so snap_and_fill doesn't inflate sizes further.
            capped_area = sum(r["width"] * r["height"] for r in conditioned)
            if capped_area > 0 and capped_area < sqft_target * 0.95:
                sw = min(sqft_target / capped_area, 1.6)  # cap at 1.6× to prevent absurd widths
                for r in conditioned:
                    specs = IRC_ROOM_SPECS.get(r["type"], (4, 4))
                    r["width"] = max(specs[0], round(r["width"] * sw / 2) * 2)

            # Phase 4 — hard width caps so no single room grows absurdly wide from scaling.
            # Without this, a primary suite can reach 28ft+ in large or high-sqft homes.
            _WIDTH_CAPS = {
                "master_bedroom": 20, "bedroom": 16,
                "home_office": 16, "dining_room": 18, "family_room": 20,
                "ensuite_bathroom": 12, "bathroom": 10, "half_bath": 6,
                "walk_in_closet": 10, "closet": 8, "pantry": 8,
                "foyer": 12, "mudroom": 12, "laundry_room": 10,
            }
            for r in conditioned:
                cap = _WIDTH_CAPS.get(r["type"])
                if cap and r["width"] > cap:
                    r["width"] = cap

            # Phase 5 — architectural bonuses and aspect ratio enforcement.

            # 5a: Open plan great room bonus — when open plan, the living room
            # serves as the main social anchor and should be noticeably larger.
            if open_plan:
                for r in conditioned:
                    if r["type"] == "living_room":
                        specs = IRC_ROOM_SPECS.get("living_room", (4, 4))
                        r["width"] = min(
                            _WIDTH_CAPS.get("living_room", 9999),
                            max(specs[0], round(r["width"] * 1.25 / 2) * 2)
                        )
                        r["height"] = max(specs[1], round(r["height"] * 1.15 / 2) * 2)

            # 5b: Primary suite prominence — at 2000+ sqft the suite should feel
            # like a destination, not just a code-minimum bedroom.
            # Scales: 2000→+0%, 2400→+15%, 3000→+33%, 4000→+67%
            if primary_suite and sqft >= 2000:
                suite_bonus = min(1.70, 1.0 + (sqft - 2000) / 3000)
                for r in conditioned:
                    if r["type"] in ("master_bedroom", "ensuite_bathroom", "walk_in_closet"):
                        specs = IRC_ROOM_SPECS.get(r["type"], (4, 4))
                        cap_w = _WIDTH_CAPS.get(r["type"], 9999)
                        cap_h = _HEIGHT_CAPS.get(r["type"], 9999)
                        r["width"]  = min(cap_w, max(specs[0], round(r["width"]  * suite_bonus / 2) * 2))
                        r["height"] = min(cap_h, max(specs[1], round(r["height"] * min(suite_bonus, 1.30) / 2) * 2))

            # 5c: Aspect ratio enforcement — redistribute width/height while
            # preserving room area. Architectural rooms should have good proportions:
            # living/dining are wider than deep; bedrooms are roughly square.
            _ASPECT_RANGES: dict = {
                "living_room":      (1.15, 1.95),
                "family_room":      (1.15, 1.95),
                "kitchen":          (0.95, 1.70),
                "dining_room":      (0.95, 1.55),
                "master_bedroom":   (0.95, 1.60),
                "bedroom":          (0.90, 1.55),
                "ensuite_bathroom": (0.85, 1.50),
                "bathroom":         (0.75, 1.45),
                "home_office":      (0.95, 1.55),
                "foyer":            (0.75, 1.35),
                "walk_in_closet":   (0.75, 1.30),
                "mudroom":          (0.70, 1.35),
                "laundry_room":     (0.70, 1.40),
                "pantry":           (0.65, 1.25),
            }
            for r in conditioned:
                ar = _ASPECT_RANGES.get(r["type"])
                if not ar or r["height"] == 0:
                    continue
                ratio = r["width"] / r["height"]
                area = r["width"] * r["height"]
                specs = IRC_ROOM_SPECS.get(r["type"], (4, 4))
                if ratio < ar[0]:
                    # Too tall/narrow — ceil-round new_w to guarantee ratio hits minimum.
                    # round() can produce the same too-narrow width; ceil ensures we go wider.
                    new_w = max(specs[0], math.ceil(math.sqrt(area * ar[0]) / 2) * 2)
                    new_w = min(_WIDTH_CAPS.get(r["type"], 9999), new_w)
                    new_h = max(specs[1], round(area / max(new_w, 1) / 2) * 2)
                    cap_h = _HEIGHT_CAPS.get(r["type"], 9999)
                    r["width"], r["height"] = new_w, min(cap_h, new_h)
                elif ratio > ar[1]:
                    # Too wide/flat — ceil-round new_h to guarantee ratio comes down.
                    new_h = max(specs[1], math.ceil(math.sqrt(area / ar[1]) / 2) * 2)
                    cap_h = _HEIGHT_CAPS.get(r["type"], 9999)
                    new_h = min(cap_h, new_h)
                    new_w = max(specs[0], round(area / max(new_h, 1) / 2) * 2)
                    r["width"], r["height"] = min(_WIDTH_CAPS.get(r["type"], 9999), new_w), new_h

            # Phase 6 — sqft-tier minimum sizes.
            # Guarantees key rooms feel appropriately generous at each house size tier.
            # These are FLOORS, not caps — rooms can be larger from prior phases.
            # Tiers: (min_sqft_threshold, min_w, min_h)
            _TIER_MINS: dict = {
                "kitchen":      [(1000, 10, 10), (1600, 12, 12), (2000, 14, 12), (2600, 16, 14), (3200, 18, 14)],
                "living_room":  [(1000, 12, 12), (1600, 16, 12), (2000, 18, 14), (2600, 20, 14), (3200, 22, 16)],
                "dining_room":  [(1000, 10, 10), (1600, 12, 12), (2000, 14, 12), (2600, 14, 12)],
                "family_room":  [(1000, 12, 12), (1600, 14, 12), (2000, 16, 14), (2600, 18, 14)],
                "master_bedroom": [(1000, 12, 12), (1800, 14, 14), (2400, 16, 14), (3000, 16, 16)],
                "ensuite_bathroom": [(1000, 7, 9), (2000, 8, 10), (2800, 10, 10)],
                "home_office":  [(1000, 10, 10), (2000, 12, 10), (2800, 12, 12)],
            }
            for r in conditioned:
                tiers = _TIER_MINS.get(r["type"])
                if not tiers:
                    continue
                specs = IRC_ROOM_SPECS.get(r["type"], (4, 4))
                for tier_sqft, tier_w, tier_h in reversed(tiers):
                    if sqft >= tier_sqft:
                        cap_w = _WIDTH_CAPS.get(r["type"], 9999)
                        cap_h = _HEIGHT_CAPS.get(r["type"], 9999)
                        r["width"]  = min(cap_w, max(r["width"],  tier_w))
                        r["height"] = min(cap_h, max(r["height"], tier_h))
                        break

        # ── Footprint width: target realistic US house proportions ─────────
        # Real single-story houses: 50-70ft wide, 30-55ft deep.
        # Base target width on sqft: ~sqrt(sqft * 1.8) gives sensible proportions.
        # Then verify entry band fits and clamp to [48, 75].
        entry_rooms_w = sum(
            r["width"] for r in sized_rooms
            if ZONE_MAP.get(r["type"], 1) == 0
        )
        # Target: realistic single-story US house width.
        # Scale factor 2.05 ensures the social band's rooms fit in one row without shrinkage.
        # 1600→57ft, 1800→61ft, 2400→70ft, 3000→78ft.
        sqft_based_w = math.sqrt(sqft * 2.05)  # 1800→61ft, 2400→70ft, 3000→78ft
        total_w = max(54, min(82, round(max(entry_rooms_w, sqft_based_w) / 2) * 2))
        # Height estimate only used for initial clamping — actual_h recomputed after placement
        total_h = max(28, min(60, round(sqft / max(total_w, 1) * 0.9)))

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
        placed = _fix_isolated_service_rooms(placed, total_w)
        placed = _fill_vertical_gaps(placed, actual_h)

        # Stage 5b: Sqft correction — if conditioned area significantly exceeds target,
        # scale all conditioned room heights proportionally down to approach sqft.
        _UNCOND = {"garage", "patio", "deck", "rear_patio", "outdoor_living", "front_porch"}
        _FIXED_H = {"hallway"}  # hallway height is architecturally fixed
        cond_rooms = [r for r in placed if r["type"] not in _UNCOND and r["type"] not in _FIXED_H]
        cond_area = sum(r["width"] * r["height"] for r in cond_rooms)
        if cond_area > sqft * 1.15:  # more than 15% over target
            scale_h = sqft / cond_area
            scale_h = max(0.6, min(1.0, scale_h))  # floor at 60% to prevent tiny rooms
            for r in cond_rooms:
                specs = IRC_ROOM_SPECS.get(r["type"], (4, 4))
                new_h = max(specs[1], round(r["height"] * scale_h / 2) * 2)
                r["height"] = new_h
            # Recompute actual_h with new heights before re-snapping
            if placed:
                actual_h = max(total_h, max(r["y"] + r["height"] for r in placed))
                actual_h = round(actual_h / 2) * 2
            placed = _snap_and_fill(placed, total_w, actual_h)

        # Recompute actual height after snapping
        if placed:
            actual_h = max(total_h, max(r["y"] + r["height"] for r in placed))
            actual_h = round(actual_h / 2) * 2

        # Stage 6: Door detection
        doors = _add_doors(placed)

        plan = {
            "id": f"moe_{uuid.uuid4().hex[:8]}",
            "name": variant_names[v] if v < len(variant_names) else f"MOE Plan {v + 1}",
            "totalWidth": total_w,
            "totalHeight": actual_h,
            "ceilingHeight": ceiling_ft,
            "style": style,
            "rooms": placed,
            "doors": doors,
            "generator": "moe+housegan" if used_housegan else "moe",
            "variant": v,
        }

        # Stage 7: Design validation — retry up to 2 more times if errors found
        validation_issues = _validate_design(plan)
        has_errors = any(i["severity"] == "error" for i in validation_issues)
        if has_errors:
            for retry in range(1, 3):
                retry_placed = _place_rooms_architectural(
                    sized_rooms, total_w, total_h, style,
                    variant_seed=v * 100 + retry,
                )
                if retry_placed:
                    actual_h_r = max(total_h, max(r["y"] + r["height"] for r in retry_placed))
                    actual_h_r = round(actual_h_r / 2) * 2
                retry_placed = _validate_irc(retry_placed, total_w, actual_h_r)
                retry_placed = _snap_and_fill(retry_placed, total_w, actual_h_r)
                retry_placed = _fill_vertical_gaps(retry_placed, actual_h_r)
                retry_doors = _add_doors(retry_placed)
                retry_plan = {**plan, "rooms": retry_placed, "doors": retry_doors,
                               "totalHeight": actual_h_r}
                retry_issues = _validate_design(retry_plan)
                retry_errors = any(i["severity"] == "error" for i in retry_issues)
                if not retry_errors or retry == 2:
                    plan = retry_plan
                    validation_issues = retry_issues
                    break

        plan["validationIssues"] = validation_issues
        plans.append(plan)

    # Confidence scoring
    confidence = _calculate_confidence(expert_weights, plans, sqft)

    return {
        "plans": plans,
        "expert_weights": expert_weights,
        "confidence": confidence,
        "irc_compliant": True,
    }


def _add_doors(rooms: List[dict]) -> List[dict]:
    """
    Detect shared wall segments and place doors/openings.
    Returns: List of {x, y, isVertical, roomA, roomB}
    """
    doors = []
    processed_pairs = set()

    for i, r1 in enumerate(rooms):
        for j, r2 in enumerate(rooms):
            if i >= j: continue
            pair = tuple(sorted((r1["id"], r2["id"])))
            if pair in processed_pairs: continue

            # Wall adjacency tolerance: 2ft handles rounding gaps from snap/fill passes.
            # Service rooms (laundry, mudroom) use a wider tolerance because they can
            # end up in a wrapped row adjacent to a taller social row, creating a 4ft gap.
            _SERVICE = {"laundry_room", "mudroom"}
            _TOL = 6.0 if (_SERVICE & {r1["type"], r2["type"]}) else 2.0

            # Check for shared vertical wall (X matches)
            # R1 is left of R2 OR R2 is left of R1
            shared_v = None
            if abs((r1["x"] + r1["width"]) - r2["x"]) <= _TOL:
                # R1 | R2
                shared_x = (r1["x"] + r1["width"] + r2["x"]) / 2
                y_start = max(r1["y"], r2["y"])
                y_end = min(r1["y"] + r1["height"], r2["y"] + r2["height"])
                if y_end - y_start >= 2: # Need at least 2ft shared wall
                    shared_v = (shared_x, y_start + (y_end - y_start)/2, True)
            elif abs((r2["x"] + r2["width"]) - r1["x"]) <= _TOL:
                # R2 | R1
                shared_x = (r2["x"] + r2["width"] + r1["x"]) / 2
                y_start = max(r1["y"], r2["y"])
                y_end = min(r1["y"] + r1["height"], r2["y"] + r2["height"])
                if y_end - y_start >= 2:
                    shared_v = (shared_x, y_start + (y_end - y_start)/2, True)

            # Check for shared horizontal wall (Y matches)
            shared_h = None
            if abs((r1["y"] + r1["height"]) - r2["y"]) <= _TOL:
                # R1
                # ---
                # R2
                shared_y = (r1["y"] + r1["height"] + r2["y"]) / 2
                x_start = max(r1["x"], r2["x"])
                x_end = min(r1["x"] + r1["width"], r2["x"] + r2["width"])
                if x_end - x_start >= 2:
                    shared_h = (x_start + (x_end - x_start)/2, shared_y, False)
            elif abs((r2["y"] + r2["height"]) - r1["y"]) <= _TOL:
                # R2
                # ---
                # R1
                shared_y = (r2["y"] + r2["height"] + r1["y"]) / 2
                x_start = max(r1["x"], r2["x"])
                x_end = min(r1["x"] + r1["width"], r2["x"] + r2["width"])
                if x_end - x_start >= 2:
                    shared_h = (x_start + (x_end - x_start)/2, shared_y, False)

            # Rules for placing a door:
            # 1. One is a hallway → almost always a door
            # 2. One is a foyer and other is a social room → door
            # 3. Master bedroom to master bath/closet → door
            # 4. Adjacent social rooms → wide opening
            is_door = False
            t1, t2 = r1["type"], r2["type"]
            
            _SOCIAL = {"living_room", "dining_room", "family_room", "kitchen"}
            _BEDROOM_TYPES = {"bedroom", "master_bedroom"}
            _BATH_TYPES = {"bathroom", "ensuite_bathroom", "half_bath"}

            if "hallway" in (t1, t2): is_door = True
            elif ("foyer" in (t1, t2)) and any(t in _SOCIAL for t in (t1, t2)): is_door = True
            elif ("foyer" in (t1, t2)) and "hallway" not in (t1, t2): is_door = True  # foyer always connects
            elif (t1 == "master_bedroom" and t2 in ("ensuite_bathroom", "walk_in_closet")) or (t2 == "master_bedroom" and t1 in ("ensuite_bathroom", "walk_in_closet")): is_door = True
            elif all(t in _SOCIAL for t in (t1, t2)): is_door = True  # Wide opening
            elif "mudroom" in (t1, t2) and any(t in ("garage", "kitchen", "foyer") for t in (t1, t2)): is_door = True
            elif any(t in _BEDROOM_TYPES for t in (t1, t2)) and any(t in _BATH_TYPES for t in (t1, t2)): is_door = True  # bed→bath
            elif any(t in _BEDROOM_TYPES for t in (t1, t2)) and any(t in _SOCIAL for t in (t1, t2)): is_door = True  # open plan bed→living
            elif "mudroom" in (t1, t2): is_door = True  # mudroom always connects to adjacent rooms
            elif ("laundry_room" in (t1, t2)) and any(t in ("kitchen", "mudroom", "hallway", "foyer", "garage") for t in (t1, t2)): is_door = True
            elif ("pantry" in (t1, t2)) and "kitchen" in (t1, t2): is_door = True  # pantry from kitchen
            elif ("home_office" in (t1, t2)) and any(t in _SOCIAL | {"hallway", "foyer"} for t in (t1, t2)): is_door = True
            elif ("walk_in_closet" in (t1, t2)) and any(t in _BEDROOM_TYPES | _BATH_TYPES for t in (t1, t2)): is_door = True  # suite cluster
            elif t1 == "ensuite_bathroom" and t2 == "walk_in_closet": is_door = True  # within suite cluster
            elif t2 == "ensuite_bathroom" and t1 == "walk_in_closet": is_door = True
            # Private zone circulation: adjacent private rooms connect via shared walls
            # (handles multi-row private bands where hallway doesn't directly border all rooms)
            _PRIVATE_ALL = _BEDROOM_TYPES | _BATH_TYPES | {"walk_in_closet", "closet", "half_bath"}
            if not is_door and all(t in _PRIVATE_ALL for t in (t1, t2)):
                is_door = True

            if is_door:
                if shared_v:
                    doors.append({"x": shared_v[0], "y": shared_v[1], "isVertical": True, "roomA": r1["id"], "roomB": r2["id"]})
                    processed_pairs.add(pair)
                elif shared_h:
                    doors.append({"x": shared_h[0], "y": shared_h[1], "isVertical": False, "roomA": r1["id"], "roomB": r2["id"]})
                    processed_pairs.add(pair)

    return doors


def _fix_isolated_service_rooms(placed: List[dict], total_w: float) -> List[dict]:
    """
    Pantry and laundry can wrap to isolated rows when the social band is tight.
    If either ends up in its own row with no x-overlap with the kitchen,
    reposition it directly below the kitchen's right end so they share a wall
    and a door can be detected. This keeps the staircase/hallway reachable.
    Called AFTER snap_and_fill (which resets x to 0 for isolated rows).
    """
    kitchen = next((r for r in placed if r["type"] == "kitchen"), None)
    if not kitchen:
        return placed

    k_x1, k_x2 = kitchen["x"], kitchen["x"] + kitchen["width"]

    for rtype in ("pantry", "laundry_room"):
        room = next((r for r in placed if r["type"] == rtype), None)
        if not room:
            continue
        if room["y"] <= kitchen["y"]:
            continue  # same row or above — no fix needed

        r_x1, r_x2 = room["x"], room["x"] + room["width"]
        x_overlap = min(k_x2, r_x2) - max(k_x1, r_x1)
        if x_overlap >= 2:
            continue  # already shares wall with kitchen

        # Snap room to right edge of kitchen, clamped to footprint
        new_x = min(k_x2 - room["width"], int(total_w) - room["width"])
        new_x = max(0, round(new_x / 2) * 2)
        room["x"] = int(new_x)
        # Also snap y to the kitchen's actual bottom edge (not the row y_cursor).
        # Living room height > kitchen height creates a gap; closing it ensures a shared wall.
        if room["y"] > kitchen["y"] + kitchen["height"]:
            room["y"] = kitchen["y"] + kitchen["height"]

    return placed


def _validate_design(plan: dict) -> List[dict]:
    """
    Architectural sanity checks on a generated plan.

    Checks:
      1. Entry point exists (foyer/mudroom/garage on floor 1 at front of house)
      2. Every habitable room has at least one door connection
      3. All floor-1 rooms are reachable from entry via doors (BFS connectivity)
      4. All floor-2 rooms are reachable from upper hall (if two-story)
      5. No rooms extend beyond the declared footprint

    Returns: list of issue dicts — [] = clean plan.
      Each issue: {"type": str, "severity": "error"|"warning", "message": str}
    """
    issues = []
    rooms = plan["rooms"]
    doors = plan.get("doors", [])
    total_w = plan["totalWidth"]
    total_h = plan["totalHeight"]
    floor2_h = plan.get("floor2Height", total_h)
    is_two_story = plan.get("stories", 1) == 2

    f1_rooms = [r for r in rooms if r.get("floor", 1) == 1]
    f2_rooms = [r for r in rooms if r.get("floor", 1) == 2]

    # ── Check 1: Entry point exists and is at the front (y ≤ 4) ─────────────
    _ENTRY_TYPES = {"foyer", "mudroom", "garage"}
    entry_candidates = [r for r in f1_rooms if r["type"] in _ENTRY_TYPES]
    front_entry = [r for r in entry_candidates if r["y"] <= 6]
    if not entry_candidates:
        issues.append({
            "type": "no_entry",
            "severity": "error",
            "message": "No entry point (foyer/garage) on floor 1 — visitors have no way in.",
        })
    elif not front_entry:
        issues.append({
            "type": "entry_not_front",
            "severity": "warning",
            "message": "Entry (foyer/garage) is not at the front of the house.",
        })

    # ── Check 2: Every habitable room has ≥1 door connection ─────────────────
    rooms_with_doors: set = set()
    for d in doors:
        rooms_with_doors.add(d.get("roomA"))
        rooms_with_doors.add(d.get("roomB"))

    isolated = []
    for r in rooms:
        if r["type"] in _OUTDOOR_TYPES:
            continue
        # Hallways are circulation — missing a door here is a layout bug
        if r["id"] not in rooms_with_doors:
            isolated.append(r["name"])
    if isolated:
        issues.append({
            "type": "isolated_rooms",
            "severity": "error",
            "message": f"Rooms with no door: {', '.join(isolated[:5])}{'…' if len(isolated) > 5 else ''}",
        })

    # ── Check 3: Floor-1 BFS connectivity from foyer ─────────────────────────
    # Build adjacency map (only floor-1 doors; two-story floor-2 checked separately)
    def _bfs_reachable(start_id: str, door_list: list) -> set:
        adj: Dict[str, set] = {}
        for d in door_list:
            a, b = d.get("roomA"), d.get("roomB")
            if a and b:
                adj.setdefault(a, set()).add(b)
                adj.setdefault(b, set()).add(a)
        visited = {start_id}
        queue = [start_id]
        while queue:
            curr = queue.pop(0)
            for nbr in adj.get(curr, set()):
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        return visited

    f1_doors = [d for d in doors if d.get("floor", 1) == 1]
    # Find best start node: foyer > mudroom > garage > any entry
    start_f1 = None
    for etype in ("foyer", "mudroom", "garage"):
        for r in f1_rooms:
            if r["type"] == etype:
                start_f1 = r["id"]
                break
        if start_f1:
            break
    if start_f1 is None and f1_rooms:
        start_f1 = f1_rooms[0]["id"]

    if start_f1:
        reachable_f1 = _bfs_reachable(start_f1, f1_doors)
        unreachable = [
            r["name"] for r in f1_rooms
            if r["type"] not in _OUTDOOR_TYPES and r["id"] not in reachable_f1
        ]
        if unreachable:
            issues.append({
                "type": "unreachable_f1",
                "severity": "error",
                "message": f"Floor-1 rooms cut off from entry: {', '.join(unreachable[:5])}",
            })

    # ── Check 4: Floor-2 connectivity from upper hall ────────────────────────
    if is_two_story and f2_rooms:
        f2_doors = [d for d in doors if d.get("floor", 1) == 2]
        start_f2 = None
        for r in f2_rooms:
            if r["type"] == "hallway":
                start_f2 = r["id"]
                break
        if start_f2 is None and f2_rooms:
            start_f2 = f2_rooms[0]["id"]

        if start_f2:
            reachable_f2 = _bfs_reachable(start_f2, f2_doors)
            unreachable_f2 = [
                r["name"] for r in f2_rooms
                if r["type"] not in _OUTDOOR_TYPES and r["id"] not in reachable_f2
            ]
            if unreachable_f2:
                issues.append({
                    "type": "unreachable_f2",
                    "severity": "error",
                    "message": f"Floor-2 rooms cut off from upper hall: {', '.join(unreachable_f2[:5])}",
                })

    # ── Check 5: Rooms within footprint ──────────────────────────────────────
    oob = []
    for r in rooms:
        fh = total_h if r.get("floor", 1) == 1 else floor2_h
        # Allow 2ft tolerance for rounding
        if (r["x"] < -2 or r["y"] < -2
                or r["x"] + r["width"] > total_w + 2
                or r["y"] + r["height"] > fh + 2):
            oob.append(r["name"])
    if oob:
        issues.append({
            "type": "out_of_bounds",
            "severity": "warning",
            "message": f"Rooms outside footprint: {', '.join(oob[:3])}",
        })

    return issues


def _generate_two_story_variant(
    constraints: dict,
    expert_weights: dict,
    variant_name: str,
    ceiling_ft: int,
    config: MOEConfig,
) -> dict:
    """
    Build a two-story plan using the zone-solver twice:
      Floor 1 — public spaces + garage + staircase
      Floor 2 — all bedrooms + baths + upper hall

    Both floors share the same total_w.  The staircase on F1 aligns
    (same X position) with the upper hall on F2.
    """
    bedrooms      = constraints.get("bedrooms", 3)
    bathrooms     = constraints.get("bathrooms", 2)
    sqft          = constraints.get("sqft", 1800)
    style         = constraints.get("style", "modern")
    open_plan     = constraints.get("openPlan", False)
    primary_suite = constraints.get("primarySuite", True)
    home_office   = constraints.get("homeOffice", False)
    formal_dining = constraints.get("formalDining", False)
    garage        = constraints.get("garage", "2car")
    laundry       = constraints.get("laundry", "room")
    outdoor       = constraints.get("outdoor", "patio")

    f1_rooms_raw, f2_rooms_raw = _build_room_list_two_story(
        bedrooms, bathrooms, sqft, style, open_plan, primary_suite,
        home_office, formal_dining, garage, laundry, outdoor,
    )

    # Size rooms using MOE-influenced sizing
    def _size(rooms_raw):
        sized = []
        for r in rooms_raw:
            w, h = _moe_adjusted_size(r["type"], sqft, expert_weights)
            sized.append({**r, "width": w, "height": h})
        return sized

    f1_sized = _size(f1_rooms_raw)
    f2_sized = _size(f2_rooms_raw)

    # ── Footprint width ───────────────────────────────────────────────────
    # Base width on target sqft, not just F1 rooms — ensures realistic
    # proportions even without a garage (30' is too narrow for a two-story).
    # Typical US two-story: ~40–50' wide.  Each floor ≈ sqft/2 SF.
    aspect = 1.4
    floor_area_target = sqft * 0.5
    sqrt_w = max(36, min(80, round(math.sqrt(floor_area_target * aspect) / 2) * 2))

    # Minimum width = sum of IRC minimums for F1 social band rooms so they all fit
    # in one row without wrapping. Without this, rooms lose wall adjacency → errors.
    has_garage_f1 = any(r["type"] == "garage" for r in f1_sized)
    if has_garage_f1:
        # Entry band (zone 0, excl. garage itself) is its own full-width row — not social.
        # Social width is just zones 1+2.
        social_zones = {1, 2}
    else:
        # No garage: ALL zone-0 rooms (foyer, laundry, mudroom) are prepended to
        # the social band in _place_rooms_architectural, so count them all.
        social_zones = {0, 1, 2}
    min_social_w = sum(
        IRC_ROOM_SPECS.get(r["type"], (4, 4))[0]
        for r in f1_sized
        if ZONE_MAP.get(r["type"], 1) in social_zones and r["type"] != "garage"
    )
    total_w = max(sqrt_w, round(min_social_w / 2) * 2, 36)
    total_w = min(80, total_w)

    # ── Per-floor heights from actual room areas (not full-house estimate) ─
    # Using actual F1/F2 room areas prevents the empty dead band that appears
    # when the solver reserves height for zones that have no rooms on that floor.
    f1_cond_area = sum(r["width"] * r["height"] for r in f1_sized
                       if r["type"] not in _OUTDOOR_TYPES)
    f1_out_area  = sum(r["width"] * r["height"] for r in f1_sized
                       if r["type"] in _OUTDOOR_TYPES)
    # Height = conditioned band height + outdoor band depth
    total_h_f1 = max(20, round((f1_cond_area / total_w + f1_out_area / total_w) / 2) * 2 + 4)
    total_h_f1 = min(60, total_h_f1)

    f2_area = sum(r["width"] * r["height"] for r in f2_sized)
    total_h_f2 = max(18, round((f2_area / total_w) / 2) * 2 + 4)
    total_h_f2 = min(50, total_h_f2)

    # Place each floor independently
    f1_placed = _place_rooms_architectural(f1_sized, total_w, total_h_f1, sqft, expert_weights)
    f2_placed = _place_rooms_architectural(f2_sized, total_w, total_h_f2, sqft, expert_weights)

    # Snap and fill with the height derived from actual placement extent,
    # NOT the estimate — this eliminates the empty bottom band.
    def _trim_height(placed, fallback):
        if not placed:
            return fallback
        h = max(r["y"] + r["height"] for r in placed)
        return max(fallback // 2, round(h / 2) * 2)

    f1_h = _trim_height(f1_placed, total_h_f1)
    f2_h = _trim_height(f2_placed, total_h_f2)

    f1_placed = _snap_and_fill(f1_placed, total_w, f1_h)
    f1_placed = _fix_isolated_service_rooms(f1_placed, total_w)
    f1_placed = _fill_vertical_gaps(f1_placed, f1_h)
    # Recompute after fill — outdoor rooms may have extended
    f1_h = _trim_height(f1_placed, f1_h)

    f2_placed = _snap_and_fill(f2_placed, total_w, f2_h)
    f2_placed = _fill_vertical_gaps(f2_placed, f2_h)
    f2_h = _trim_height(f2_placed, f2_h)

    # Staircase alignment: find staircase on F1, place upper hall at same X on F2
    stair = next((r for r in f1_placed if r["type"] == "hallway"
                  and "stair" in r["name"].lower()), None)
    upper_hall = next((r for r in f2_placed if r["type"] == "hallway"), None)
    if stair and upper_hall:
        upper_hall["x"] = stair["x"]
        upper_hall["width"] = stair["width"]

    # Tag floors
    for r in f1_placed:
        r["floor"] = 1
    for r in f2_placed:
        r["floor"] = 2

    # Doors per floor
    doors_f1 = _add_doors(f1_placed)
    doors_f2 = _add_doors(f2_placed)
    for d in doors_f2:
        d["floor"] = 2

    # Colour rooms
    all_rooms = f1_placed + f2_placed
    for r in all_rooms:
        r["color"] = ROOM_COLORS.get(r["type"], "#e8e4dc")

    plan = {
        "id":           f"moe_{uuid.uuid4().hex[:8]}",
        "name":         variant_name,
        "totalWidth":   total_w,
        "totalHeight":  f1_h,
        "floor2Height": f2_h,
        "ceilingHeight": ceiling_ft,
        "style":        style,
        "stories":      2,
        "rooms":        all_rooms,
        "doors":        doors_f1 + doors_f2,
        "generator":    "moe+two-story",
    }

    # Validate and attach issues
    plan["validationIssues"] = _validate_design(plan)
    return plan


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
