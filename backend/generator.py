"""
Deterministic architectural floor plan generator.

Each layout uses named row bands.  Within each band every room gets the full
band height; widths are scaled proportionally from the room's natural width so
aspect ratios stay close to the real-world spec.

Three variants give visually distinct but always correct plans (zero overlaps,
zero boundary overflows, sensible room shapes).
"""
import uuid, math

CEILING_HEIGHTS = {"standard": 9, "high": 10, "vaulted": 12}
VARIANTS = ["Open Wing", "Traditional Split", "Compact Core"]

ROOM_COLORS = {
    "great":    "#F5E0C8", "living":   "#F5E6D3", "family":   "#FAE8D3",
    "kitchen":  "#D3E8F5", "dining":   "#D3F5E6", "master":   "#F0C8E0",
    "bedroom":  "#F5D3E8", "bathroom": "#E8D3F5", "ensuite":  "#DEC8F0",
    "closet":   "#F0E8E0", "garage":   "#DCDCDC", "hallway":  "#F0F0E8",
    "foyer":    "#EEEAE0", "mudroom":  "#E8E4D8", "office":   "#F5F0D3",
    "laundry":  "#D3F5F5", "utility":  "#E8E8D3", "patio":    "#E0EED8",
    "deck":     "#E8E4D0", "pantry":   "#EDE8DC", "powder":   "#E8D8F0",
}


def get_color(rtype: str) -> str:
    t = rtype.lower().replace("_", " ").replace("-", " ")
    for k, v in ROOM_COLORS.items():
        if k in t:
            return v
    return "#F0F0F0"


def rv(v: float, d: int = 1) -> float:
    return round(v, d)


# ─────────────────────────────────────────────────────────────────────────────
# Room spec builder
# ─────────────────────────────────────────────────────────────────────────────

def build_specs(c: dict) -> list[dict]:
    """Return specs with natural (un-scaled) widths and heights."""
    sqft  = int(c.get("sqft", 1800))
    beds  = int(c.get("bedrooms", 3))
    baths = int(c.get("bathrooms", 2))
    sf    = (sqft / 1800) ** 0.45   # mild scale so large homes have larger rooms

    def s(v):  return max(5.0, round(v * sf))

    sp = []
    def add(id_, name, rtype, w, h, zone):
        sp.append({"id": id_, "name": name, "type": rtype,
                    "w": float(w), "h": float(h), "zone": zone})

    # ── Public / living ──────────────────────────────────────────────────────
    if c.get("openPlan"):
        add("great", "Great Room", "great_room", s(32), s(18), "public")
    else:
        add("living",  "Living Room",  "living_room", s(20), s(16), "public")
        add("kitchen", "Kitchen",      "kitchen",     s(14), s(12), "public")
        add("dining",  "Dining Area" if not c.get("formalDining") else "Dining Room",
            "dining_room", s(12), s(10), "public")

    if c.get("homeOffice"):
        add("office", "Home Office", "home_office", s(12), s(10), "public")

    # ── Bedrooms ─────────────────────────────────────────────────────────────
    for i in range(beds):
        if i == 0:
            add("bed1", "Primary Bedroom", "master_bedroom", s(14), s(13), "bed_upper")
        else:
            add(f"bed{i+1}", f"Bedroom {i+1}", "bedroom", s(12), s(11), "bed_upper")

    # ── Baths / closets (bed_lower row — below bedrooms) ────────────────────
    if c.get("primarySuite"):
        add("ensuite", "Primary En-Suite Bath", "ensuite_bathroom", s(9),  s(7), "bed_lower")
        add("wcloset", "Walk-in Closet",          "closet",           s(7),  s(6), "bed_lower")

    shared = max(0, baths - (1 if c.get("primarySuite") else 0))
    for i in range(shared):
        add(f"bath{i+1}",
            "Full Bathroom" if i == 0 else "Powder Bath",
            "bathroom" if i == 0 else "powder_bath",
            s(9) if i == 0 else s(6),
            s(7) if i == 0 else s(5),
            "bed_lower")

    # ── Service / circulation ────────────────────────────────────────────────
    add("foyer", "Entry Foyer", "foyer",   s(10), s(8), "service")
    add("hall",  "Hallway",     "hallway", s(8),  s(6), "service")

    if c.get("laundry") == "room":
        add("laundry", "Laundry Room",   "laundry_room", s(8), s(7), "service")
    elif c.get("laundry") == "closet":
        add("laundry", "Laundry Closet", "laundry_room", s(5), s(5), "service")

    # ── Garage ───────────────────────────────────────────────────────────────
    garage = c.get("garage", "none")
    if garage != "none":
        gw = {"1car": 12.0, "2car": 24.0, "3car": 34.0}[garage]
        add("garage", "Garage", "garage", gw, 22.0, "garage")

    # ── Outdoor ──────────────────────────────────────────────────────────────
    outdoor = c.get("outdoor", "none")
    if outdoor in ("patio", "both"):
        add("patio", "Patio", "patio", s(18), s(12), "outdoor")
    if outdoor in ("deck", "both"):
        add("deck",  "Deck",  "deck",  s(14), s(10), "outdoor")

    return sp


# ─────────────────────────────────────────────────────────────────────────────
# Footprint
# ─────────────────────────────────────────────────────────────────────────────

def calc_footprint(specs: list[dict], c: dict) -> tuple[float, float]:
    sqft = int(c.get("sqft", 1800))
    fp   = sqft / 0.82   # gross footprint (walls + mechanical)

    # Minimum H per zone so no band is crushed (feet).
    # "public" covers both kitchen/dining (in the garage band) and living room
    # (separate band), so count it once for the living band only.
    MIN_ZONE_H = {
        "garage": 20.0, "public": 13.0, "service": 6.0,
        "bed_upper": 11.0, "bed_lower": 7.0, "outdoor": 8.0,
    }
    zones_present = {s["zone"] for s in specs}
    H_min = sum(MIN_ZONE_H.get(z, 6.0) for z in MIN_ZONE_H if z in zones_present)

    H_fp = math.sqrt(fp / 1.35)          # sqft-derived height
    H    = rv(max(H_fp, H_min))           # always tall enough for all zones
    W    = rv(fp / H)                     # width from area constraint

    # Ensure wide enough for garage
    gk = c.get("garage", "none")
    if gk == "2car": W = max(W, 46.0)
    if gk == "3car": W = max(W, 58.0)
    W = max(W, 36.0)
    H = rv(fp / W)                        # recompute if W was bumped
    H = max(H, H_min)
    return rv(W), rv(H)


# ─────────────────────────────────────────────────────────────────────────────
# Primitives — exact overlap-free row / column packing
# ─────────────────────────────────────────────────────────────────────────────

def fill_row(rooms: list[dict], x0: float, y0: float,
             W: float, H: float) -> list[dict]:
    """Pack rooms as a horizontal band.
    Widths are proportional to rooms' natural widths.  All rooms get height H.
    """
    if not rooms:
        return []
    nat_w = [rm["w"] for rm in rooms]
    total = sum(nat_w)
    widths = [rv(nw / total * W) for nw in nat_w]
    # Last room absorbs rounding residual
    widths[-1] = rv(W - sum(widths[:-1]))

    out, cx = [], x0
    for rm, w in zip(rooms, widths):
        out.append({**rm, "x": rv(cx), "y": rv(y0), "width": w, "height": rv(H)})
        cx = rv(cx + w)
    return out


def fill_col(rooms: list[dict], x0: float, y0: float,
             W: float, H: float) -> list[dict]:
    """Pack rooms as a vertical column.
    Heights are proportional to rooms' natural heights.  All rooms get width W.
    """
    if not rooms:
        return []
    nat_h = [rm["h"] for rm in rooms]
    total = sum(nat_h)
    heights = [rv(nh / total * H) for nh in nat_h]
    heights[-1] = rv(H - sum(heights[:-1]))

    out, cy = [], y0
    for rm, h in zip(rooms, heights):
        out.append({**rm, "x": rv(x0), "y": rv(cy), "width": rv(W), "height": h})
        cy = rv(cy + h)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────────────────────────────────────

def zone_heights(bands: list[float], H: float) -> list[float]:
    """Turn fractional band shares into pixel heights that sum exactly to H."""
    total = sum(bands)
    hs = [rv(b / total * H) for b in bands]
    hs[-1] = rv(H - sum(hs[:-1]))
    return hs


def _g(specs: list[dict]) -> dict:
    z: dict[str, list] = {}
    for s in specs:
        z.setdefault(s["zone"], []).append(s)
    return z


# ─────────────────────────────────────────────────────────────────────────────
# Layout A — Open Wing
# ─────────────────────────────────────────────────────────────────────────────
# ┌─GARAGE─┬──KITCHEN──┬──DINING──┐   band 0
# ├─────── LIVING ROOM ───────────┤   band 1
# ├─SERVICE (foyer+hall+laundry)──┤   band 2
# ├─BED_UPPER (all bed widths)────┤   band 3
# ├─BED_LOWER (ensuite+closet+bath┤   band 4
# └─────── OUTDOOR ───────────────┘   band 5

def layout_open_wing(specs, W, H):
    z = _g(specs)
    garage  = z.get("garage", [])
    public  = z.get("public", [])
    service = z.get("service", [])
    bed_up  = z.get("bed_upper", [])
    bed_lo  = z.get("bed_lower", [])
    outdoor = z.get("outdoor", [])

    # Split public into kitchen/dining vs living
    living  = [r for r in public if "living" in r["type"] or "great" in r["type"]]
    kitchen = [r for r in public if r not in living]

    # Band shares (proportional weights, not fixed fractions)
    shares = [
        max(22.0, sum(r["h"] for r in garage) if garage else 0) or 14,  # b0 garage row
        sum(r["h"] for r in living) / max(1, len(living)) if living else 12,  # b1 living
        max(r["h"] for r in service) if service else 8,                       # b2 service
        max(r["h"] for r in bed_up)  if bed_up  else 12,                      # b3 bed_upper
        max(r["h"] for r in bed_lo)  if bed_lo  else 7,                       # b4 bed_lower
        max(r["h"] for r in outdoor) if outdoor else 0,                       # b5 outdoor
    ]
    # Remove empty outdoor band if no outdoor
    if not outdoor:
        shares = shares[:5]

    hs = zone_heights(shares, H)
    y, placed = 0.0, []

    # Band 0: Garage (left column) + Kitchen/Dining stacked vertically (right)
    b0 = hs[0]
    gw = garage[0]["w"] if garage else 0.0
    if garage:
        placed += fill_col(garage, 0, y, gw, b0)
    if kitchen:
        placed += fill_col(kitchen, gw, y, W - gw, b0)
    y += b0

    # Band 1: Living (full width)
    b1 = hs[1]
    if living:
        placed += fill_row(living, 0, y, W, b1)
    y += b1

    # Band 2: Service (full width)
    b2 = hs[2]
    if service:
        placed += fill_row(service, 0, y, W, b2)
    y += b2

    # Band 3: Bed upper — ALL bedrooms side by side
    b3 = hs[3]
    if bed_up:
        placed += fill_row(bed_up, 0, y, W, b3)
    y += b3

    # Band 4: Bed lower — ensuite, closet, baths side by side
    b4 = hs[4]
    if bed_lo:
        placed += fill_row(bed_lo, 0, y, W, b4)
    y += b4

    # Band 5: Outdoor
    if outdoor:
        placed += fill_row(outdoor, 0, y, W, hs[5])

    return placed


# ─────────────────────────────────────────────────────────────────────────────
# Layout B — Traditional Split
# ─────────────────────────────────────────────────────────────────────────────
# ┌─GARAGE─┬──KITCHEN──┬──DINING──┐   band 0  (row)
# ├─────── SERVICE BELT ──────────┤   band 1  (row)
# ├─LIVING (col)─┬─BED_UP (row)──┤   band 2  (split)
# │              ├─BED_LO (row)──┤
# └──────────────┴── OUTDOOR ────┘   band 3

def layout_traditional_split(specs, W, H):
    z = _g(specs)
    garage  = z.get("garage", [])
    public  = z.get("public", [])
    service = z.get("service", [])
    bed_up  = z.get("bed_upper", [])
    bed_lo  = z.get("bed_lower", [])
    outdoor = z.get("outdoor", [])

    living  = [r for r in public if "living" in r["type"] or "great" in r["type"]]
    kitchen = [r for r in public if r not in living]

    gw = garage[0]["w"] if garage else rv(W * 0.44)

    shares = [
        max(22.0, sum(r["h"] for r in garage) if garage else 14),   # b0
        max(r["h"] for r in service) if service else 8,              # b1
        max(r["h"] for r in bed_up)  if bed_up  else 12,            # b2 (living+beds share)
        max(r["h"] for r in bed_lo)  if bed_lo  else 7,             # b3 (bed_lower)
        max(r["h"] for r in outdoor) if outdoor else 0,             # b4
    ]
    if not outdoor:
        shares = shares[:4]

    hs = zone_heights(shares, H)
    y, placed = 0.0, []

    # Band 0: Garage (left col) + Kitchen/Dining stacked vertically (right)
    b0 = hs[0]
    if garage:
        placed += fill_col(garage, 0, y, gw, b0)
    if kitchen:
        placed += fill_col(kitchen, gw, y, W - gw, b0)
    y += b0

    # Band 1: Service (full width)
    b1 = hs[1]
    if service:
        placed += fill_row(service, 0, y, W, b1)
    y += b1

    # Band 2+3: Living (left column) + Bedrooms (right column, 2 rows)
    left_w  = rv(W * 0.40)
    right_w = rv(W - left_w)
    b2, b3  = hs[2], hs[3]

    if living:
        placed += fill_col(living, 0, y, left_w, b2 + b3)

    if bed_up:
        placed += fill_row(bed_up, left_w, y, right_w, b2)
    if bed_lo:
        placed += fill_row(bed_lo, left_w, y + b2, right_w, b3)

    y += b2 + b3

    # Band 4: Outdoor
    if outdoor:
        placed += fill_row(outdoor, 0, y, W, hs[4])

    return placed


# ─────────────────────────────────────────────────────────────────────────────
# Layout C — Compact Core
# ─────────────────────────────────────────────────────────────────────────────
# ┌─GARAGE─┬──KITCHEN──┬──DINING──┐   band 0  (garage + kitchen side by side)
# ├────── SERVICE BELT ───────────┤   band 1  (foyer, hall, laundry)
# ├────── GREAT/LIVING ROOM ──────┤   band 2  (living space)
# ├──────── BED_UPPER ────────────┤   band 3
# ├──────── BED_LOWER ────────────┤   band 4
# └──────── OUTDOOR ──────────────┘   band 5

def layout_compact_core(specs, W, H):
    z = _g(specs)
    garage  = z.get("garage", [])
    public  = z.get("public", [])
    service = z.get("service", [])
    bed_up  = z.get("bed_upper", [])
    bed_lo  = z.get("bed_lower", [])
    outdoor = z.get("outdoor", [])

    living  = [r for r in public if "living" in r["type"] or "great" in r["type"]]
    kitchen = [r for r in public if r not in living]

    gw = garage[0]["w"] if garage else rv(W * 0.44)

    shares = [
        max(22.0, sum(r["h"] for r in garage) if garage else 14),  # b0 garage+kitchen
        max(r["h"] for r in service) if service else 8,            # b1 service belt
        max(r["h"] for r in living)  if living  else 14,           # b2 living
        max(r["h"] for r in bed_up)  if bed_up  else 12,           # b3 bed_upper
        max(r["h"] for r in bed_lo)  if bed_lo  else 7,            # b4 bed_lower
        max(r["h"] for r in outdoor) if outdoor else 0,            # b5 outdoor
    ]
    if not outdoor:
        shares = shares[:5]

    hs = zone_heights(shares, H)
    y, placed = 0.0, []

    # Band 0: Garage (left) + Kitchen/Dining stacked vertically (right)
    b0 = hs[0]
    if garage:
        placed += fill_col(garage, 0, y, gw, b0)
    if kitchen:
        placed += fill_col(kitchen, gw, y, W - gw, b0)
    y += b0

    # Band 1: Service (full width)
    b1 = hs[1]
    if service:
        placed += fill_row(service, 0, y, W, b1)
    y += b1

    # Band 2: Living (full width)
    b2 = hs[2]
    if living:
        placed += fill_row(living, 0, y, W, b2)
    y += b2

    # Band 3: Bed upper
    b3 = hs[3]
    if bed_up:
        placed += fill_row(bed_up, 0, y, W, b3)
    y += b3

    # Band 4: Bed lower
    b4 = hs[4]
    if bed_lo:
        placed += fill_row(bed_lo, 0, y, W, b4)
    y += b4

    # Band 5: Outdoor
    if outdoor:
        placed += fill_row(outdoor, 0, y, W, hs[5])

    return placed


LAYOUTS = [layout_open_wing, layout_traditional_split, layout_compact_core]


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def generate_floor_plan(constraints: dict, variant_idx: int = 0) -> dict:
    ceiling_ft = CEILING_HEIGHTS.get(constraints.get("ceilingHeight", "standard"), 9)
    variant    = VARIANTS[variant_idx % len(VARIANTS)]

    specs = build_specs(constraints)
    W, H  = calc_footprint(specs, constraints)

    placed = LAYOUTS[variant_idx % len(LAYOUTS)](specs, W, H)

    rooms = []
    for p in placed:
        w = rv(p["width"])
        h = rv(p["height"])
        x = rv(max(0.0, min(p["x"], W - w)))
        y = rv(max(0.0, min(p["y"], H - h)))
        rooms.append({
            "id":     p.get("id", str(uuid.uuid4())),
            "name":   p["name"],
            "type":   p["type"],
            "x": x, "y": y, "width": w, "height": h,
            "color":  get_color(p["type"]),
        })

    return {
        "id":            str(uuid.uuid4()),
        "name":          f"{variant} Plan",
        "totalWidth":    W,
        "totalHeight":   H,
        "ceilingHeight": ceiling_ft,
        "rooms":         rooms,
    }
