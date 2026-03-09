# US residential construction cost estimates (2024 national averages, $/sqft)
# Tuple: (economy, mid-range, premium)
ROOM_COST_RATES: dict[str, tuple[int, int, int]] = {
    "master_bedroom":    (85, 110, 155),
    "bedroom":           (75,  95, 135),
    "bathroom":         (150, 200, 300),
    "ensuite_bathroom": (175, 225, 350),
    "half_bath":        (120, 160, 250),
    "kitchen":          (180, 250, 400),
    "living_room":       (70,  90, 130),
    "dining_room":       (70,  90, 130),
    "family_room":       (70,  90, 130),
    "home_office":       (80, 105, 150),
    "garage":            (45,  60,  90),
    "laundry_room":      (90, 120, 180),
    "utility_room":      (70,  95, 140),
    "mudroom":           (80, 105, 150),
    "foyer":             (90, 120, 170),
    "hallway":           (55,  70, 100),
    "closet":            (50,  65,  95),
    "walk_in_closet":    (60,  80, 120),
    "pantry":            (70,  90, 130),
    "patio":             (25,  40,  65),
    "deck":              (30,  50,  80),
}

REGION_MULTIPLIERS = {
    "National Average": 1.00,
    "Northeast":        1.35,
    "West Coast":       1.45,
    "Southeast":        0.90,
    "Midwest":          0.95,
    "Southwest":        1.00,
    "Mountain":         1.05,
}


def _normalize_type(rtype: str) -> str:
    key = rtype.lower().replace(" ", "_").replace("-", "_")
    if key in ROOM_COST_RATES:
        return key
    # partial match fallback
    for k in ROOM_COST_RATES:
        if k in key or key in k:
            return k
    return "living_room"


def estimate_cost(floor_plan: dict, region: str = "National Average") -> dict:
    rooms = floor_plan.get("rooms", [])
    mult = REGION_MULTIPLIERS.get(region, 1.0)
    total_w = floor_plan.get("totalWidth", 50)
    total_h = floor_plan.get("totalHeight", 40)

    room_rows = []
    total_low = total_mid = total_high = 0

    for room in rooms:
        sqft = room["width"] * room["height"]
        key = _normalize_type(room.get("type", "living_room"))
        lo, mi, hi = ROOM_COST_RATES[key]
        low  = round(sqft * lo * mult)
        mid  = round(sqft * mi * mult)
        high = round(sqft * hi * mult)
        room_rows.append({
            "room": room["name"], "type": room.get("type", ""),
            "sqft": round(sqft), "low": low, "mid": mid, "high": high,
        })
        total_low  += low
        total_mid  += mid
        total_high += high

    # Foundation
    footprint = total_w * total_h
    fnd = (round(footprint * 8 * mult), round(footprint * 11 * mult), round(footprint * 15 * mult))

    # Roof
    roof = (round(footprint * 6 * mult), round(footprint * 9 * mult), round(footprint * 12 * mult))

    sub_lo = total_low  + fnd[0] + roof[0]
    sub_mi = total_mid  + fnd[1] + roof[1]
    sub_hi = total_high + fnd[2] + roof[2]

    # MEP (mechanical / electrical / plumbing) — 18-28% of subtotal
    mep = (round(sub_lo * 0.18), round(sub_mi * 0.22), round(sub_hi * 0.28))

    grand_lo = sub_lo + mep[0]
    grand_mi = sub_mi + mep[1]
    grand_hi = sub_hi + mep[2]

    total_sqft = max(1, sum(r["sqft"] for r in room_rows))

    return {
        "region": region,
        "rooms": room_rows,
        "foundation":  {"low": fnd[0],   "mid": fnd[1],   "high": fnd[2]},
        "roof":        {"low": roof[0],   "mid": roof[1],  "high": roof[2]},
        "mep":         {"low": mep[0],    "mid": mep[1],   "high": mep[2]},
        "total":       {"low": grand_lo,  "mid": grand_mi, "high": grand_hi},
        "per_sqft":    {"low": round(grand_lo/total_sqft), "mid": round(grand_mi/total_sqft), "high": round(grand_hi/total_sqft)},
    }
