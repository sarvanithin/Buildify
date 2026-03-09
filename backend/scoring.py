import math


def score_design(floor_plan: dict) -> dict:
    rooms = floor_plan.get("rooms", [])
    total_w = floor_plan.get("totalWidth", 1)
    total_h = floor_plan.get("totalHeight", 1)

    scores = {
        "adjacency":     _score_adjacency(rooms),
        "natural_light": _score_natural_light(rooms, total_w, total_h),
        "circulation":   _score_circulation(rooms, total_w, total_h),
        "privacy":       _score_privacy(rooms),
        "efficiency":    _score_efficiency(rooms, total_w, total_h),
    }

    overall = sum(scores.values()) / len(scores)

    return {
        "scores": scores,
        "overall": round(overall, 1),
        "grade": _grade(overall),
        "insights": _insights(scores, rooms),
    }


# ── helpers ──────────────────────────────────────────────────────────────────

def _centroid(r):
    return (r["x"] + r["width"] / 2, r["y"] + r["height"] / 2)


def _dist(r1, r2):
    c1, c2 = _centroid(r1), _centroid(r2)
    return math.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


def _score_adjacency(rooms):
    IDEAL_PAIRS = [
        ("kitchen", "dining"),
        ("kitchen", "family_room"),
        ("kitchen", "living_room"),
        ("master_bedroom", "ensuite_bathroom"),
        ("bedroom", "bathroom"),
        ("foyer", "living_room"),
        ("laundry_room", "garage"),
        ("dining_room", "kitchen"),
    ]

    total, checked = 0, 0
    for t1, t2 in IDEAL_PAIRS:
        r1s = [r for r in rooms if t1 in r.get("type", "").lower()]
        r2s = [r for r in rooms if t2 in r.get("type", "").lower()]
        if r1s and r2s:
            checked += 1
            d = min(_dist(a, b) for a in r1s for b in r2s)
            if d <= 15:   total += 100
            elif d <= 25: total += 75
            elif d <= 40: total += 50
            else:         total += 25

    return round(total / max(1, checked), 1) if checked else 75.0


def _score_natural_light(rooms, total_w, total_h):
    scores = []
    for r in rooms:
        on_top    = r["y"] < 5
        on_bottom = (r["y"] + r["height"]) > (total_h - 5)
        on_left   = r["x"] < 5
        on_right  = (r["x"] + r["width"]) > (total_w - 5)
        walls = sum([on_top, on_bottom, on_left, on_right])
        rtype = r.get("type", "")
        needs = any(t in rtype for t in ["bedroom", "living", "dining", "kitchen", "office"])
        if needs:
            scores.append(100 if walls >= 2 else 80 if walls == 1 else 40)
        else:
            scores.append(85)
    return round(sum(scores) / max(1, len(scores)), 1) if scores else 75.0


def _score_circulation(rooms, total_w, total_h):
    has_foyer   = any("foyer"  in r.get("type","").lower() or "entry" in r.get("type","").lower() for r in rooms)
    has_hallway = any("hallway" in r.get("type","").lower() or "hall"  in r.get("type","").lower() for r in rooms)
    total_sqft  = sum(r["width"] * r["height"] for r in rooms)
    circ_sqft   = sum(r["width"] * r["height"] for r in rooms
                      if any(k in r.get("type","").lower() for k in ["hallway","hall","foyer","entry"]))
    circ_ratio  = circ_sqft / max(1, total_sqft)

    score = 60
    if has_foyer:   score += 15
    if has_hallway: score += 15
    if 0.06 <= circ_ratio <= 0.15: score += 10
    elif circ_ratio < 0.06:        score -= 5
    return min(100, round(score, 1))


def _score_privacy(rooms):
    bedrooms = [r for r in rooms if "bedroom" in r.get("type","").lower()]
    public   = [r for r in rooms if any(t in r.get("type","").lower()
                                        for t in ["living","kitchen","dining","foyer","garage"])]
    if not bedrooms or not public:
        return 80.0
    scores = []
    for bed in bedrooms:
        d = min(_dist(bed, p) for p in public)
        scores.append(100 if d >= 20 else 80 if d >= 12 else 55 if d >= 6 else 30)
    return round(sum(scores) / len(scores), 1)


def _score_efficiency(rooms, total_w, total_h):
    used = sum(r["width"] * r["height"] for r in rooms)
    eff  = used / max(1, total_w * total_h)
    if 0.80 <= eff <= 0.98: return 95.0
    if 0.70 <= eff <  0.80: return 80.0
    if eff > 0.98:          return 70.0
    return 60.0


def _grade(score):
    if score >= 90: return "A"
    if score >= 80: return "B"
    if score >= 70: return "C"
    if score >= 60: return "D"
    return "F"


def _insights(scores, rooms):
    insights = []
    if scores["adjacency"]     < 70: insights.append("Consider moving kitchen closer to dining/living area.")
    if scores["natural_light"] < 70: insights.append("Some rooms lack perimeter walls — shift bedrooms toward exterior edges.")
    if scores["circulation"]   < 70: insights.append("Add an entry foyer or hallway for better traffic flow.")
    if scores["privacy"]       < 70: insights.append("Bedrooms are close to public zones — consider a bedroom wing.")
    if scores["efficiency"]    < 70: insights.append("Rooms don't fully utilize the footprint — check for gaps.")
    if not insights:
        insights.append("Excellent layout! Great room placement and circulation.")
    return insights
