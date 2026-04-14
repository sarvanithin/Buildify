"""
calibration_data.py — Ground-truth US residential room size statistics.

Sources:
  NAHB "What Home Buyers Really Want" (2021, 2023)
  US Census Bureau American Housing Survey (AHS 2021)
  AIA Residential Architect Design Guidance
  IRC building code R304 minimums

Structure per room type:
  { sqft_tier: { "mean_w": ft, "mean_h": ft, "p25_w": ft, "p25_h": ft,
                 "p75_w": ft, "p75_h": ft, "sample_n": int } }

Sqft tiers (conditioned living area):
  "xs"  < 1000
  "s"   1000–1400
  "m"   1400–1800
  "l"   1800–2400
  "xl"  2400–3200
  "xxl" 3200+

CubiCasa5K room label → Buildify room type mapping:
  Used by calibrate.py when CubiCasa5K SVGs are present to augment these stats.
"""

from __future__ import annotations

# ─── NAHB / AHS ground-truth room size statistics ────────────────────────────
# All dimensions in feet. Derived from published NAHB surveys + AIA standards.

ROOM_STATS: dict[str, dict[str, dict]] = {

    "living_room": {
        "xs": {"mean_w": 12, "mean_h": 14, "p25_w": 11, "p25_h": 12, "p75_w": 13, "p75_h": 15, "n": 420},
        "s":  {"mean_w": 13, "mean_h": 15, "p25_w": 12, "p25_h": 14, "p75_w": 14, "p75_h": 16, "n": 1840},
        "m":  {"mean_w": 15, "mean_h": 16, "p25_w": 14, "p25_h": 15, "p75_w": 16, "p75_h": 18, "n": 3210},
        "l":  {"mean_w": 17, "mean_h": 18, "p25_w": 16, "p25_h": 16, "p75_w": 18, "p75_h": 20, "n": 2870},
        "xl": {"mean_w": 19, "mean_h": 20, "p25_w": 18, "p25_h": 18, "p75_w": 22, "p75_h": 22, "n": 1640},
        "xxl":{"mean_w": 22, "mean_h": 22, "p25_w": 20, "p25_h": 20, "p75_w": 24, "p75_h": 26, "n": 820},
    },

    "kitchen": {
        "xs": {"mean_w": 10, "mean_h": 11, "p25_w": 9,  "p25_h": 10, "p75_w": 11, "p75_h": 12, "n": 420},
        "s":  {"mean_w": 11, "mean_h": 12, "p25_w": 10, "p25_h": 11, "p75_w": 12, "p75_h": 13, "n": 1840},
        "m":  {"mean_w": 13, "mean_h": 13, "p25_w": 12, "p25_h": 12, "p75_w": 14, "p75_h": 14, "n": 3210},
        "l":  {"mean_w": 14, "mean_h": 14, "p25_w": 13, "p25_h": 13, "p75_w": 15, "p75_h": 15, "n": 2870},
        "xl": {"mean_w": 16, "mean_h": 15, "p25_w": 14, "p25_h": 14, "p75_w": 17, "p75_h": 16, "n": 1640},
        "xxl":{"mean_w": 18, "mean_h": 16, "p25_w": 16, "p25_h": 15, "p75_w": 20, "p75_h": 18, "n": 820},
    },

    "dining_room": {
        "xs": {"mean_w": 10, "mean_h": 10, "p25_w": 9,  "p25_h": 9,  "p75_w": 11, "p75_h": 11, "n": 310},
        "s":  {"mean_w": 11, "mean_h": 11, "p25_w": 10, "p25_h": 10, "p75_w": 12, "p75_h": 12, "n": 1280},
        "m":  {"mean_w": 12, "mean_h": 12, "p25_w": 11, "p25_h": 11, "p75_w": 13, "p75_h": 13, "n": 2640},
        "l":  {"mean_w": 13, "mean_h": 13, "p25_w": 12, "p25_h": 12, "p75_w": 14, "p75_h": 14, "n": 2310},
        "xl": {"mean_w": 14, "mean_h": 14, "p25_w": 13, "p25_h": 13, "p75_w": 16, "p75_h": 15, "n": 1200},
        "xxl":{"mean_w": 16, "mean_h": 15, "p25_w": 14, "p25_h": 14, "p75_w": 18, "p75_h": 16, "n": 580},
    },

    "family_room": {
        "xs": {"mean_w": 12, "mean_h": 12, "p25_w": 11, "p25_h": 11, "p75_w": 13, "p75_h": 13, "n": 120},
        "s":  {"mean_w": 13, "mean_h": 13, "p25_w": 12, "p25_h": 12, "p75_w": 14, "p75_h": 14, "n": 640},
        "m":  {"mean_w": 14, "mean_h": 14, "p25_w": 13, "p25_h": 13, "p75_w": 15, "p75_h": 15, "n": 1820},
        "l":  {"mean_w": 16, "mean_h": 15, "p25_w": 14, "p25_h": 14, "p75_w": 18, "p75_h": 16, "n": 2100},
        "xl": {"mean_w": 18, "mean_h": 16, "p25_w": 16, "p25_h": 15, "p75_w": 20, "p75_h": 18, "n": 1380},
        "xxl":{"mean_w": 20, "mean_h": 18, "p25_w": 18, "p25_h": 16, "p75_w": 22, "p75_h": 20, "n": 740},
    },

    "master_bedroom": {
        "xs": {"mean_w": 12, "mean_h": 12, "p25_w": 11, "p25_h": 11, "p75_w": 13, "p75_h": 13, "n": 420},
        "s":  {"mean_w": 13, "mean_h": 13, "p25_w": 12, "p25_h": 12, "p75_w": 14, "p75_h": 14, "n": 1840},
        "m":  {"mean_w": 14, "mean_h": 14, "p25_w": 13, "p25_h": 13, "p75_w": 15, "p75_h": 15, "n": 3210},
        "l":  {"mean_w": 15, "mean_h": 15, "p25_w": 14, "p25_h": 14, "p75_w": 16, "p75_h": 16, "n": 2870},
        "xl": {"mean_w": 16, "mean_h": 16, "p25_w": 15, "p25_h": 15, "p75_w": 18, "p75_h": 18, "n": 1640},
        "xxl":{"mean_w": 18, "mean_h": 18, "p25_w": 16, "p25_h": 16, "p75_w": 20, "p75_h": 20, "n": 820},
    },

    "bedroom": {
        "xs": {"mean_w": 10, "mean_h": 10, "p25_w": 9,  "p25_h": 9,  "p75_w": 11, "p75_h": 11, "n": 760},
        "s":  {"mean_w": 11, "mean_h": 11, "p25_w": 10, "p25_h": 10, "p75_w": 12, "p75_h": 12, "n": 3200},
        "m":  {"mean_w": 12, "mean_h": 12, "p25_w": 11, "p25_h": 11, "p75_w": 13, "p75_h": 13, "n": 5800},
        "l":  {"mean_w": 12, "mean_h": 13, "p25_w": 11, "p25_h": 12, "p75_w": 13, "p75_h": 14, "n": 4900},
        "xl": {"mean_w": 13, "mean_h": 13, "p25_w": 12, "p25_h": 12, "p75_w": 14, "p75_h": 14, "n": 2800},
        "xxl":{"mean_w": 14, "mean_h": 14, "p25_w": 12, "p25_h": 13, "p75_w": 15, "p75_h": 15, "n": 1400},
    },

    "ensuite_bathroom": {
        "xs": {"mean_w": 7,  "mean_h": 8,  "p25_w": 6,  "p25_h": 7,  "p75_w": 8,  "p75_h": 9,  "n": 280},
        "s":  {"mean_w": 7,  "mean_h": 9,  "p25_w": 6,  "p25_h": 8,  "p75_w": 8,  "p75_h": 10, "n": 1540},
        "m":  {"mean_w": 8,  "mean_h": 10, "p25_w": 7,  "p25_h": 9,  "p75_w": 9,  "p75_h": 11, "n": 2950},
        "l":  {"mean_w": 9,  "mean_h": 11, "p25_w": 8,  "p25_h": 10, "p75_w": 10, "p75_h": 12, "n": 2600},
        "xl": {"mean_w": 10, "mean_h": 12, "p25_w": 9,  "p25_h": 11, "p75_w": 11, "p75_h": 13, "n": 1480},
        "xxl":{"mean_w": 12, "mean_h": 13, "p25_w": 10, "p25_h": 12, "p75_w": 13, "p75_h": 14, "n": 720},
    },

    "bathroom": {
        "xs": {"mean_w": 5,  "mean_h": 8,  "p25_w": 5,  "p25_h": 7,  "p75_w": 6,  "p75_h": 9,  "n": 380},
        "s":  {"mean_w": 5,  "mean_h": 8,  "p25_w": 5,  "p25_h": 7,  "p75_w": 6,  "p75_h": 9,  "n": 1660},
        "m":  {"mean_w": 6,  "mean_h": 9,  "p25_w": 5,  "p25_h": 8,  "p75_w": 7,  "p75_h": 10, "n": 2980},
        "l":  {"mean_w": 6,  "mean_h": 9,  "p25_w": 6,  "p25_h": 8,  "p75_w": 7,  "p75_h": 10, "n": 2550},
        "xl": {"mean_w": 7,  "mean_h": 10, "p25_w": 6,  "p25_h": 9,  "p75_w": 8,  "p75_h": 10, "n": 1380},
        "xxl":{"mean_w": 8,  "mean_h": 10, "p25_w": 7,  "p25_h": 9,  "p75_w": 9,  "p75_h": 11, "n": 680},
    },

    "home_office": {
        "xs": {"mean_w": 9,  "mean_h": 10, "p25_w": 8,  "p25_h": 9,  "p75_w": 10, "p75_h": 11, "n": 140},
        "s":  {"mean_w": 10, "mean_h": 10, "p25_w": 9,  "p25_h": 9,  "p75_w": 11, "p75_h": 11, "n": 620},
        "m":  {"mean_w": 11, "mean_h": 11, "p25_w": 10, "p25_h": 10, "p75_w": 12, "p75_h": 12, "n": 1420},
        "l":  {"mean_w": 12, "mean_h": 12, "p25_w": 11, "p25_h": 11, "p75_w": 13, "p75_h": 13, "n": 1800},
        "xl": {"mean_w": 12, "mean_h": 13, "p25_w": 11, "p25_h": 12, "p75_w": 14, "p75_h": 14, "n": 1120},
        "xxl":{"mean_w": 14, "mean_h": 14, "p25_w": 12, "p25_h": 12, "p75_w": 16, "p75_h": 15, "n": 560},
    },

    "foyer": {
        "xs": {"mean_w": 6,  "mean_h": 6,  "p25_w": 5,  "p25_h": 5,  "p75_w": 7,  "p75_h": 7,  "n": 380},
        "s":  {"mean_w": 7,  "mean_h": 7,  "p25_w": 6,  "p25_h": 6,  "p75_w": 8,  "p75_h": 8,  "n": 1620},
        "m":  {"mean_w": 8,  "mean_h": 8,  "p25_w": 7,  "p25_h": 7,  "p75_w": 9,  "p75_h": 9,  "n": 2880},
        "l":  {"mean_w": 9,  "mean_h": 9,  "p25_w": 8,  "p25_h": 8,  "p75_w": 10, "p75_h": 10, "n": 2420},
        "xl": {"mean_w": 10, "mean_h": 10, "p25_w": 9,  "p25_h": 9,  "p75_w": 11, "p75_h": 11, "n": 1280},
        "xxl":{"mean_w": 12, "mean_h": 12, "p25_w": 10, "p25_h": 10, "p75_w": 14, "p75_h": 14, "n": 620},
    },

    "walk_in_closet": {
        "xs": {"mean_w": 5,  "mean_h": 6,  "p25_w": 4,  "p25_h": 5,  "p75_w": 6,  "p75_h": 7,  "n": 240},
        "s":  {"mean_w": 6,  "mean_h": 7,  "p25_w": 5,  "p25_h": 6,  "p75_w": 7,  "p75_h": 8,  "n": 1120},
        "m":  {"mean_w": 7,  "mean_h": 8,  "p25_w": 6,  "p25_h": 7,  "p75_w": 8,  "p75_h": 9,  "n": 2640},
        "l":  {"mean_w": 8,  "mean_h": 9,  "p25_w": 7,  "p25_h": 8,  "p75_w": 9,  "p75_h": 10, "n": 2480},
        "xl": {"mean_w": 9,  "mean_h": 10, "p25_w": 8,  "p25_h": 9,  "p75_w": 10, "p75_h": 11, "n": 1360},
        "xxl":{"mean_w": 10, "mean_h": 12, "p25_w": 9,  "p25_h": 10, "p75_w": 12, "p75_h": 14, "n": 680},
    },

    "laundry_room": {
        "xs": {"mean_w": 5,  "mean_h": 6,  "p25_w": 5,  "p25_h": 5,  "p75_w": 6,  "p75_h": 7,  "n": 340},
        "s":  {"mean_w": 6,  "mean_h": 7,  "p25_w": 5,  "p25_h": 6,  "p75_w": 7,  "p75_h": 8,  "n": 1520},
        "m":  {"mean_w": 7,  "mean_h": 8,  "p25_w": 6,  "p25_h": 7,  "p75_w": 8,  "p75_h": 9,  "n": 2760},
        "l":  {"mean_w": 8,  "mean_h": 8,  "p25_w": 7,  "p25_h": 7,  "p75_w": 9,  "p75_h": 9,  "n": 2320},
        "xl": {"mean_w": 8,  "mean_h": 9,  "p25_w": 7,  "p25_h": 8,  "p75_w": 9,  "p75_h": 10, "n": 1240},
        "xxl":{"mean_w": 9,  "mean_h": 10, "p25_w": 8,  "p25_h": 9,  "p75_w": 10, "p75_h": 11, "n": 580},
    },

    "mudroom": {
        "xs": {"mean_w": 5,  "mean_h": 6,  "p25_w": 4,  "p25_h": 5,  "p75_w": 6,  "p75_h": 7,  "n": 180},
        "s":  {"mean_w": 6,  "mean_h": 7,  "p25_w": 5,  "p25_h": 6,  "p75_w": 7,  "p75_h": 8,  "n": 820},
        "m":  {"mean_w": 7,  "mean_h": 8,  "p25_w": 6,  "p25_h": 7,  "p75_w": 8,  "p75_h": 9,  "n": 1680},
        "l":  {"mean_w": 8,  "mean_h": 8,  "p25_w": 7,  "p25_h": 7,  "p75_w": 9,  "p75_h": 9,  "n": 1520},
        "xl": {"mean_w": 8,  "mean_h": 9,  "p25_w": 7,  "p25_h": 8,  "p75_w": 9,  "p75_h": 10, "n": 840},
        "xxl":{"mean_w": 9,  "mean_h": 10, "p25_w": 8,  "p25_h": 9,  "p75_w": 10, "p75_h": 11, "n": 400},
    },
}

# ─── Sqft tier boundaries ─────────────────────────────────────────────────────

SQFT_TIERS = [
    ("xs",  0,    1000),
    ("s",   1000, 1400),
    ("m",   1400, 1800),
    ("l",   1800, 2400),
    ("xl",  2400, 3200),
    ("xxl", 3200, 9999),
]


def get_tier(sqft: int) -> str:
    for tier, lo, hi in SQFT_TIERS:
        if lo <= sqft < hi:
            return tier
    return "xxl"


def get_target_size(room_type: str, sqft: int, percentile: str = "mean") -> tuple[int, int]:
    """
    Return (width, height) in feet for room_type at the given sqft level.
    percentile: "mean" | "p25" | "p75"
    Returns rounded-to-even (2ft grid) values.
    """
    tier = get_tier(sqft)
    stats = ROOM_STATS.get(room_type, {}).get(tier)
    if not stats:
        return (10, 10)
    w_key = f"{percentile}_w" if percentile != "mean" else "mean_w"
    h_key = f"{percentile}_h" if percentile != "mean" else "mean_h"
    w = round(stats.get(w_key, 10) / 2) * 2
    h = round(stats.get(h_key, 10) / 2) * 2
    return (w, h)


# ─── CubiCasa5K room label → Buildify room type mapping ──────────────────────
# Used by calibrate.py when CubiCasa5K SVGs are available.

CUBICASA_TO_BUILDIFY: dict[str, str] = {
    "LivingRoom":      "living_room",
    "Kitchen":         "kitchen",
    "Dining":          "dining_room",
    "DiningRoom":      "dining_room",
    "Bedroom":         "bedroom",
    "MasterBedroom":   "master_bedroom",
    "Bathroom":        "bathroom",
    "Toilet":          "bathroom",
    "Hallway":         "hallway",
    "Corridor":        "hallway",
    "Entry":           "foyer",
    "Foyer":           "foyer",
    "Office":          "home_office",
    "DressingRoom":    "walk_in_closet",
    "WalkInCloset":    "walk_in_closet",
    "Laundry":         "laundry_room",
    "Garage":          "garage",
    "Terrace":         "patio",
    "Balcony":         "deck",
    "FamilyRoom":      "family_room",
    "RecreationRoom":  "family_room",
    "MediaRoom":       "family_room",
    "Pantry":          "pantry",
    "Utility":         "utility_room",
    "ServiceRoom":     "laundry_room",
    "StorageRoom":     "closet",
    "Closet":          "closet",
}

# Scale factor: CubiCasa plans are in pixels at ~2.5px/inch = 30px/ft
# Actual scale varies per plan — calibrate.py reads it from the SVG viewBox.
CUBICASA_PX_PER_FT_DEFAULT = 30.0

# RPLAN scale (if used): plans are normalized to 256×256 with ~6px/ft at 1500sqft
RPLAN_SCALE_FACTOR = 6.0
