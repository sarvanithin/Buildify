"""
Synthetic training data generator for the Buildify MOE model.

Generates 100K+ training samples from US IRC building codes,
architectural standards, and the Buildify arch_knowledge.json.
Each sample: (encoded constraints) → (optimal room layout).
"""
import json
import math
import random
from pathlib import Path
from typing import List, Tuple, Dict

import torch
from torch.utils.data import Dataset, DataLoader

from .config import MOEConfig

# ─────────────────────────────────────────────────────────────────────────────
# IRC Building Code Reference Data (US Residential)
# ─────────────────────────────────────────────────────────────────────────────

IRC_ROOM_SPECS = {
    # type: (min_w, min_h, std_w, std_h, premium_w, premium_h) in feet
    # Sizes based on IRC R304 minimums and US residential architectural standards
    "living_room":       (12, 14, 14, 18, 18, 22),
    "kitchen":           (10, 12, 12, 14, 14, 16),   # IRC R304: ≥70 sqft; 10×12=120 sqft standard
    "dining_room":       (11, 11, 12, 14, 14, 16),
    "family_room":       (12, 14, 14, 16, 16, 20),
    "master_bedroom":    (12, 14, 14, 15, 16, 18),   # Primary suite: 12×14=168 sqft minimum
    "bedroom":           (10, 10, 11, 12, 12, 14),   # IRC R304: ≥70 sqft; 10×10=100 sqft
    "ensuite_bathroom":  (7, 9, 9, 11, 11, 13),
    "bathroom":          (5, 8, 6, 9, 8, 10),
    "half_bath":         (3, 6, 4, 6, 5, 7),
    "hallway":           (4, 4, 4, 6, 5, 8),         # IRC R311.7: ≥36in (3ft) min; 4ft preferred
    "foyer":             (6, 6, 8, 8, 10, 12),
    "home_office":       (9, 10, 10, 12, 12, 14),
    "laundry_room":      (5, 6, 7, 8, 8, 10),
    "garage":            (20, 20, 24, 24, 32, 26),   # 1-car: 20×20; 2-car: 24×24 (IRC standard)
    "walk_in_closet":    (5, 7, 7, 8, 8, 10),
    "closet":            (3, 3, 4, 5, 5, 6),
    "pantry":            (3, 4, 4, 6, 5, 7),
    "mudroom":           (5, 6, 7, 8, 8, 10),
    "utility_room":      (5, 6, 6, 8, 8, 10),
    "patio":             (12, 14, 16, 20, 20, 24),
    "deck":              (10, 12, 14, 18, 16, 22),
}

# Adjacency preferences (room_a → room_b, strength 0-1)
ADJACENCY_RULES = [
    ("kitchen", "dining_room", 1.0),
    ("kitchen", "family_room", 0.9),
    ("kitchen", "living_room", 0.8),
    ("dining_room", "living_room", 0.8),
    ("master_bedroom", "ensuite_bathroom", 1.0),
    ("master_bedroom", "walk_in_closet", 1.0),
    ("bedroom", "bathroom", 0.7),
    ("bedroom", "closet", 0.8),
    ("foyer", "living_room", 0.9),
    ("garage", "kitchen", 0.6),
    ("garage", "mudroom", 0.9),
    ("mudroom", "kitchen", 0.7),
    ("laundry_room", "hallway", 0.6),
    ("living_room", "patio", 0.8),
    ("kitchen", "patio", 0.7),
    ("kitchen", "pantry", 0.9),
    ("hallway", "bedroom", 0.7),
]

# Zone assignments — US residential front-to-back architecture
# 0=ENTRY (garage+foyer front), 1=PUBLIC, 2=KITCHEN_SERVICE, 3=HALLWAY, 4=PRIVATE, 5=OUTDOOR
ZONE_MAP = {
    # Entry zone: garage front-left, mudroom and laundry tucked in service column
    "garage": 0, "foyer": 0, "mudroom": 0, "laundry_room": 0, "utility_room": 0,
    # Public zone: living areas visible from entry
    "living_room": 1, "dining_room": 1, "family_room": 1, "home_office": 1,
    # Kitchen/service zone: kitchen faces backyard, pantry adjacent
    "kitchen": 2, "pantry": 2,
    # Hallway zone: 4ft circulation spine separating public from private
    "hallway": 3, "half_bath": 3,
    # Private zone: master suite grouped, secondary bedrooms, shared baths
    "master_bedroom": 4, "bedroom": 4, "ensuite_bathroom": 4,
    "bathroom": 4, "walk_in_closet": 4, "closet": 4,
    # Outdoor zone: patio/deck at rear
    "patio": 5, "deck": 5,
}

# Style templates: which layout bands to use
STYLE_TEMPLATES = {
    "modern":       {"open_plan_weight": 0.9, "formal_rooms": False, "outdoor_emphasis": 0.7},
    "traditional":  {"open_plan_weight": 0.2, "formal_rooms": True,  "outdoor_emphasis": 0.4},
    "craftsman":    {"open_plan_weight": 0.5, "formal_rooms": False, "outdoor_emphasis": 0.6},
    "ranch":        {"open_plan_weight": 0.6, "formal_rooms": False, "outdoor_emphasis": 0.7},
    "farmhouse":    {"open_plan_weight": 0.7, "formal_rooms": False, "outdoor_emphasis": 0.8},
    "contemporary": {"open_plan_weight": 0.8, "formal_rooms": False, "outdoor_emphasis": 0.6},
    "colonial":     {"open_plan_weight": 0.3, "formal_rooms": True,  "outdoor_emphasis": 0.3},
    "cape_cod":     {"open_plan_weight": 0.4, "formal_rooms": True,  "outdoor_emphasis": 0.5},
}


# ─────────────────────────────────────────────────────────────────────────────
# Room list builder (what rooms go in a house given constraints)
# ─────────────────────────────────────────────────────────────────────────────

def _build_room_list(bedrooms: int, bathrooms: int, sqft: int,
                     style: str, open_plan: bool, primary_suite: bool,
                     home_office: bool, formal_dining: bool,
                     garage: str, laundry: str, outdoor: str) -> List[dict]:
    """Generate the list of rooms with target sizes for given constraints."""
    rooms = []
    room_id = 0
    rng = random.Random()  # seeded externally

    # Size factor based on total sqft
    sf = sqft / 2000.0  # 2000 sqft is baseline

    def _dims(room_type: str, scale: float = 1.0):
        specs = IRC_ROOM_SPECS[room_type]
        # Interpolate between min and premium based on sqft
        t = min(1.0, max(0.0, (sf - 0.7) / 0.8))
        w = specs[0] + (specs[4] - specs[0]) * t
        h = specs[1] + (specs[5] - specs[1]) * t
        return round(w * scale, 1), round(h * scale, 1)

    def _add(rtype, name=None, scale=1.0):
        nonlocal room_id
        w, h = _dims(rtype, scale)
        rooms.append({
            "id": f"r{room_id}",
            "type": rtype,
            "name": name or rtype.replace("_", " ").title(),
            "target_w": w, "target_h": h,
            "zone": ZONE_MAP.get(rtype, 0),
            "is_exterior": 1 if rtype in ("patio", "deck") else 0,
        })
        room_id += 1

    # Living areas
    if open_plan:
        _add("living_room", "Great Room", scale=1.3)
    else:
        _add("living_room")
        if formal_dining:
            _add("dining_room", "Formal Dining")
        _add("family_room")

    _add("kitchen")
    if not (open_plan and not formal_dining):
        _add("dining_room", "Dining Area")

    # Entry
    _add("foyer", "Entry Foyer")

    # Bedrooms
    if primary_suite:
        _add("master_bedroom", "Primary Suite")
        _add("ensuite_bathroom", "Primary Bath")
        _add("walk_in_closet", "Primary Closet")
    else:
        _add("master_bedroom", "Primary Bedroom")
        _add("bathroom", "Primary Bath")
        _add("closet", "Primary Closet")

    for i in range(1, bedrooms):
        _add("bedroom", f"Bedroom {i + 1}")
        _add("closet", f"Closet {i + 1}")

    # Bathrooms
    shared_baths = max(0, bathrooms - (1 if primary_suite or bedrooms >= 1 else 0))
    for i in range(shared_baths):
        if i == 0 and shared_baths >= 1:
            _add("bathroom", "Shared Bath")
        else:
            _add("half_bath", f"Half Bath")

    # Hallway
    _add("hallway", "Main Hallway")

    # Optional rooms
    if home_office:
        _add("home_office", "Home Office")

    # Laundry
    if laundry == "room":
        _add("laundry_room", "Laundry Room")
    elif laundry == "closet":
        _add("laundry_room", "Laundry Closet", scale=0.5)

    # Garage
    if garage == "1car":
        _add("garage", "1-Car Garage", scale=0.5)
    elif garage == "2car":
        _add("garage", "2-Car Garage")
    elif garage == "3car":
        _add("garage", "3-Car Garage", scale=1.3)
        _add("mudroom", "Mudroom")

    if garage != "none":
        _add("mudroom", "Mudroom")
        _add("pantry", "Pantry")

    # Outdoor
    if outdoor in ("patio", "both"):
        _add("patio", "Rear Patio")
    if outdoor in ("deck", "both"):
        _add("deck", "Rear Deck")

    return rooms


# ─────────────────────────────────────────────────────────────────────────────
# Layout solver (places rooms into x,y positions)
# ─────────────────────────────────────────────────────────────────────────────

def _solve_layout(rooms: List[dict], total_sqft: int,
                  style_template: dict) -> Tuple[List[dict], float, float]:
    """
    Simple zone-based layout solver:
    Places rooms in horizontal bands by zone (public, private, service, outdoor).
    Returns (placed_rooms, total_width, total_height).
    """
    # Calculate footprint from sqft (assume ~90% efficiency)
    conditioned_rooms = [r for r in rooms if not r["is_exterior"]]
    total_room_area = sum(r["target_w"] * r["target_h"] for r in conditioned_rooms)

    # Target aspect ratio based on style
    aspect = 1.4 if style_template.get("open_plan_weight", 0.5) > 0.6 else 1.2
    total_w = math.sqrt(total_room_area * aspect)
    total_h = total_room_area / total_w

    # Ensure reasonable bounds
    total_w = max(30, min(80, round(total_w)))
    total_h = max(25, min(70, round(total_h)))

    # Group rooms by zone (0=ENTRY, 1=PUBLIC, 2=KITCHEN, 3=HALLWAY, 4=PRIVATE, 5=OUTDOOR)
    zones = {0: [], 1: [], 2: [], 3: [], 4: [], 5: []}
    for r in rooms:
        zones[r["zone"]].append(r)

    # US residential front-to-back band order
    band_order = [0, 1, 2, 3, 4, 5]
    placed = []
    y_cursor = 0

    for zone_id in band_order:
        zone_rooms = zones[zone_id]
        if not zone_rooms:
            continue

        # Calculate band height from room sizes
        band_h = max((r["target_h"] for r in zone_rooms), default=10)
        x_cursor = 0

        for r in zone_rooms:
            # Scale width to fit within total_w
            w = min(r["target_w"], total_w - x_cursor)
            h = min(r["target_h"], band_h)

            if x_cursor + w > total_w:
                # Wrap to next row within the band
                x_cursor = 0
                y_cursor += band_h
                band_h = r["target_h"]

            placed.append({
                "id": r["id"],
                "type": r["type"],
                "name": r["name"],
                "x": round(x_cursor, 1),
                "y": round(y_cursor, 1),
                "width": round(w, 1),
                "height": round(h, 1),
                "zone": r["zone"],
                "is_exterior": r["is_exterior"],
            })
            x_cursor += w

        y_cursor += band_h

    total_h = max(total_h, y_cursor)

    return placed, total_w, total_h


# ─────────────────────────────────────────────────────────────────────────────
# Feature encoding (constraints → tensor)
# ─────────────────────────────────────────────────────────────────────────────

def encode_constraints(bedrooms: int, bathrooms: int, sqft: int,
                       stories: int, style: str, open_plan: bool,
                       primary_suite: bool, home_office: bool,
                       formal_dining: bool, garage: str,
                       laundry: str, outdoor: str,
                       ceiling_height: str,
                       config: MOEConfig) -> torch.Tensor:
    """Encode constraints into a fixed-size feature vector."""
    features = []

    # Numeric features (normalized)
    features.append(bedrooms / 6.0)
    features.append(bathrooms / 5.0)
    features.append(sqft / 5000.0)
    features.append(stories / 3.0)

    # Style one-hot (8 styles)
    style_idx = config.STYLES.index(style) if style in config.STYLES else 0
    style_vec = [0.0] * len(config.STYLES)
    style_vec[style_idx] = 1.0
    features.extend(style_vec)

    # Boolean features
    features.append(1.0 if open_plan else 0.0)
    features.append(1.0 if primary_suite else 0.0)
    features.append(1.0 if home_office else 0.0)
    features.append(1.0 if formal_dining else 0.0)

    # Garage encoding
    garage_map = {"none": 0.0, "1car": 0.33, "2car": 0.67, "3car": 1.0}
    features.append(garage_map.get(garage, 0.67))

    # Laundry encoding
    laundry_map = {"none": 0.0, "closet": 0.5, "room": 1.0}
    features.append(laundry_map.get(laundry, 1.0))

    # Outdoor encoding
    outdoor_map = {"none": 0.0, "patio": 0.33, "deck": 0.67, "both": 1.0}
    features.append(outdoor_map.get(outdoor, 0.33))

    # Ceiling height
    ceil_map = {"standard": 0.0, "high": 0.5, "vaulted": 1.0}
    features.append(ceil_map.get(ceiling_height, 0.0))

    return torch.tensor(features, dtype=torch.float32)


def encode_rooms(rooms: List[dict], total_w: float, total_h: float,
                 config: MOEConfig) -> torch.Tensor:
    """Encode a room list into target tensor (batch of room features)."""
    max_rooms = config.max_rooms
    room_features = config.room_features  # x, y, w, h, type_idx, zone, exterior

    tensor = torch.zeros(max_rooms, room_features)

    for i, r in enumerate(rooms[:max_rooms]):
        type_idx = config.ROOM_TYPES.index(r["type"]) if r["type"] in config.ROOM_TYPES else 0
        tensor[i] = torch.tensor([
            r["x"] / max(total_w, 1),
            r["y"] / max(total_h, 1),
            r["width"] / max(total_w, 1),
            r["height"] / max(total_h, 1),
            type_idx / max(config.num_room_types, 1),
            r["zone"] / 3.0,
            float(r["is_exterior"]),
        ])

    return tensor


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class FloorPlanDataset(Dataset):
    """
    Synthetic dataset of (constraints, optimal_layout) pairs.
    Generates data deterministically from IRC rules and style templates.
    """

    def __init__(self, num_samples: int, config: MOEConfig, seed: int = 42):
        self.config = config
        self.samples = []
        rng = random.Random(seed)

        # Parameter ranges
        bedroom_range = [2, 3, 4, 5, 6]
        bathroom_range = [1, 2, 3, 4]
        sqft_range = [1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3500, 4000]
        stories_range = [1, 2]
        styles = list(STYLE_TEMPLATES.keys())
        garage_options = ["none", "1car", "2car", "3car"]
        laundry_options = ["none", "closet", "room"]
        outdoor_options = ["none", "patio", "deck", "both"]
        ceiling_options = ["standard", "high", "vaulted"]

        for _ in range(num_samples):
            bedrooms = rng.choice(bedroom_range)
            bathrooms = min(rng.choice(bathroom_range), bedrooms)
            sqft = rng.choice(sqft_range)
            stories = rng.choice(stories_range)
            style = rng.choice(styles)
            open_plan = rng.random() < STYLE_TEMPLATES[style]["open_plan_weight"]
            primary_suite = rng.random() < 0.75
            home_office = rng.random() < 0.4
            formal_dining = rng.random() < (0.6 if not open_plan else 0.15)
            garage = rng.choice(garage_options)
            laundry = rng.choice(laundry_options)
            outdoor = rng.choice(outdoor_options)
            ceiling = rng.choice(ceiling_options)

            # Apply jitter for augmentation
            jitter = 1.0 + rng.uniform(-self.config.augmentation_jitter,
                                        self.config.augmentation_jitter)
            sqft_jittered = int(sqft * jitter)

            # Build room list and solve layout
            room_list = _build_room_list(
                bedrooms, bathrooms, sqft_jittered, style,
                open_plan, primary_suite, home_office, formal_dining,
                garage, laundry, outdoor,
            )

            style_template = STYLE_TEMPLATES.get(style, STYLE_TEMPLATES["modern"])
            placed, total_w, total_h = _solve_layout(
                room_list, sqft_jittered, style_template
            )

            # Encode
            constraint_vec = encode_constraints(
                bedrooms, bathrooms, sqft_jittered, stories, style,
                open_plan, primary_suite, home_office, formal_dining,
                garage, laundry, outdoor, ceiling, config
            )

            room_tensor = encode_rooms(placed, total_w, total_h, config)
            num_rooms = min(len(placed), config.max_rooms)

            # Room type targets
            type_targets = torch.zeros(config.max_rooms, dtype=torch.long)
            for i, r in enumerate(placed[:config.max_rooms]):
                idx = config.ROOM_TYPES.index(r["type"]) if r["type"] in config.ROOM_TYPES else 0
                type_targets[i] = idx

            # Coordinate targets (normalized)
            coord_targets = torch.zeros(config.max_rooms, 4)
            for i, r in enumerate(placed[:config.max_rooms]):
                coord_targets[i] = torch.tensor([
                    r["x"] / max(total_w, 1),
                    r["y"] / max(total_h, 1),
                    r["width"] / max(total_w, 1),
                    r["height"] / max(total_h, 1),
                ])

            self.samples.append({
                "constraints": constraint_vec,
                "room_tensor": room_tensor,
                "coord_targets": coord_targets,
                "type_targets": type_targets,
                "num_rooms": num_rooms,
                "total_w": total_w,
                "total_h": total_h,
                "metadata": {
                    "bedrooms": bedrooms, "bathrooms": bathrooms,
                    "sqft": sqft_jittered, "style": style,
                },
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return {
            "constraints": s["constraints"],
            "room_tensor": s["room_tensor"],
            "coord_targets": s["coord_targets"],
            "type_targets": s["type_targets"],
            "num_rooms": s["num_rooms"],
        }


def create_dataloaders(config: MOEConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    train_ds = FloorPlanDataset(config.num_train_samples, config, seed=42)
    val_ds   = FloorPlanDataset(config.num_val_samples,   config, seed=123)
    test_ds  = FloorPlanDataset(config.num_test_samples,  config, seed=456)

    train_dl = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True,
                          num_workers=0, pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=config.batch_size, shuffle=False,
                          num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=config.batch_size, shuffle=False,
                          num_workers=0)

    return train_dl, val_dl, test_dl
