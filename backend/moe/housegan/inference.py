"""
HouseGAN++ Inference — generates spatial floor plan layouts from bubble diagrams.

Pipeline:
  BubbleDiagram → HouseGAN++ generator → binary masks → bounding boxes
               → scale to real feet → US house post-processing

Two modes:
  1. Local:  loads model weights locally (backend/moe/housegan/weights/)
  2. Remote: calls HuggingFace Space API endpoint (free ZeroGPU)
"""
from __future__ import annotations

import os
import json
import uuid
import math
import time
import httpx
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from .bubble_diagram import BubbleDiagram, BubbleRoom
from .model import HouseGANGenerator, MASK_SIZE, NUM_ROOM_TYPES

# ── Constants ─────────────────────────────────────────────────────────────────
WEIGHTS_DIR   = Path(__file__).parent / "weights"
HF_SPACE_URL  = os.getenv(
    "HOUSEGAN_HF_URL",
    "https://buildify-housegan.hf.space/api/predict"
)
WALL_BUFFER   = 0.5   # ft buffer from house edge


# ── Mask → Bounding Box ──────────────────────────────────────────────────────

def masks_to_bboxes(masks: np.ndarray,
                    threshold: float = 0.5) -> List[Optional[Tuple[float,float,float,float]]]:
    """
    Convert (N, 1, 64, 64) float masks to bounding boxes.
    Returns list of (x1, y1, x2, y2) normalised 0-1, or None if empty mask.
    """
    bboxes = []
    for mask in masks:
        m = (mask[0] >= threshold).astype(np.uint8)
        ys, xs = np.where(m)
        if len(xs) == 0:
            bboxes.append(None)
        else:
            x1 = float(xs.min()) / MASK_SIZE
            y1 = float(ys.min()) / MASK_SIZE
            x2 = float(xs.max() + 1) / MASK_SIZE
            y2 = float(ys.max() + 1) / MASK_SIZE
            bboxes.append((x1, y1, x2, y2))
    return bboxes


def scale_bboxes_to_feet(bboxes: List[Optional[Tuple]],
                          rooms: List[BubbleRoom],
                          W: float, H: float) -> List[Dict]:
    """
    Scale normalised bounding boxes to real-world feet.
    Enforces minimum dimensions from IRC specs.
    Returns list of room dicts compatible with floor plan JSON.
    """
    placed = []
    for room, bbox in zip(rooms, bboxes):
        if bbox is None:
            # Fallback: place in a default position
            x, y = WALL_BUFFER, WALL_BUFFER
            w = max(room.min_w, 10.0)
            h = max(room.min_h, 10.0)
        else:
            x1, y1, x2, y2 = bbox
            x = round(x1 * W, 1)
            y = round(y1 * H, 1)
            w = round((x2 - x1) * W, 1)
            h = round((y2 - y1) * H, 1)

            # Enforce IRC minimums
            w = max(w, room.min_w)
            h = max(h, room.min_h)

            # Clamp to footprint
            x = max(WALL_BUFFER, min(x, W - w - WALL_BUFFER))
            y = max(WALL_BUFFER, min(y, H - h - WALL_BUFFER))

        placed.append({
            "id":     room.id,
            "name":   room.name,
            "type":   room.buildify_type,
            "x":      x,
            "y":      y,
            "width":  w,
            "height": h,
            "zone":   room.zone,
        })
    return placed


# ── US House Post-Processor ───────────────────────────────────────────────────

def apply_us_conventions(placed: List[Dict], W: float, H: float) -> List[Dict]:
    """
    Apply US residential floor plan conventions on top of HouseGAN output.

    HouseGAN was trained on Chinese apartments (no garages, different culture).
    This fixes the most common issues:
      1. Garage → forced to front-left (y ≈ 0)
      2. Foyer → adjacent to garage, on exterior wall
      3. Patio/Deck → forced to rear (y ≈ H)
      4. Bedrooms → pushed toward the back half of the house
    """
    id_map = {r["id"]: r for r in placed}
    idx_map = {r["id"]: i for i, r in enumerate(placed)}

    def move(rid: str, target_y: float, target_x: Optional[float] = None):
        if rid not in id_map: return
        r = id_map[rid]
        r["y"] = round(max(WALL_BUFFER, min(target_y, H - r["height"] - WALL_BUFFER)), 1)
        if target_x is not None:
            r["x"] = round(max(WALL_BUFFER, min(target_x, W - r["width"] - WALL_BUFFER)), 1)

    # Garage: top-left corner
    if "garage" in id_map:
        g = id_map["garage"]
        g["x"], g["y"] = WALL_BUFFER, WALL_BUFFER

        # Mudroom: right of garage
        if "mudroom" in id_map:
            m = id_map["mudroom"]
            m["x"] = round(g["x"] + g["width"] + 0.2, 1)
            m["y"] = WALL_BUFFER

    # Foyer: front, right of garage/mudroom
    if "foyer" in id_map:
        ref_x = WALL_BUFFER
        if "mudroom" in id_map:
            m = id_map["mudroom"]
            ref_x = m["x"] + m["width"] + 0.2
        elif "garage" in id_map:
            g = id_map["garage"]
            ref_x = g["x"] + g["width"] + 0.2
        move("foyer", WALL_BUFFER, ref_x)

    # Patio / Deck: rear (bottom of plan)
    for oid in ("patio", "deck"):
        if oid in id_map:
            r = id_map[oid]
            r["y"] = round(H - r["height"] - WALL_BUFFER, 1)

    # Bedrooms: back half
    for rid, r in id_map.items():
        if r["type"] in ("master_bedroom", "bedroom") and r["y"] < H * 0.4:
            r["y"] = round(H * 0.55, 1)

    return placed


def resolve_overlaps(placed: List[Dict], W: float, H: float,
                     max_iters: int = 20) -> List[Dict]:
    """
    Simple iterative overlap resolver: push overlapping rooms apart.
    Not perfect but good enough for post-HouseGAN cleanup.
    """
    def overlap(a, b):
        return (a["x"] < b["x"] + b["width"] and
                a["x"] + a["width"] > b["x"] and
                a["y"] < b["y"] + b["height"] and
                a["y"] + a["height"] > b["y"])

    def area_overlap(a, b):
        ox = min(a["x"]+a["width"], b["x"]+b["width"]) - max(a["x"], b["x"])
        oy = min(a["y"]+a["height"], b["y"]+b["height"]) - max(a["y"], b["y"])
        return max(0, ox) * max(0, oy)

    for _ in range(max_iters):
        moved = False
        for i, a in enumerate(placed):
            for j, b in enumerate(placed):
                if i >= j: continue
                if not overlap(a, b): continue

                # Push the smaller room away from the larger
                ao = area_overlap(a, b)
                if ao < 0.5: continue

                cx_a = a["x"] + a["width"]  / 2
                cx_b = b["x"] + b["width"]  / 2
                cy_a = a["y"] + a["height"] / 2
                cy_b = b["y"] + b["height"] / 2

                dx, dy = cx_b - cx_a, cy_b - cy_a
                if abs(dx) > abs(dy):
                    # Push horizontally
                    push = (a["width"] + b["width"]) / 2 - abs(dx) + 0.2
                    if dx > 0:
                        b["x"] = round(min(b["x"] + push, W - b["width"]), 1)
                    else:
                        b["x"] = round(max(b["x"] - push, 0), 1)
                else:
                    # Push vertically
                    push = (a["height"] + b["height"]) / 2 - abs(dy) + 0.2
                    if dy > 0:
                        b["y"] = round(min(b["y"] + push, H - b["height"]), 1)
                    else:
                        b["y"] = round(max(b["y"] - push, 0), 1)
                moved = True
        if not moved:
            break

    return placed


# ── Local Inference ───────────────────────────────────────────────────────────

_cached_model: Optional[HouseGANGenerator] = None


def _get_local_model() -> Optional[HouseGANGenerator]:
    """Load HouseGAN++ from local weights if available."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model

    weights = WEIGHTS_DIR / "housegan_pp.pt"
    if not weights.exists():
        return None

    try:
        from .model import load_pretrained
        _cached_model = load_pretrained(str(weights))
        return _cached_model
    except Exception as e:
        print(f"[HouseGAN] Failed to load local weights: {e}")
        return None


def _run_local(diagram: BubbleDiagram,
               num_samples: int = 5) -> List[List[Dict]]:
    """Run HouseGAN++ locally, return num_samples candidate layouts."""
    model = _get_local_model()
    if model is None:
        raise RuntimeError("No local HouseGAN++ weights found.")

    N = diagram.n
    room_types = torch.tensor(diagram.hg_type_vector, dtype=torch.long)
    adj = torch.tensor(diagram.binary_adj, dtype=torch.float32)

    results = []
    with torch.no_grad():
        for _ in range(num_samples):
            masks = model(room_types, adj)         # (N, 1, 64, 64)
            masks_np = masks.cpu().numpy()
            bboxes = masks_to_bboxes(masks_np)
            placed = scale_bboxes_to_feet(bboxes, diagram.rooms,
                                           diagram.house_w, diagram.house_h)
            placed = apply_us_conventions(placed, diagram.house_w, diagram.house_h)
            placed = resolve_overlaps(placed, diagram.house_w, diagram.house_h)
            results.append(placed)

    return results


# ── Remote Inference (HuggingFace Space) ─────────────────────────────────────

async def _run_remote(diagram: BubbleDiagram,
                      num_samples: int = 3) -> List[List[Dict]]:
    """Call HouseGAN++ running on Hugging Face Spaces (free ZeroGPU)."""
    payload = {
        "data": [
            diagram.hg_type_vector.tolist(),
            diagram.binary_adj.tolist(),
            diagram.house_w,
            diagram.house_h,
            num_samples,
        ]
    }

    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(HF_SPACE_URL, json=payload)
        resp.raise_for_status()
        data = resp.json()

    # HF Spaces Gradio API returns {"data": [...]}
    raw_layouts = data["data"][0]   # list of num_samples layouts

    results = []
    for layout in raw_layouts:
        placed = []
        for room, box in zip(diagram.rooms, layout):
            x1, y1, x2, y2 = box
            w = max(room.min_w, round((x2 - x1) * diagram.house_w, 1))
            h = max(room.min_h, round((y2 - y1) * diagram.house_h, 1))
            placed.append({
                "id": room.id, "name": room.name,
                "type": room.buildify_type,
                "x": round(x1 * diagram.house_w, 1),
                "y": round(y1 * diagram.house_h, 1),
                "width": w, "height": h,
                "zone": room.zone,
            })
        placed = apply_us_conventions(placed, diagram.house_w, diagram.house_h)
        placed = resolve_overlaps(placed, diagram.house_w, diagram.house_h)
        results.append(placed)

    return results


# ── Main entry point ──────────────────────────────────────────────────────────

async def generate_layouts(
    diagram: BubbleDiagram,
    num_variants: int = 3,
    mode: str = "auto",          # "auto" | "local" | "remote"
) -> List[List[Dict]]:
    """
    Generate floor plan room layouts from a bubble diagram.

    Returns num_variants candidate layouts, each as a list of room dicts
    with keys: id, name, type, x, y, width, height, zone.

    Falls back gracefully:
      local weights → HF remote → template fallback
    """
    # Try local first
    if mode in ("auto", "local"):
        model = _get_local_model()
        if model is not None:
            try:
                return _run_local(diagram, num_samples=num_variants)
            except Exception as e:
                print(f"[HouseGAN] Local inference failed: {e}")
                if mode == "local":
                    raise

    # Try remote HF Space
    if mode in ("auto", "remote"):
        try:
            return await _run_remote(diagram, num_samples=num_variants)
        except Exception as e:
            print(f"[HouseGAN] Remote inference failed: {e}")

    # Both failed — return empty (caller should fall back to template layout)
    return []
