"""
Improved Training Dataset Generator for Buildify MOE.

Strategy:
  1. Generate diverse US house configurations (constraints)
  2. Build bubble diagrams for each
  3. Run HouseGAN++ inference to get spatial layouts
  4. Extract room dimension statistics from layouts
  5. Combine with IRC rules for final training samples

This gives MOE access to "real" room proportion distributions
learned by HouseGAN++ (from the RPLAN dataset) rather than just
synthetic IRC minimums.

Usage:
  python -m moe.training.generate_dataset --samples 50000 --out data/train.pt
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import uuid
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

# Add parent to path so we can import without installing
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from moe.config import MOEConfig
from moe.data import encode_constraints, IRC_ROOM_SPECS, ADJACENCY_RULES
from moe.housegan.bubble_diagram import build_bubble_diagram
from moe.housegan.inference import generate_layouts


# ── Constraint sampler ────────────────────────────────────────────────────────

STYLES = ["modern", "traditional", "craftsman", "ranch",
          "farmhouse", "contemporary", "colonial", "cape_cod"]
GARAGES = ["none", "1car", "2car", "2car", "3car"]  # weighted toward 2car
LAUNDRY  = ["none", "closet", "room", "room"]
OUTDOOR  = ["none", "patio", "deck", "both"]


def sample_constraints() -> dict:
    """Sample a random but realistic set of US house constraints."""
    beds  = random.choices([1,2,3,3,4,4,5,6], k=1)[0]
    baths = max(1, min(beds, random.randint(1, beds + 1)))
    sqft  = random.randint(800, 5500)
    # Correlate sqft with beds: larger beds → larger sqft
    sqft  = max(sqft, beds * 350)

    return {
        "sqft":         sqft,
        "bedrooms":     beds,
        "bathrooms":    baths,
        "stories":      1,
        "style":        random.choice(STYLES),
        "openPlan":     random.random() < 0.45,
        "primarySuite": beds >= 2 and random.random() < 0.75,
        "homeOffice":   random.random() < 0.30,
        "formalDining": random.random() < 0.25,
        "garage":       random.choice(GARAGES),
        "laundry":      random.choice(LAUNDRY),
        "outdoor":      random.choice(OUTDOOR),
        "ceilingHeight": random.choice(["standard", "standard", "high", "vaulted"]),
    }


# ── IRC-based target generator (fallback) ────────────────────────────────────

def irc_room_targets(room_type: str, sqft: int) -> Tuple[float, float]:
    """
    Return (width, height) target for a room type based on home sqft.
    Interpolates between IRC minimum, standard, and premium.
    """
    spec = IRC_ROOM_SPECS.get(room_type)
    if spec is None:
        return (10.0, 10.0)

    min_w, min_h, std_w, std_h, prem_w, prem_h = spec

    if sqft < 1200:
        t = 0.0
    elif sqft < 2000:
        t = (sqft - 1200) / 800
    elif sqft < 3500:
        t = 0.5 + (sqft - 2000) / 3000
    else:
        t = 1.0
    t = min(1.0, t)

    if t <= 0.5:
        t2 = t * 2
        w = min_w + t2 * (std_w - min_w)
        h = min_h + t2 * (std_h - min_h)
    else:
        t2 = (t - 0.5) * 2
        w = std_w + t2 * (prem_w - std_w)
        h = std_h + t2 * (prem_h - std_h)

    # Add ±8% jitter
    w *= random.uniform(0.92, 1.08)
    h *= random.uniform(0.92, 1.08)
    return (round(w, 1), round(h, 1))


# ── Sample → training tuple ───────────────────────────────────────────────────

async def constraints_to_training_sample(
    constraints: dict,
    use_housegan: bool = True,
) -> Dict:
    """
    Convert constraints into a training sample.

    Returns:
      {
        "constraint_vec": (20,) float32 tensor,
        "rooms": [{"type": str, "width": float, "height": float, "zone": str, ...}]
        "footprint": (W, H),
        "adj_pairs": [(i, j), ...],
      }
    """
    config = MOEConfig()
    sqft = constraints["sqft"]

    # Encode constraints to vector
    constraint_vec = encode_constraints(constraints)

    # Build bubble diagram
    diagram = build_bubble_diagram(constraints)

    rooms_out = []

    if use_housegan:
        # Try HouseGAN++ for spatial layout
        try:
            layouts = await generate_layouts(diagram, num_variants=1, mode="auto")
            if layouts:
                layout = layouts[0]
                for placed_room in layout:
                    rooms_out.append({
                        "id":     placed_room["id"],
                        "name":   placed_room["name"],
                        "type":   placed_room["type"],
                        "width":  placed_room["width"],
                        "height": placed_room["height"],
                        "x":      placed_room["x"],
                        "y":      placed_room["y"],
                        "zone":   placed_room["zone"],
                    })
        except Exception:
            pass  # Fall through to IRC-based

    # Fallback: IRC-based dimensions if HouseGAN unavailable
    if not rooms_out:
        for room in diagram.rooms:
            w, h = irc_room_targets(room.buildify_type, sqft)
            rooms_out.append({
                "id":     room.id,
                "name":   room.name,
                "type":   room.buildify_type,
                "width":  w,
                "height": h,
                "x":      0.0,
                "y":      0.0,
                "zone":   room.zone,
            })

    # Build adjacency pair list
    adj_pairs = []
    n = diagram.n
    for i in range(n):
        for j in range(i+1, n):
            if diagram.adj_matrix[i, j] > 0.4:
                adj_pairs.append((i, j, float(diagram.adj_matrix[i, j])))

    return {
        "constraint_vec":  torch.tensor(constraint_vec, dtype=torch.float32),
        "rooms":           rooms_out,
        "footprint":       (diagram.house_w, diagram.house_h),
        "adj_pairs":       adj_pairs,
        "constraints":     constraints,
    }


# ── Dataset class ─────────────────────────────────────────────────────────────

class BuildifyDataset(Dataset):
    """
    PyTorch Dataset of (constraint_vector, room_layout) pairs.
    Can be loaded from a cached .pt file or generated on-the-fly.
    """

    def __init__(self, samples: List[Dict], config: MOEConfig = None):
        self.samples = samples
        self.config  = config or MOEConfig()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        return {
            "constraints":  s["constraint_vec"],
            "rooms":        s["rooms"],
            "footprint":    s["footprint"],
            "adj_pairs":    s["adj_pairs"],
        }

    @classmethod
    def from_file(cls, path: str) -> "BuildifyDataset":
        data = torch.load(path, weights_only=False)
        return cls(data["samples"], MOEConfig())

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"samples": self.samples}, path)
        print(f"[Dataset] Saved {len(self.samples)} samples → {path}")


# ── Generator entry point ─────────────────────────────────────────────────────

async def generate_dataset_async(
    num_samples: int = 10_000,
    use_housegan: bool = True,
    out_path: str = "moe/data/train.pt",
    log_every: int = 500,
) -> BuildifyDataset:
    """Generate a training dataset asynchronously."""
    print(f"[Dataset] Generating {num_samples} samples "
          f"(HouseGAN={'on' if use_housegan else 'off'})...")

    samples = []
    errors  = 0

    for i in range(num_samples):
        c = sample_constraints()
        try:
            s = await constraints_to_training_sample(c, use_housegan)
            if s["rooms"]:
                samples.append(s)
        except Exception as e:
            errors += 1

        if (i + 1) % log_every == 0:
            print(f"  {i+1}/{num_samples}  valid={len(samples)}  errors={errors}")

    print(f"[Dataset] Done — {len(samples)} valid / {num_samples} attempted")
    ds = BuildifyDataset(samples)
    ds.save(out_path)
    return ds


def generate_dataset(num_samples=10_000, use_housegan=True,
                     out_path="moe/data/train.pt"):
    """Sync wrapper for generate_dataset_async."""
    return asyncio.run(generate_dataset_async(num_samples, use_housegan, out_path))


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Buildify MOE training data")
    parser.add_argument("--samples",     type=int,  default=10_000)
    parser.add_argument("--no-housegan", action="store_true")
    parser.add_argument("--out",         type=str,  default="moe/data/train.pt")
    args = parser.parse_args()

    asyncio.run(generate_dataset_async(
        num_samples=args.samples,
        use_housegan=not args.no_housegan,
        out_path=args.out,
    ))
