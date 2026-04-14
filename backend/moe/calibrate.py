"""
calibrate.py — Room size calibration pipeline.

Modes:
  1. Embedded (default) — uses NAHB/AHS statistics in calibration_data.py
  2. CubiCasa5K         — parses SVG floor plans to extract real room dimensions
  3. Both               — merges CubiCasa stats (weighted 60%) with NAHB (40%)

Usage:
  python -m moe.calibrate                          # embedded NAHB data only
  python -m moe.calibrate --cubicasa /path/to/data # parse CubiCasa5K SVGs
  python -m moe.calibrate --write                  # write updated constants to data.py + inference.py
  python -m moe.calibrate --retrain                # calibrate + retrain model
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Optional

# ─── Calibration data ─────────────────────────────────────────────────────────

from .calibration_data import (
    ROOM_STATS, SQFT_TIERS, CUBICASA_TO_BUILDIFY,
    CUBICASA_PX_PER_FT_DEFAULT, get_tier,
)

# ─── CubiCasa SVG parser ──────────────────────────────────────────────────────

def _parse_cubicasa_svg(svg_path: Path) -> Optional[list[dict]]:
    """
    Parse one CubiCasa5K model.svg and return a list of
    {"type": buildify_type, "width_ft": float, "height_ft": float, "area_ft2": float}
    Returns None if parsing fails.
    """
    try:
        text = svg_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    # Extract viewBox to compute px→ft scale
    vb_match = re.search(r'viewBox="([^"]+)"', text)
    if not vb_match:
        return None
    vb = [float(x) for x in vb_match.group(1).split()]
    if len(vb) < 4:
        return None
    svg_w_px, svg_h_px = vb[2], vb[3]

    # CubiCasa plans are typically drawn at ~1:50 scale; SVG units ≈ mm in many exports
    # Heuristic: if largest dim > 500, likely px at ~30px/ft; else mm at ~300mm/ft
    max_dim = max(svg_w_px, svg_h_px)
    if max_dim > 500:
        px_per_ft = CUBICASA_PX_PER_FT_DEFAULT
    else:
        # millimetre units: 1ft = 304.8mm
        px_per_ft = 304.8

    rooms: list[dict] = []

    # Find room polygons: <polygon ... class="... RoomType ..." points="..."/>
    # or <path ... inkscape:label="RoomType" ... />
    for match in re.finditer(
        r'<(?:polygon|path|rect)[^>]+(?:class|inkscape:label)="([^"]*)"[^>]*(?:points|d|width)="([^"]*)"',
        text,
    ):
        label_raw = match.group(1)
        # Find matching CubiCasa room type
        btype = None
        for cubi_label, bt in CUBICASA_TO_BUILDIFY.items():
            if cubi_label.lower() in label_raw.lower():
                btype = bt
                break
        if not btype:
            continue

        # Try to compute bounding box from points
        pts_match = re.search(r'points="([^"]+)"', match.group(0))
        if not pts_match:
            # Try rect width/height
            w_m = re.search(r'width="([\d.]+)"', match.group(0))
            h_m = re.search(r'height="([\d.]+)"', match.group(0))
            if w_m and h_m:
                w_ft = float(w_m.group(1)) / px_per_ft
                h_ft = float(h_m.group(1)) / px_per_ft
                if 4 <= w_ft <= 60 and 4 <= h_ft <= 60:
                    rooms.append({"type": btype, "width_ft": w_ft, "height_ft": h_ft,
                                  "area_ft2": w_ft * h_ft})
            continue

        pts_str = pts_match.group(1)
        coords = re.findall(r"([\d.+-]+)[, ]([\d.+-]+)", pts_str)
        if len(coords) < 3:
            continue
        xs = [float(c[0]) for c in coords]
        ys = [float(c[1]) for c in coords]
        bb_w = (max(xs) - min(xs)) / px_per_ft
        bb_h = (max(ys) - min(ys)) / px_per_ft

        # Sanity filter: realistic US residential room sizes
        if 4 <= bb_w <= 60 and 4 <= bb_h <= 60:
            rooms.append({"type": btype, "width_ft": bb_w, "height_ft": bb_h,
                          "area_ft2": bb_w * bb_h})

    return rooms if rooms else None


def parse_cubicasa_dataset(data_dir: Path) -> dict[str, list[float]]:
    """
    Walk CubiCasa5K directory structure and collect room dimension samples.
    Returns {room_type: [area_ft2, ...]} (raw samples for percentile computation).
    """
    samples: dict[str, list[dict]] = {}
    svg_files = list(data_dir.rglob("model.svg"))
    print(f"[calibrate] Found {len(svg_files)} SVG files in {data_dir}")

    parsed = 0
    for svg_path in svg_files:
        rooms = _parse_cubicasa_svg(svg_path)
        if not rooms:
            continue
        parsed += 1
        for r in rooms:
            t = r["type"]
            samples.setdefault(t, []).append(r)

    print(f"[calibrate] Parsed {parsed}/{len(svg_files)} SVGs successfully")
    for t, rs in sorted(samples.items()):
        print(f"  {t:25}: {len(rs)} samples  avg {statistics.mean(r['area_ft2'] for r in rs):.0f} sqft")

    return samples


# ─── Stat computation ─────────────────────────────────────────────────────────

def compute_stats_from_samples(samples: list[dict]) -> dict:
    """Compute mean, p25, p75 for width/height from a list of room dicts."""
    ws = sorted(r["width_ft"] for r in samples)
    hs = sorted(r["height_ft"] for r in samples)
    n = len(ws)

    def _p(lst, pct):
        idx = max(0, min(n - 1, int(n * pct / 100)))
        return round(lst[idx])

    return {
        "mean_w": round(statistics.mean(ws)),
        "mean_h": round(statistics.mean(hs)),
        "p25_w":  _p(ws, 25),
        "p25_h":  _p(hs, 25),
        "p75_w":  _p(ws, 75),
        "p75_h":  _p(hs, 75),
        "n": n,
    }


# ─── Merged calibration ───────────────────────────────────────────────────────

def build_calibrated_tier_mins(
    cubicasa_samples: Optional[dict] = None,
    cubicasa_weight: float = 0.6,
) -> dict[str, list[tuple]]:
    """
    Build _TIER_MINS dict from NAHB stats, optionally blended with CubiCasa data.
    Returns {room_type: [(sqft_floor, min_w, min_h), ...]}
    """
    # Sqft thresholds corresponding to tiers (lower bound of each tier)
    tier_sqft = {"xs": 0, "s": 1000, "m": 1400, "l": 1800, "xl": 2400, "xxl": 3200}

    result: dict[str, list] = {}

    for room_type, tier_data in ROOM_STATS.items():
        rows = []
        for tier, lo, _ in SQFT_TIERS:
            if tier not in tier_data:
                continue
            nahb = tier_data[tier]
            w = nahb["mean_w"]
            h = nahb["mean_h"]

            if cubicasa_samples and room_type in cubicasa_samples:
                # Filter samples approximately matching this sqft tier
                # (CubiCasa doesn't have sqft labels, so we use room size as proxy)
                tier_area_range = {
                    "xs": (0, 80),   "s": (80, 140),  "m": (140, 200),
                    "l": (200, 280), "xl": (280, 400), "xxl": (400, 9999),
                }
                lo_a, hi_a = tier_area_range.get(tier, (0, 9999))
                tier_samples = [
                    s for s in cubicasa_samples[room_type]
                    if lo_a <= s["area_ft2"] < hi_a
                ]
                if len(tier_samples) >= 10:
                    cubi_stats = compute_stats_from_samples(tier_samples)
                    # Weighted blend: CubiCasa is Western but not US-specific
                    w = round(cubicasa_weight * cubi_stats["mean_w"] + (1 - cubicasa_weight) * w)
                    h = round(cubicasa_weight * cubi_stats["mean_h"] + (1 - cubicasa_weight) * h)

            # Use p25 as the minimum (floor guarantee), not the mean
            min_w = max(nahb["p25_w"], w - 2)
            min_h = max(nahb["p25_h"], h - 2)
            rows.append((lo, min_w, min_h))

        if rows:
            result[room_type] = rows

    return result


def build_calibrated_irc_specs(
    cubicasa_samples: Optional[dict] = None,
) -> dict[str, tuple]:
    """
    Build updated IRC_ROOM_SPECS from calibrated data.
    Format: (min_w, min_h, std_w, std_h, premium_w, premium_h)
    """
    from .data import IRC_ROOM_SPECS as current_specs

    result = dict(current_specs)

    for room_type, tier_data in ROOM_STATS.items():
        xs = tier_data.get("xs", {})
        m  = tier_data.get("m",  {})
        xxl = tier_data.get("xxl", {})

        if not (xs and m and xxl):
            continue

        min_w = xs.get("p25_w", current_specs.get(room_type, (10, 10))[0])
        min_h = xs.get("p25_h", current_specs.get(room_type, (10, 10))[1])
        std_w = m.get("mean_w",  current_specs.get(room_type, (10, 10, 14, 14))[2] if len(current_specs.get(room_type, ())) > 2 else 14)
        std_h = m.get("mean_h",  current_specs.get(room_type, (10, 10, 14, 14))[3] if len(current_specs.get(room_type, ())) > 3 else 14)
        prem_w = xxl.get("p75_w", std_w + 4)
        prem_h = xxl.get("p75_h", std_h + 4)

        result[room_type] = (min_w, min_h, std_w, std_h, prem_w, prem_h)

    return result


# ─── Code generation ──────────────────────────────────────────────────────────

def generate_tier_mins_code(tier_mins: dict[str, list]) -> str:
    lines = ["            _TIER_MINS: dict = {"]
    for rtype, rows in sorted(tier_mins.items()):
        row_strs = ", ".join(f"({lo}, {w}, {h})" for lo, w, h in rows)
        lines.append(f'                "{rtype}": [{row_strs}],')
    lines.append("            }")
    return "\n".join(lines)


def generate_irc_specs_code(specs: dict) -> str:
    lines = ["IRC_ROOM_SPECS = {"]
    for rtype, vals in sorted(specs.items()):
        lines.append(f'    "{rtype}": {vals},')
    lines.append("}")
    return "\n".join(lines)


# ─── Apply to source files ────────────────────────────────────────────────────

def apply_tier_mins(inference_path: Path, new_code: str) -> bool:
    """Replace _TIER_MINS dict in inference.py with calibrated values."""
    text = inference_path.read_text()
    # Find the existing _TIER_MINS block
    pattern = re.compile(
        r'(_TIER_MINS: dict = \{[^}]+\})',
        re.DOTALL,
    )
    new_block = new_code.strip()
    if pattern.search(text):
        updated = pattern.sub(new_block, text, count=1)
        inference_path.write_text(updated)
        return True
    return False


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_calibration(
    cubicasa_dir: Optional[Path] = None,
    write: bool = False,
    retrain: bool = False,
) -> dict:
    print("[calibrate] Starting calibration pipeline...")

    # Parse CubiCasa5K if available
    cubicasa_samples = None
    if cubicasa_dir and cubicasa_dir.exists():
        cubicasa_samples = parse_cubicasa_dataset(cubicasa_dir)
        print(f"[calibrate] CubiCasa5K: {sum(len(v) for v in cubicasa_samples.values())} room samples")
    else:
        print("[calibrate] CubiCasa5K not available — using NAHB/AHS data only")

    # Build calibrated constants
    tier_mins = build_calibrated_tier_mins(cubicasa_samples)
    irc_specs = build_calibrated_irc_specs(cubicasa_samples)

    print("\n[calibrate] Calibrated _TIER_MINS:")
    for rtype, rows in sorted(tier_mins.items()):
        print(f"  {rtype:25}: {rows}")

    if write:
        base = Path(__file__).parent
        inference_path = base / "inference.py"
        data_path      = base / "data.py"

        tier_code = generate_tier_mins_code(tier_mins)
        if apply_tier_mins(inference_path, tier_code):
            print(f"\n[calibrate] Updated _TIER_MINS in {inference_path}")
        else:
            print(f"\n[calibrate] WARNING: Could not find _TIER_MINS pattern in {inference_path}")
            print("Generated code:\n", tier_code)

        # Save calibration report
        report = {
            "tier_mins": {k: v for k, v in tier_mins.items()},
            "irc_specs": {k: list(v) for k, v in irc_specs.items()},
            "sources": ["NAHB 2023", "AHS 2021"]
            + (["CubiCasa5K"] if cubicasa_samples else []),
        }
        report_path = base / "weights" / "calibration_report.json"
        report_path.parent.mkdir(exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2))
        print(f"[calibrate] Report saved to {report_path}")

    if retrain:
        print("\n[calibrate] Starting model retraining with calibrated data...")
        from .train import MOETrainer
        from .config import MOEConfig
        config = MOEConfig()
        config.num_train_samples = 150_000
        config.num_val_samples   = 15_000
        config.num_test_samples  = 15_000
        config.epochs            = 120
        config.learning_rate     = 3e-4
        trainer = MOETrainer(config)
        metrics = trainer.train()
        print(f"[calibrate] Retraining complete. Test loss: {metrics['total']:.4f}")

    return {"tier_mins": tier_mins, "irc_specs": irc_specs}


def main():
    parser = argparse.ArgumentParser(description="Calibrate Buildify MOE from real housing data")
    parser.add_argument("--cubicasa", type=Path, default=None,
                        help="Path to CubiCasa5K data directory containing model.svg files")
    parser.add_argument("--write", action="store_true",
                        help="Write calibrated constants to inference.py and data.py")
    parser.add_argument("--retrain", action="store_true",
                        help="Retrain model after calibrating constants")
    args = parser.parse_args()

    run_calibration(
        cubicasa_dir=args.cubicasa,
        write=args.write,
        retrain=args.retrain,
    )


if __name__ == "__main__":
    main()
