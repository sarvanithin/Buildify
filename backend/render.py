"""
render.py — Photorealistic room renders via fal.ai Flux Schnell.
Fast (~3s), high quality, perfect for architectural interior visualization.
"""

import hashlib
import json
import os
from pathlib import Path

import httpx

RENDER_CACHE_FILE = Path("render_cache.json")

# ─── Style descriptors ────────────────────────────────────────────────────────

STYLE_DESCRIPTORS: dict[str, str] = {
    "modern":       "minimalist modern, clean lines, neutral whites and grays, open concept, recessed lighting",
    "contemporary": "contemporary design, mixed textures, bold accent wall, statement lighting, current trends",
    "craftsman":    "craftsman style, exposed wood beams, warm oak tones, built-in shelving, wainscoting",
    "farmhouse":    "modern farmhouse, shiplap accent wall, warm whites, barn wood accents, cozy atmosphere",
    "ranch":        "ranch style, warm earth tones, rustic wood flooring, low horizontal profile, casual comfort",
    "traditional":  "traditional style, crown molding, rich warm colors, classic millwork, formal proportions",
}

# ─── Per-room prompt templates ────────────────────────────────────────────────

ROOM_PROMPTS: dict[str, str] = {
    "living_room":        "photorealistic interior photograph of a {style} living room, {w}ft wide by {d}ft deep, {ch}ft ceiling height, large windows, natural afternoon light, hardwood floors, plush sofa, coffee table",
    "great_room":         "photorealistic interior of a {style} great room, {w}ft × {d}ft, {ch}ft ceilings, open concept with kitchen view, fireplace, cathedral ceiling, abundant natural light",
    "kitchen":            "photorealistic {style} kitchen interior, {w}ft × {d}ft, {ch}ft ceilings, quartz countertops, modern cabinetry, stainless steel appliances, under-cabinet lighting, subway tile backsplash",
    "dining_room":        "photorealistic {style} dining room, {w}ft × {d}ft, {ch}ft ceiling, statement chandelier, large dining table for 8, hardwood floors, large windows",
    "master_bedroom":     "photorealistic luxury {style} primary bedroom suite, {w}ft × {d}ft, {ch}ft ceilings, king bed with upholstered headboard, bedside tables, walk-in closet doors visible, natural morning light",
    "primary_bedroom":    "photorealistic luxury {style} primary bedroom suite, {w}ft × {d}ft, {ch}ft ceilings, king bed, soft natural light, high-end finishes",
    "bedroom":            "photorealistic {style} bedroom, {w}ft × {d}ft, {ch}ft ceilings, queen bed, natural light through windows, clean comfortable design",
    "bathroom":           "photorealistic {style} bathroom, {w}ft × {d}ft, {ch}ft ceilings, tile floor, modern vanity, frameless glass shower, natural light",
    "ensuite_bathroom":   "photorealistic luxury {style} ensuite bathroom, {w}ft × {d}ft, {ch}ft ceilings, marble tile, freestanding soaking tub, double vanity, spa-like atmosphere, warm lighting",
    "half_bath":          "photorealistic {style} powder room, {w}ft × {d}ft, statement wallpaper, vessel sink, designer fixtures, sophisticated small space",
    "home_office":        "photorealistic {style} home office, {w}ft × {d}ft, {ch}ft ceilings, built-in bookshelves, large desk, natural light, productive professional atmosphere",
    "foyer":              "photorealistic {style} entry foyer, {w}ft × {d}ft, {ch}ft ceilings, statement light fixture, tile or hardwood floor, elegant first impression, architectural photography",
    "hallway":            "photorealistic {style} interior hallway, {w}ft wide, {ch}ft ceilings, recessed lighting, hardwood floors, art on walls",
    "laundry_room":       "photorealistic {style} laundry room, {w}ft × {d}ft, built-in cabinets, front-load washer and dryer, tile floor, organized clean space",
    "mudroom":            "photorealistic {style} mudroom, {w}ft × {d}ft, built-in bench with cubbies, hooks for coats, tile floor, organized drop zone",
    "garage":             "photorealistic {style} garage interior, {w}ft × {d}ft, epoxy floor, organized wall storage, clean modern look",
    "family_room":        "photorealistic {style} family room, {w}ft × {d}ft, {ch}ft ceilings, comfortable sectional, fireplace, large TV wall, cozy atmosphere",
    "dining_area":        "photorealistic {style} dining area, {w}ft × {d}ft, modern dining set, pendant light, adjacent to kitchen",
    "pantry":             "photorealistic {style} walk-in pantry, {w}ft × {d}ft, floor-to-ceiling shelving, organized dry goods, under-shelf lighting",
    "walk_in_closet":     "photorealistic {style} walk-in closet, {w}ft × {d}ft, custom built-in system, center island with drawers, LED lighting, luxury dressing room",
}

FALLBACK_PROMPT = "photorealistic {style} interior room, {w}ft × {d}ft, {ch}ft ceilings, professional architectural photography, natural light, high-end finishes"

# ─── Prompt builder ───────────────────────────────────────────────────────────

def build_prompt(room_type: str, style: str, width_ft: float, depth_ft: float, ceiling_h: int) -> str:
    style_lower = style.lower()
    style_desc = STYLE_DESCRIPTORS.get(style_lower, STYLE_DESCRIPTORS["modern"])

    template = ROOM_PROMPTS.get(room_type.lower(), FALLBACK_PROMPT)
    base = template.format(
        style=style_lower,
        w=int(width_ft),
        d=int(depth_ft),
        ch=ceiling_h,
    )

    prompt = (
        f"{base}, {style_desc}, "
        "interior design photography, photorealistic, "
        "8K resolution, shot with wide-angle lens, "
        "professional real estate photography, no people, no text"
    )
    return prompt

# ─── Cache helpers ────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if RENDER_CACHE_FILE.exists():
        try:
            return json.loads(RENDER_CACHE_FILE.read_text())
        except Exception:
            return {}
    return {}

def _save_cache(cache: dict) -> None:
    try:
        RENDER_CACHE_FILE.write_text(json.dumps(cache, indent=2))
    except Exception:
        pass

def _cache_key(room_type: str, style: str, w: float, d: float, ch: int) -> str:
    raw = f"{room_type}|{style.lower()}|{int(w)}|{int(d)}|{ch}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]

# ─── Pollinations.ai (free, no key) ──────────────────────────────────────────

def _pollinations_url(prompt: str, seed: int) -> str:
    """Build a Pollinations.ai FLUX image URL — completely free, no API key."""
    import urllib.parse
    encoded = urllib.parse.quote(prompt)
    return (
        f"https://image.pollinations.ai/prompt/{encoded}"
        f"?width=1280&height=960&model=flux&nologo=true&seed={seed}"
    )

# ─── fal.ai (paid, faster) ────────────────────────────────────────────────────

async def _fal_render(prompt: str, fal_key: str) -> str:
    """Call fal.ai Flux Schnell. Returns image URL."""
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.post(
            "https://fal.run/fal-ai/flux/schnell",
            headers={"Authorization": f"Key {fal_key}", "Content-Type": "application/json"},
            json={"prompt": prompt, "image_size": "landscape_4_3",
                  "num_inference_steps": 4, "num_images": 1},
        )
        resp.raise_for_status()
        return resp.json()["images"][0]["url"]

# ─── Main render function ─────────────────────────────────────────────────────

async def generate_render(
    room_type: str,
    style: str,
    width_ft: float,
    depth_ft: float,
    ceiling_h: int,
) -> dict:
    """
    Generate a photorealistic render.
    Uses fal.ai if FAL_KEY is set (fast, ~3s), otherwise Pollinations.ai (free, ~10s).
    Returns {"image_url": str, "cached": bool, "prompt": str}
    """
    key = _cache_key(room_type, style, width_ft, depth_ft, ceiling_h)
    cache = _load_cache()

    if key in cache:
        return {"image_url": cache[key], "cached": True, "prompt": ""}

    prompt = build_prompt(room_type, style, width_ft, depth_ft, ceiling_h)
    fal_key = os.getenv("FAL_KEY", "").strip()

    print(f"[render] {'fal.ai' if fal_key else 'pollinations'}: {room_type}/{style} — {prompt[:70]}…")

    if fal_key:
        # Paid path — fal.ai Flux Schnell (~3s)
        image_url = await _fal_render(prompt, fal_key)
    else:
        # Free path — Pollinations.ai FLUX (no key needed, ~10-15s)
        seed = int(key, 16) % 2147483647
        image_url = _pollinations_url(prompt, seed)
        # Pollinations returns the image at this URL — warm it up async so
        # the browser gets it fast when the frontend sets it as <img src>
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                await client.get(image_url)  # pre-fetch to warm CDN
        except Exception:
            pass  # even if warm-up fails, the URL still works in the browser

    if fal_key:
        # Only cache fal.ai URLs (pollinations URLs are deterministic by seed)
        cache[key] = image_url
        _save_cache(cache)

    return {"image_url": image_url, "cached": False, "prompt": prompt}
