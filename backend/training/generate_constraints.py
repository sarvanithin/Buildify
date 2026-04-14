"""
Generates diverse constraint combinations for training data generation.
Produces a JSONL file of constraint dicts covering the full realistic space.
"""
import json
import random
import itertools
from pathlib import Path

random.seed(42)

STYLES    = ['modern', 'craftsman', 'farmhouse', 'traditional', 'ranch']
GARAGES   = ['none', '1car', '2car', '3car']
LAUNDRY   = ['room', 'closet', 'none']
OUTDOOR   = ['none', 'patio', 'deck', 'porch']
CEILING   = [8, 9, 10]
STORIES   = [1, 2]

# Sqft bands that make sense per bedroom count
SQFT_RANGES = {
    1: (600,  1000),
    2: (800,  1400),
    3: (1100, 2200),
    4: (1500, 3200),
    5: (2000, 4200),
    6: (2800, 5500),
}

def feasible(c: dict) -> bool:
    """Quick feasibility filter — mirrors backend validation."""
    sqft      = c['sqft']
    bedrooms  = c['bedrooms']
    bathrooms = c['bathrooms']

    # Base conditioned minimum
    base  = 168 + 120 + 200 + 40   # living + kitchen + hallway + foyer
    base += 240 if c.get('primarySuite', True) else 168
    secondary = max(0, bedrooms - 1)
    base += secondary * 100
    base += max(0, bathrooms - 1) * 40
    if c.get('homeOffice'):   base += 90
    if c.get('formalDining'): base += 121
    if c.get('laundry') == 'room': base += 30

    if sqft < base:
        return False
    if bathrooms > bedrooms + 2:
        return False
    if c['stories'] == 2 and sqft < 1100:
        return False
    return True


def generate_all(n_target: int = 60_000) -> list[dict]:
    constraints = []
    seen = set()

    # --- Grid pass: cover every key combination at least once ---
    for beds, style, garage, stories in itertools.product(
        range(1, 7), STYLES, GARAGES, STORIES
    ):
        sqft_lo, sqft_hi = SQFT_RANGES[beds]
        for sqft_step in range(sqft_lo, sqft_hi + 1, 200):
            baths = max(1, min(beds, round(beds * 0.7)))
            c = {
                'sqft':        sqft_step,
                'bedrooms':    beds,
                'bathrooms':   baths,
                'stories':     stories,
                'style':       style,
                'garage':      garage,
                'laundry':     'room',
                'outdoor':     'patio' if stories == 1 else 'deck',
                'ceilingHeight': 9,
                'primarySuite':  True,
                'homeOffice':    False,
                'formalDining':  False,
                'masterBath':    True,
                'walkInCloset':  beds >= 3,
            }
            if feasible(c):
                key = json.dumps(c, sort_keys=True)
                if key not in seen:
                    seen.add(key)
                    constraints.append(c)

    # --- Random pass: fill up to n_target with varied optional rooms ---
    rng = random.Random(0)
    attempts = 0
    while len(constraints) < n_target and attempts < n_target * 10:
        attempts += 1
        beds   = rng.choices([2, 3, 3, 4, 4, 5, 6, 1], k=1)[0]
        sqft_lo, sqft_hi = SQFT_RANGES[beds]
        sqft   = rng.randrange(sqft_lo, sqft_hi + 1, 100)
        baths  = rng.randint(1, min(beds + 1, 5))
        stories = rng.choices([1, 1, 2], k=1)[0]
        style  = rng.choice(STYLES)
        garage = rng.choice(GARAGES)

        c = {
            'sqft':          sqft,
            'bedrooms':      beds,
            'bathrooms':     baths,
            'stories':       stories,
            'style':         style,
            'garage':        garage,
            'laundry':       rng.choice(LAUNDRY),
            'outdoor':       rng.choice(OUTDOOR),
            'ceilingHeight': rng.choice(CEILING),
            'primarySuite':  True,
            'homeOffice':    rng.random() < 0.30,
            'formalDining':  rng.random() < 0.25 and sqft > 1600,
            'masterBath':    True,
            'walkInCloset':  beds >= 3 and rng.random() < 0.70,
        }
        if not feasible(c):
            continue
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            constraints.append(c)

    rng.shuffle(constraints)
    return constraints


if __name__ == '__main__':
    out = Path(__file__).parent / 'constraints_all.jsonl'
    constraints = generate_all(60_000)
    with open(out, 'w') as f:
        for c in constraints:
            f.write(json.dumps(c) + '\n')
    print(f"Generated {len(constraints):,} constraint sets → {out}")
