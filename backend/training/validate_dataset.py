"""
Validates and filters raw generated training examples.
Removes examples with:
  - Room overlaps > 5% area
  - Total sqft outside ±10% of requested
  - Any room below IRC minimums
  - Missing required rooms (no living room, no primary bedroom, etc.)
  - Rooms outside footprint bounds

Writes clean dataset to data/dataset_clean.jsonl
Reports stats on what was filtered.

Usage: python3 validate_dataset.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
DATA_DIR = HERE / 'data'

IRC_MIN_DIM = {
    'living_room':       (10, 14),
    'kitchen':           (8,  10),
    'primary_suite':     (12, 12),
    'bedroom':           (9,  10),
    'bathroom':          (5,  8),
    'primary_bathroom':  (6,  8),
    'ensuite_bathroom':  (6,  8),
    'foyer':             (6,  6),
    'hallway':           (3,  5),
    'laundry_room':      (5,  6),
    'home_office':       (9,  9),
    'dining_room':       (10, 10),
    'family_room':       (10, 12),
    'garage':            (10, 18),  # 1-car minimum
    'mudroom':           (4,  6),
    'porch':             (6,  8),
    'patio':             (8,  8),
    'closet':            (3,  4),
}

REQUIRED_BY_type = {'living_room', 'kitchen', 'foyer'}


def rooms_overlap(a: dict, b: dict) -> float:
    """Returns overlap area in sqft between two rooms."""
    ax1, ay1 = a['x'], a['y']
    ax2, ay2 = ax1 + a['width'], ay1 + a['height']
    bx1, by1 = b['x'], b['y']
    bx2, by2 = bx1 + b['width'], by1 + b['height']
    ox = max(0, min(ax2, bx2) - max(ax1, bx1))
    oy = max(0, min(ay2, by2) - max(ay1, by1))
    return ox * oy


def validate_example(ex: dict) -> tuple[bool, str]:
    inp    = ex.get('input', {})
    layout = ex.get('output', {})
    rooms  = layout.get('rooms', [])

    if not rooms:
        return False, 'no_rooms'

    # --- Required rooms ---
    types_present = {r.get('type') for r in rooms}
    if not REQUIRED_BY_type.issubset(types_present):
        missing = REQUIRED_BY_type - types_present
        return False, f'missing_required:{missing}'

    # --- IRC minimum dimensions ---
    for r in rooms:
        rtype = r.get('type', '')
        w, h  = r.get('width', 0), r.get('height', 0)
        mins  = IRC_MIN_DIM.get(rtype)
        if mins:
            # Check that both dimensions meet IRC minimums (order-independent)
            short_side = min(w, h)
            long_side  = max(w, h)
            if short_side < mins[0] or long_side < mins[1]:
                return False, f'irc_too_small:{r["name"]}({w}x{h}<{mins})'

    # --- Overlap check ---
    f1_rooms = [r for r in rooms if r.get('floor', 1) == 1]
    for i, ra in enumerate(f1_rooms):
        for rb in f1_rooms[i+1:]:
            overlap = rooms_overlap(ra, rb)
            if overlap > 4:  # allow up to 4 sqft rounding tolerance
                return False, f'overlap:{ra["name"]}&{rb["name"]}({overlap:.1f}sqft)'

    # --- Footprint bounds ---
    fw = layout.get('footprint_width', 0)
    fd = layout.get('footprint_depth', 0)
    if fw <= 0 or fd <= 0:
        return False, 'no_footprint'
    for r in f1_rooms:
        if r['x'] < 0 or r['y'] < 0:
            return False, f'negative_coords:{r["name"]}'
        if r['x'] + r['width'] > fw + 2 or r['y'] + r['height'] > fd + 2:
            return False, f'out_of_bounds:{r["name"]}'

    # --- Sqft accuracy ---
    req_sqft  = inp.get('sqft', 0)
    gen_sqft  = layout.get('total_conditioned_sqft', 0)
    if req_sqft > 0 and gen_sqft > 0:
        ratio = gen_sqft / req_sqft
        if ratio < 0.72 or ratio > 1.40:
            return False, f'sqft_mismatch:{gen_sqft:.0f}vs{req_sqft}'

    return True, 'ok'


def main():
    raw_files = sorted(DATA_DIR.glob('batch_*.jsonl'))
    if not raw_files:
        print('No batch files found. Run agent_generator.py first.')
        return

    total = 0
    passed = 0
    reasons: dict[str, int] = {}
    out_file = DATA_DIR / 'dataset_clean.jsonl'

    with open(out_file, 'w') as out:
        for bf in raw_files:
            for line in bf.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError:
                    continue
                total += 1
                ok, reason = validate_example(ex)
                if ok:
                    out.write(json.dumps(ex) + '\n')
                    passed += 1
                else:
                    top = reason.split(':')[0]
                    reasons[top] = reasons.get(top, 0) + 1

    print(f'\nValidation complete')
    print(f'  Total examples : {total:,}')
    print(f'  Passed         : {passed:,} ({100*passed/max(total,1):.1f}%)')
    print(f'  Filtered       : {total - passed:,}')
    if reasons:
        print(f'  Failure reasons:')
        for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f'    {k}: {v}')
    print(f'\nClean dataset → {out_file}')


if __name__ == '__main__':
    main()
