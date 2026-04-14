"""
Architectural layout engine v2 — row-band placement, no overlaps guaranteed.

Floor plan uses horizontal bands stacked front-to-back (y increases southward):
  Band 0: Garage / Porch (y=0)
  Band 1: Foyer + Public rooms (living, dining, office) — side by side
  Band 2: Kitchen + Family room / semi-public
  Band 3: Hallway spine (4ft)
  Band 4: Primary suite + Secondary bedrooms — side by side
  Band 5: Primary bath + Shared baths + Laundry
  Band 6: Patio / Deck (optional)

All x/y/width/height are snapped to 2ft grid. No overlaps by construction.

Usage: python3 layout_engine.py [--count N] [--start I]
"""
import json, math, argparse
from pathlib import Path
from dataclasses import dataclass, field

HERE     = Path(__file__).parent
DATA_DIR = HERE / 'data'
DATA_DIR.mkdir(exist_ok=True)

GARAGE_W = {'none': 0, '1car': 12, '2car': 22, '3car': 32}
GARAGE_D = 22

STYLE_REASONING = {
    'craftsman':   ("Craftsman layout: wide covered porch anchors the entry. Open living-"
                    "dining-kitchen flows along the south face for natural light. "
                    "Bedroom wing retreats off the central hallway. "
                    "Mudroom at side entry creates a practical service transition."),
    'farmhouse':   ("Farmhouse layout: the kitchen is the heart, central and large. "
                    "Full-width front porch frames arrival. Primary suite on main floor. "
                    "Laundry off mudroom at rear; rear porch extends outdoor living."),
    'modern':      ("Modern layout: open plan — living, dining, kitchen merge into one "
                    "flowing space along the south facade. Primary suite isolated at one end. "
                    "Minimal corridors; sliding doors create indoor-outdoor connection."),
    'traditional': ("Traditional layout: formal foyer as distinct arrival room. "
                    "Formal living and dining flank the entry axis symmetrically. "
                    "Family room and kitchen at the rear. Primary suite strictly separated."),
    'ranch':       ("Ranch split-bedroom layout: primary suite one side, secondary bedrooms "
                    "the other. Living room at the front, kitchen and family room central. "
                    "Covered rear patio extends the living space outdoors."),
}


@dataclass
class Room:
    name: str; type: str
    x: float; y: float; width: float; height: float
    floor: int = 1
    adjacencies: list = field(default_factory=list)
    notes: str = ''

    def area(self): return self.width * self.height
    def x2(self): return self.x + self.width
    def y2(self): return self.y + self.height
    def to_dict(self):
        return {'name': self.name, 'type': self.type,
                'x': int(self.x), 'y': int(self.y),
                'width': int(self.width), 'height': int(self.height),
                'floor': self.floor, 'adjacencies': self.adjacencies,
                'notes': self.notes}

def s2(v): return max(2, round(v / 2) * 2)  # snap to 2ft grid, min 2


def generate_single_story(c: dict) -> list[Room]:
    sqft      = c['sqft']
    style     = c.get('style', 'modern')
    garage    = c.get('garage', '2car')
    bedrooms  = int(c['bedrooms'])
    bathrooms = int(c['bathrooms'])
    has_off   = c.get('homeOffice', False)
    has_din   = c.get('formalDining', False)
    has_lau   = c.get('laundry') == 'room'
    outdoor   = c.get('outdoor', 'none')
    has_porch = style in ('craftsman', 'farmhouse', 'ranch') or outdoor == 'porch'

    gw = GARAGE_W.get(garage, 0)

    # ── Footprint width ───────────────────────────────────────────────────────
    # Estimate total footprint area = sqft + garage
    total_area = sqft + (gw * GARAGE_D if gw > 0 else 0)
    FW = s2(math.sqrt(total_area * 1.35))   # ~1.35 aspect ratio
    FW = max(FW, gw + 28 if gw > 0 else 32)
    FW = min(FW, 80)

    rooms = []
    cur_y = 0

    # ── Band 0: Garage + Porch ────────────────────────────────────────────────
    if gw > 0:
        rooms.append(Room('Garage', 'garage', x=0, y=cur_y, width=gw, height=GARAGE_D,
                          notes='Attached garage, driveway approach from street.'))
    house_x = gw
    house_w = FW - gw

    if has_porch:
        pd = 8
        rooms.append(Room('Porch', 'porch', x=house_x, y=cur_y, width=house_w, height=pd,
                          notes=f'{style.title()} covered porch, full facade width.'))
        cur_y += pd

    if gw > 0:
        mud_w = s2(house_w * 0.28); mud_w = max(mud_w, 8); mud_w = min(mud_w, 12)
        mud_d = s2(GARAGE_D * 0.36); mud_d = max(mud_d, 8)
        rooms.append(Room('Mudroom', 'mudroom', x=house_x, y=cur_y, width=mud_w, height=mud_d,
                          adjacencies=['Garage', 'Kitchen'],
                          notes='Transition space: coat hooks, bench, direct kitchen access.'))
        pub_x = house_x + mud_w
        pub_w = house_w - mud_w
    else:
        pub_x  = house_x
        pub_w  = house_w
        mud_d  = 0

    # ── Band 1: Foyer + Public rooms ─────────────────────────────────────────
    pub_band_h = 16  # living room depth
    foyer_w    = s2(min(12, pub_w * 0.25)); foyer_w = max(foyer_w, 8)
    foyer_h    = 8

    rooms.append(Room('Foyer', 'foyer', x=pub_x, y=cur_y, width=foyer_w, height=foyer_h,
                      adjacencies=['Living Room', 'Hallway'],
                      notes='Distinct entry space; compressed scale before main living.'))

    # Living room fills rest of band 1, spanning full height
    pub_types = [('living_room', 'Living Room')]
    if has_din:  pub_types.append(('dining_room', 'Dining Room'))
    if has_off:  pub_types.append(('home_office', 'Home Office'))

    n_pub    = len(pub_types)
    remain_w = pub_w - foyer_w
    cell_w   = s2(remain_w / n_pub)
    cell_w   = max(cell_w, 12)

    lx = pub_x + foyer_w
    for pt, pname in pub_types:
        w = min(cell_w, remain_w - sum(cell_w for _ in pub_types[:pub_types.index((pt, pname))]))
        w = s2(w); w = max(w, 12)
        rooms.append(Room(pname, pt, x=lx, y=cur_y, width=w, height=pub_band_h,
                          adjacencies=['Foyer', 'Kitchen'],
                          notes='South-facing for natural light.'))
        lx += w

    # Foyer column above foyer_h (remainder of band 1 height)
    if pub_band_h > foyer_h:
        rooms.append(Room('Entry Hall', 'hallway',
                          x=pub_x, y=cur_y + foyer_h,
                          width=foyer_w, height=pub_band_h - foyer_h,
                          adjacencies=['Foyer', 'Kitchen'],
                          notes='Entry corridor connecting foyer to kitchen zone.'))

    cur_y += pub_band_h

    # ── Band 2: Kitchen + semi-public ────────────────────────────────────────
    kit_band_h = 14
    kit_w = s2(house_w * 0.45); kit_w = max(kit_w, 12); kit_w = min(kit_w, 24)
    fam_w = house_w - kit_w

    rooms.append(Room('Kitchen', 'kitchen', x=house_x, y=cur_y, width=kit_w, height=kit_band_h,
                      adjacencies=['Living Room', 'Family Room'],
                      notes='Open to family room; island provides workspace and casual dining.'))

    if fam_w >= 12:
        rooms.append(Room('Family Room', 'family_room',
                          x=house_x + kit_w, y=cur_y, width=fam_w, height=kit_band_h,
                          adjacencies=['Kitchen', 'Hallway'],
                          notes='Everyday living space; direct rear yard access.'))

    if has_lau:
        # Laundry replaces part of kitchen band or appends
        lau_w = 8; lau_d = 8
        # Only add if kitchen band has space (place beside kitchen at same y)
        rooms.append(Room('Laundry', 'laundry_room',
                          x=house_x + kit_w + fam_w, y=cur_y, width=lau_w, height=lau_d,
                          adjacencies=['Kitchen', 'Hallway'],
                          notes='Laundry room with rear service access.'))

    cur_y += kit_band_h

    # ── Band 3: Hallway spine ─────────────────────────────────────────────────
    hall_h = 4
    rooms.append(Room('Hallway', 'hallway', x=house_x, y=cur_y, width=house_w, height=hall_h,
                      adjacencies=['Primary Suite', 'Bedroom 2', 'Bathroom'],
                      notes='Hallway spine separating public and private zones.'))
    cur_y += hall_h

    # ── Band 4: Bedrooms ──────────────────────────────────────────────────────
    bed_band_h = 14
    secondary  = bedrooms - 1  # secondary bedrooms (non-primary)

    # Primary suite: left side (west, quieter)
    ps_frac = 0.45 if secondary == 0 else max(0.35, 0.5 - secondary * 0.04)
    ps_w    = s2(house_w * ps_frac); ps_w = max(ps_w, 14)

    rooms.append(Room('Primary Suite', 'primary_suite',
                      x=house_x, y=cur_y, width=ps_w, height=bed_band_h,
                      adjacencies=['Primary Bathroom', 'Hallway'],
                      notes='Northwest corner for quiet; private retreat.'))

    remain_bw = house_w - ps_w
    if secondary > 0 and remain_bw >= 10:
        bw = s2(remain_bw / secondary); bw = max(bw, 10)
        bx = house_x + ps_w
        for i in range(secondary):
            actual_w = s2(min(bw, remain_bw - i * bw)); actual_w = max(actual_w, 10)
            rooms.append(Room(f'Bedroom {i+2}', 'bedroom',
                              x=bx, y=cur_y, width=actual_w, height=bed_band_h,
                              adjacencies=['Hallway', 'Bathroom'],
                              notes='East-facing; morning light.'))
            bx += actual_w

    cur_y += bed_band_h

    # ── Band 5: Bathrooms ─────────────────────────────────────────────────────
    bath_band_h = 8
    # Primary bathroom
    rooms.append(Room('Primary Bathroom', 'primary_bathroom',
                      x=house_x, y=cur_y, width=ps_w, height=bath_band_h,
                      adjacencies=['Primary Suite'],
                      notes='En-suite with double vanity, walk-in shower, soaking tub.'))

    shared_baths = max(0, bathrooms - 1)
    bx = house_x + ps_w
    bw_per = s2(remain_bw / max(shared_baths, 1)) if remain_bw > 0 else 0
    bw_per = max(bw_per, 6)
    for i in range(shared_baths):
        rooms.append(Room(f'Bathroom {i+2}', 'bathroom',
                          x=bx, y=cur_y, width=min(bw_per, remain_bw - i * bw_per),
                          height=bath_band_h,
                          adjacencies=['Hallway'],
                          notes='Shared hall bath.'))
        bx += bw_per

    cur_y += bath_band_h

    # ── Band 6: Outdoor ───────────────────────────────────────────────────────
    if outdoor in ('patio', 'deck'):
        rooms.append(Room(outdoor.title(), 'patio',
                          x=house_x, y=cur_y, width=house_w, height=12,
                          notes=f'Rear {outdoor}; direct access from family room.'))

    return rooms


def compute_totals(rooms):
    exclude = {'garage', 'porch', 'patio'}
    cond = sum(r.area() for r in rooms if r.type not in exclude and r.floor == 1)
    fw   = max((r.x2() for r in rooms if r.floor == 1), default=0)
    fd   = max((r.y2() for r in rooms if r.floor == 1), default=0)
    return round(cond), round(fw), round(fd)


def build_zones(rooms):
    mapping = {
        'living_room': 'public', 'foyer': 'public', 'home_office': 'public', 'dining_room': 'public',
        'kitchen': 'semi_public', 'family_room': 'semi_public',
        'primary_suite': 'private', 'primary_bathroom': 'private',
        'bedroom': 'private', 'bathroom': 'private', 'ensuite_bathroom': 'private',
        'garage': 'service', 'laundry_room': 'service', 'mudroom': 'service',
    }
    z = {'public': [], 'semi_public': [], 'private': [], 'service': []}
    for r in rooms:
        bucket = mapping.get(r.type)
        if bucket: z[bucket].append(r.name)
    return z


def generate_example(c: dict) -> dict:
    style = c.get('style', 'modern')
    rooms = generate_single_story(c)  # 2-story: stack floor 2 above floor 1
    if c.get('stories', 1) == 2:
        # Move private zones to floor 2 — but only the bedroom-wing hallway, not entry hall
        private_types = {'primary_suite','primary_bathroom','bedroom','bathroom','ensuite_bathroom'}
        # Find y-threshold: bedroom band starts after kitchen band
        kitchen_y2 = max((r.y2() for r in rooms if r.type in ('kitchen','family_room')), default=0)
        for r in rooms:
            if r.type in private_types or (r.type == 'hallway' and r.y >= kitchen_y2):
                r.floor = 2

    cond, fw, fd = compute_totals(rooms)
    garage = c.get('garage', 'none')
    entry  = (['Driveway', 'Garage', 'Mudroom', 'Foyer'] if garage != 'none'
               else ['Porch', 'Foyer'] if any(r.type == 'porch' for r in rooms)
               else ['Foyer'])
    return {
        'reasoning': STYLE_REASONING.get(style, STYLE_REASONING['modern']),
        'entry_sequence': entry,
        'zones': build_zones(rooms),
        'circulation_notes': 'Central hallway spine distributes traffic between public and private zones without cross-contamination.',
        'solar_notes': 'Living areas south-facing; bedrooms northeast for morning light; service rooms north.',
        'rooms': [r.to_dict() for r in rooms],
        'total_conditioned_sqft': cond,
        'footprint_width': fw,
        'footprint_depth': fd,
        'massing_notes': f'Compact rectangular {style} massing; clear zone separation from street to rear.',
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--count', type=int, default=1000)
    parser.add_argument('--start', type=int, default=0)
    args = parser.parse_args()

    lines = (HERE / 'constraints_all.jsonl').read_text().splitlines()
    batch = [json.loads(l) for l in lines[args.start: args.start + args.count]]

    tag = f'batch_engine_{args.start:06d}'
    out = DATA_DIR / f'{tag}.jsonl'
    written = errors = 0
    with open(out, 'w') as f:
        for c in batch:
            try:
                layout = generate_example(c)
                f.write(json.dumps({'input': c, 'output': layout, 'source': 'engine_v2'}) + '\n')
                written += 1
            except Exception as e:
                errors += 1
    print(f'{written} examples ({errors} errors) → {out}')


if __name__ == '__main__':
    main()
