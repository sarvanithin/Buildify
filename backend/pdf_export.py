"""
pdf_export.py — Architectural PDF floor plan export using fpdf2.
Produces a letter-size (11×8.5 landscape) PDF with:
  - Title block (plan name, sqft, style, date)
  - Floor plan drawing (scaled rooms with hatch, labels, dimensions)
  - North arrow
  - Room schedule table
"""

from __future__ import annotations

import math
from datetime import date
from io import BytesIO
from typing import List

from fpdf import FPDF

# ─── Paper & margin constants ─────────────────────────────────────────────────

PAGE_W_MM = 279.4   # letter landscape width  (11")
PAGE_H_MM = 215.9   # letter landscape height (8.5")
MARGIN_MM  = 12.0   # document margin
TITLE_H_MM = 28.0   # title block height at bottom

# Drawing area
DRAW_X = MARGIN_MM
DRAW_Y = MARGIN_MM
DRAW_W = PAGE_W_MM - MARGIN_MM * 2 - 60  # leave 60mm right column for schedule
DRAW_H = PAGE_H_MM - MARGIN_MM * 2 - TITLE_H_MM

SCHED_X = DRAW_X + DRAW_W + 4
SCHED_W = PAGE_W_MM - SCHED_X - MARGIN_MM

# ─── Room fill colours (greyscale tints for PDF) ─────────────────────────────

ROOM_GREY: dict[str, tuple] = {
    "living_room":       (240, 244, 251),
    "great_room":        (238, 243, 250),
    "kitchen":           (245, 245, 238),
    "dining_room":       (242, 245, 238),
    "master_bedroom":    (245, 238, 245),
    "bedroom":           (245, 238, 245),
    "bathroom":          (238, 245, 245),
    "ensuite_bathroom":  (238, 245, 245),
    "hallway":           (248, 248, 244),
    "foyer":             (244, 244, 238),
    "home_office":       (238, 242, 248),
    "garage":            (240, 240, 236),
    "patio":             (238, 245, 238),
    "deck":              (238, 245, 238),
    "laundry_room":      (244, 240, 248),
    "mudroom":           (245, 240, 236),
    "closet":            (240, 236, 240),
    "walk_in_closet":    (232, 216, 232),
    "pantry":            (237, 232, 220),
}


def _room_fill(room_type: str) -> tuple:
    return ROOM_GREY.get(room_type.lower(), (245, 245, 245))


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _ft_label(feet: float) -> str:
    """Format feet as feet-inches string, e.g. 13.5 → 13'-6"."""
    whole = int(feet)
    inches = round((feet - whole) * 12)
    if inches == 12:
        whole += 1
        inches = 0
    return f"{whole}'-{inches}\"" if inches else f"{whole}'"


def _draw_north_arrow(pdf: FPDF, cx: float, cy: float, r: float = 4.0) -> None:
    """Draw a simple north arrow centred at (cx, cy) with radius r mm."""
    pdf.set_draw_color(30, 30, 30)
    pdf.set_fill_color(30, 30, 30)
    # Circle
    pdf.ellipse(cx - r, cy - r, r * 2, r * 2, style="D")
    # Arrow shaft (north = up in plan = decreasing Y on page)
    pdf.set_line_width(0.5)
    pdf.line(cx, cy + r * 0.7, cx, cy - r * 0.7)
    # Arrowhead
    pdf.line(cx, cy - r * 0.7, cx - r * 0.4, cy - r * 0.15)
    pdf.line(cx, cy - r * 0.7, cx + r * 0.4, cy - r * 0.15)
    # "N" label
    pdf.set_font("Helvetica", "B", 5)
    pdf.set_xy(cx - 1.2, cy - r - 4)
    pdf.cell(2.4, 3, "N", align="C")


def _draw_scale_bar(pdf: FPDF, x: float, y: float, S: float) -> None:
    """Draw a 10-ft scale bar. S = mm per foot."""
    bar_ft = 10
    bar_mm = bar_ft * S
    pdf.set_draw_color(30, 30, 30)
    pdf.set_fill_color(30, 30, 30)
    pdf.set_line_width(0.3)
    # Three segments: 0-5ft light, 5-10ft filled
    half = bar_mm / 2
    pdf.rect(x, y, half, 1.5, style="D")
    pdf.rect(x + half, y, half, 1.5, style="F")
    # Ticks
    for ft in (0, 5, 10):
        px = x + ft * S
        pdf.line(px, y, px, y + 2.5)
    pdf.set_font("Helvetica", "", 4.5)
    pdf.set_xy(x - 1, y + 3.0)
    pdf.cell(3, 2, "0", align="C")
    pdf.set_xy(x + half - 1, y + 3.0)
    pdf.cell(3, 2, "5'", align="C")
    pdf.set_xy(x + bar_mm - 2, y + 3.0)
    pdf.cell(4, 2, "10'", align="C")
    pdf.set_xy(x, y + 5.5)
    pdf.cell(bar_mm, 2, "SCALE 1/16\" = 1'-0\"", align="C")


# ─── Plan drawing ─────────────────────────────────────────────────────────────

def _draw_plan(pdf: FPDF, plan: dict, x0: float, y0: float, w: float, h: float,
               floor: int = 1) -> float:
    """
    Draw floor plan rooms inside box (x0, y0, w, h) in mm.
    Returns the scale factor S (mm per foot) used.
    """
    rooms = [r for r in plan.get("rooms", [])
             if r.get("floor", 1) == floor]
    if not rooms:
        return 1.0

    plan_w = plan.get("totalWidth", 60)
    plan_h = plan.get("totalHeight", 40)
    if floor == 2:
        plan_h = plan.get("floor2Height", plan_h)

    S = min(w / plan_w, h / plan_h)
    # Centre the plan in the available space
    off_x = x0 + (w - plan_w * S) / 2
    off_y = y0 + (h - plan_h * S) / 2

    # ── Room fills ──
    pdf.set_line_width(0.15)
    for r in rooms:
        rx = off_x + r["x"] * S
        ry = off_y + r["y"] * S
        rw = r["width"] * S
        rh = r["height"] * S
        fill = _room_fill(r.get("type", ""))
        pdf.set_fill_color(*fill)
        pdf.set_draw_color(100, 100, 100)
        pdf.rect(rx, ry, rw, rh, style="FD")

    # ── Exterior boundary ──
    pdf.set_draw_color(20, 20, 20)
    pdf.set_line_width(0.6)
    pdf.rect(off_x, off_y, plan_w * S, plan_h * S, style="D")

    # ── Room labels ──
    for r in rooms:
        rx = off_x + r["x"] * S
        ry = off_y + r["y"] * S
        rw = r["width"] * S
        rh = r["height"] * S
        cx_r = rx + rw / 2
        cy_r = ry + rh / 2

        name = r.get("name", "")
        # Wrap long names
        words = name.upper().split()
        lines_: list[str] = []
        cur = ""
        for w_ in words:
            if len(cur) + len(w_) + 1 > 12 and cur:
                lines_.append(cur.strip())
                cur = w_
            else:
                cur = (cur + " " + w_).strip()
        if cur:
            lines_.append(cur)

        fs = max(3.0, min(5.5, rw * 0.55))
        pdf.set_font("Helvetica", "B", fs)
        pdf.set_text_color(40, 40, 40)
        line_h = fs * 0.45
        total_h = line_h * len(lines_)
        for li, line_text in enumerate(lines_):
            lx = cx_r - rw / 2 + 0.5
            ly = cy_r - total_h / 2 + li * line_h
            pdf.set_xy(lx, ly)
            pdf.cell(rw - 1, line_h, line_text, align="C")

        # Dimension label
        dim_text = f"{int(r['width'])}×{int(r['height'])}"
        fs2 = max(2.5, min(4.0, rw * 0.4))
        pdf.set_font("Helvetica", "", fs2)
        pdf.set_text_color(100, 100, 100)
        pdf.set_xy(cx_r - rw / 2 + 0.5, cy_r + total_h / 2 + 0.5)
        pdf.cell(rw - 1, fs2 * 0.4, dim_text, align="C")

    pdf.set_text_color(0, 0, 0)

    # ── Dimension strings: top + left ──
    pdf.set_font("Helvetica", "", 4)
    pdf.set_draw_color(80, 80, 80)
    pdf.set_line_width(0.2)

    total_w_label = _ft_label(plan_w)
    pdf.set_xy(off_x, off_y - 5)
    pdf.cell(plan_w * S, 3, total_w_label, align="C")
    pdf.line(off_x, off_y - 3, off_x + plan_w * S, off_y - 3)

    total_h_label = _ft_label(plan_h)
    # Rotated vertical — manual chars down the side
    pdf.set_xy(off_x - 8, off_y + plan_h * S / 2 - 3)
    pdf.cell(8, 3, total_h_label, align="C")

    # ── North arrow + scale bar ──
    _draw_north_arrow(pdf, off_x + plan_w * S + 6, off_y + 6, r=3)
    _draw_scale_bar(pdf, off_x + plan_w * S - 20 * S, off_y + plan_h * S + 3, S)

    return S


# ─── Room schedule ────────────────────────────────────────────────────────────

def _draw_schedule(pdf: FPDF, plan: dict, x0: float, y0: float, w: float, h: float) -> None:
    """Draw a room area schedule table in the right column."""
    rooms = plan.get("rooms", [])
    f1 = [r for r in rooms if r.get("floor", 1) == 1]
    f2 = [r for r in rooms if r.get("floor", 1) == 2]

    pdf.set_font("Helvetica", "B", 5.5)
    pdf.set_text_color(20, 20, 20)
    pdf.set_xy(x0, y0)
    pdf.cell(w, 4, "ROOM SCHEDULE", align="C")
    pdf.ln(5)

    row_h = 4.5
    col_name = w * 0.62
    col_sf   = w * 0.38

    def header():
        pdf.set_fill_color(40, 40, 40)
        pdf.set_text_color(255, 255, 255)
        pdf.set_font("Helvetica", "B", 4.5)
        cx = pdf.get_x()
        cy = pdf.get_y()
        pdf.rect(cx, cy, col_name, row_h, style="F")
        pdf.set_xy(cx, cy)
        pdf.cell(col_name, row_h, "ROOM", align="L", border=0)
        pdf.rect(cx + col_name, cy, col_sf, row_h, style="F")
        pdf.set_xy(cx + col_name, cy)
        pdf.cell(col_sf, row_h, "SF", align="R", border=0)
        pdf.ln(row_h)
        pdf.set_text_color(20, 20, 20)

    def room_rows(room_list: list, floor_label: str = "") -> int:
        if floor_label:
            pdf.set_font("Helvetica", "B", 4)
            pdf.set_fill_color(210, 210, 210)
            cy = pdf.get_y()
            pdf.rect(x0, cy, w, row_h - 0.5, style="F")
            pdf.set_xy(x0, cy)
            pdf.cell(w, row_h - 0.5, floor_label, align="C")
            pdf.ln(row_h - 0.5)

        total_sf = 0
        _SKIP = {"garage", "patio", "deck", "rear_patio", "outdoor_living", "front_porch"}
        for i, r in enumerate(room_list):
            sf = int(r["width"] * r["height"])
            total_sf += sf
            bg = (252, 252, 252) if i % 2 == 0 else (243, 243, 243)
            pdf.set_fill_color(*bg)
            cy = pdf.get_y()
            pdf.rect(x0, cy, col_name, row_h - 0.3, style="F")
            pdf.set_font("Helvetica", "", 4.0)
            name_short = r["name"].replace("\u2014", "-").replace("\u2013", "-")[:20]
            pdf.set_xy(x0 + 0.5, cy)
            pdf.cell(col_name - 1, row_h - 0.3, name_short, align="L", border=0)
            pdf.rect(x0 + col_name, cy, col_sf, row_h - 0.3, style="F")
            pdf.set_xy(x0 + col_name, cy)
            pdf.cell(col_sf, row_h - 0.3, str(sf), align="R", border=0)
            pdf.ln(row_h - 0.3)

        # Floor total
        pdf.set_font("Helvetica", "B", 4.5)
        pdf.set_fill_color(220, 220, 220)
        cy = pdf.get_y()
        pdf.rect(x0, cy, w, row_h, style="F")
        pdf.set_xy(x0 + 0.5, cy)
        label = "FLOOR TOTAL" if floor_label else "SUBTOTAL"
        pdf.cell(col_name - 1, row_h, label, align="L", border=0)
        pdf.set_xy(x0 + col_name, cy)
        pdf.cell(col_sf, row_h, f"{total_sf:,}", align="R", border=0)
        pdf.ln(row_h + 1)
        return total_sf

    header()
    if f2:
        t1 = room_rows(f1, "FLOOR 1")
        t2 = room_rows(f2, "FLOOR 2")
        # Grand total
        pdf.set_font("Helvetica", "B", 5)
        pdf.set_fill_color(40, 40, 40)
        pdf.set_text_color(255, 255, 255)
        cy = pdf.get_y()
        pdf.rect(x0, cy, w, row_h + 0.5, style="F")
        pdf.set_xy(x0 + 0.5, cy)
        pdf.cell(col_name - 1, row_h + 0.5, "TOTAL LIVING AREA", align="L", border=0)
        pdf.set_xy(x0 + col_name, cy)
        pdf.cell(col_sf, row_h + 0.5, f"{t1 + t2:,} SF", align="R", border=0)
        pdf.set_text_color(20, 20, 20)
    else:
        room_rows(f1)


# ─── Title block ──────────────────────────────────────────────────────────────

def _draw_title_block(pdf: FPDF, plan: dict) -> None:
    """Draw bottom title block across full page width."""
    ty = PAGE_H_MM - MARGIN_MM - TITLE_H_MM
    tw = PAGE_W_MM - MARGIN_MM * 2

    pdf.set_draw_color(20, 20, 20)
    pdf.set_line_width(0.4)
    pdf.line(MARGIN_MM, ty, MARGIN_MM + tw, ty)

    # Project name
    name = plan.get("name", "Floor Plan").replace("\u2014", "-").replace("\u2013", "-")
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(20, 20, 20)
    pdf.set_xy(MARGIN_MM + 2, ty + 3)
    pdf.cell(tw * 0.4, 7, name.upper(), align="L")

    # Style + sqft
    style_label = plan.get("style", "modern").title()
    rooms = plan.get("rooms", [])
    living_sf = sum(
        r["width"] * r["height"] for r in rooms
        if r.get("type") not in {"garage", "patio", "deck", "rear_patio", "front_porch"}
    )
    footprint = f"{int(plan.get('totalWidth', 0))}' × {int(plan.get('totalHeight', 0))}'"

    pdf.set_font("Helvetica", "", 7.5)
    pdf.set_xy(MARGIN_MM + 2, ty + 12)
    pdf.cell(tw * 0.4, 5, f"{style_label} Style | {int(living_sf):,} SF Living | {footprint} Footprint")

    # Date + "BUILDIFY" branding
    pdf.set_font("Helvetica", "B", 7)
    pdf.set_xy(MARGIN_MM + tw * 0.65, ty + 3)
    pdf.cell(tw * 0.35, 5, "BUILDIFY AI", align="R")

    pdf.set_font("Helvetica", "", 6)
    pdf.set_xy(MARGIN_MM + tw * 0.65, ty + 9)
    pdf.cell(tw * 0.35, 4, f"Generated {date.today().strftime('%B %d, %Y')}", align="R")

    # IRC note
    pdf.set_font("Helvetica", "I", 5)
    pdf.set_text_color(120, 120, 120)
    pdf.set_xy(MARGIN_MM + 2, ty + 19)
    pdf.cell(tw * 0.9, 4, "PRELIMINARY DESIGN - NOT FOR CONSTRUCTION  |  IRC compliant minimum dimensions")
    pdf.set_text_color(0, 0, 0)


# ─── Public API ───────────────────────────────────────────────────────────────

def export_to_pdf(floor_plan: dict) -> bytes:
    """
    Generate a PDF floor plan export.
    Returns raw PDF bytes.
    """
    pdf = FPDF(orientation="L", unit="mm", format="Letter")
    pdf.set_margins(0, 0, 0)
    pdf.set_auto_page_break(False)
    pdf.add_page()

    stories = floor_plan.get("stories", 1)

    if stories == 2:
        # Two-page PDF: one per floor
        for floor in (1, 2):
            if floor == 2:
                pdf.add_page()
            _draw_plan(pdf, floor_plan, DRAW_X, DRAW_Y, DRAW_W, DRAW_H, floor=floor)
            _draw_schedule(pdf, floor_plan, SCHED_X, DRAW_Y, SCHED_W, DRAW_H)
            _draw_title_block(pdf, floor_plan)
    else:
        _draw_plan(pdf, floor_plan, DRAW_X, DRAW_Y, DRAW_W, DRAW_H, floor=1)
        _draw_schedule(pdf, floor_plan, SCHED_X, DRAW_Y, SCHED_W, DRAW_H)
        _draw_title_block(pdf, floor_plan)

    return bytes(pdf.output())
