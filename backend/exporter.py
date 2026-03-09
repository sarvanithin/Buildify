import ezdxf
from ezdxf.enums import TextEntityAlignment
import io


def export_to_dxf(floor_plan: dict) -> bytes:
    doc = ezdxf.new("R2010")
    msp = doc.modelspace()

    doc.layers.add("BOUNDARY", color=1)
    doc.layers.add("ROOMS", color=7)
    doc.layers.add("LABELS", color=3)
    doc.layers.add("DIMENSIONS", color=2)

    total_w = float(floor_plan.get("totalWidth", 40))
    total_h = float(floor_plan.get("totalHeight", 35))

    # Outer boundary
    msp.add_lwpolyline(
        [(0, 0), (total_w, 0), (total_w, total_h), (0, total_h)],
        close=True,
        dxfattribs={"layer": "BOUNDARY", "lineweight": 50},
    )

    for room in floor_plan.get("rooms", []):
        rx = float(room["x"])
        ry_dxf = total_h - float(room["y"]) - float(room["height"])  # flip Y axis
        rw = float(room["width"])
        rh = float(room["height"])

        # Room rectangle
        msp.add_lwpolyline(
            [(rx, ry_dxf), (rx + rw, ry_dxf), (rx + rw, ry_dxf + rh), (rx, ry_dxf + rh)],
            close=True,
            dxfattribs={"layer": "ROOMS", "lineweight": 25},
        )

        cx = rx + rw / 2
        cy = ry_dxf + rh / 2
        font_h = min(rw, rh) * 0.09

        name_text = msp.add_text(
            room.get("name", "Room"),
            dxfattribs={"layer": "LABELS", "height": max(1.0, font_h)},
        )
        name_text.set_placement((cx, cy + 0.6), align=TextEntityAlignment.MIDDLE_CENTER)

        dim_text = msp.add_text(
            f"{rw:.0f}' x {rh:.0f}'",
            dxfattribs={"layer": "DIMENSIONS", "height": max(0.7, font_h * 0.7)},
        )
        dim_text.set_placement((cx, cy - 0.6), align=TextEntityAlignment.MIDDLE_CENTER)

    stream = io.BytesIO()
    doc.write(stream)
    stream.seek(0)
    return stream.getvalue()
