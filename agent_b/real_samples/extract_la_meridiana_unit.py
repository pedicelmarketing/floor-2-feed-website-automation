"""
First real-file test: extracts one apartment-sized cluster of rooms from the actual
"LA MERIDIANA" DWG (converted to DXF via dwg2dxf/LibreDWG) and builds the same blockout +
control-map pipeline validated against the synthetic fixture.

This file's layer convention is completely different from the synthetic fixture's
(agent_b/dwg_parser.py's A-ROOM/A-ROOM-ID/A-ROOM-DATA convention doesn't exist here):

  0-AREAS         closed LWPOLYLINE per room (areas 2-49 m2 -- individual rooms, not
                  whole apartments; overlapping/duplicate polygons occur in practice)
  A-ANOT-TEXTO    MTEXT area labels like "A1= 5,6 mt²" near (not always exactly aligned
                  with) each room's area -- the closest thing to a room name available
  A-MUROS         wall footprint polygons (already drawn as thickness rectangles)
  A-PUERTAS       door INSERTs, referencing anonymized blocks (LibreDWG renames blocks
                  it can't preserve the original name for to "*U###") -- width has to be
                  read from the block's own local geometry, not the block name
  A-VIDRIO        windows (none found within reach of this specific room cluster)
  A-SANITARIOS    bathroom/laundry fixtures (toilets, sinks, washing machines)
  A-MOB           furniture

Room selection: automatic proximity-clustering (buffer + union on 0-AREAS polygons)
found exactly one small (3-7 room) cluster isolated enough to read as a single unit
rather than a whole shared-wall floor -- everything else merged into building-sized
blobs, which is a real limitation to solve later (apartment-unit segmentation on a
shared-wall floor is a harder, separate problem from single-room extraction).

Two of that cluster's five 0-AREAS polygons (handles 15BE25 and 17EBE1) overlap almost
exactly and both contain the same laundry-machine fixtures -- a real duplicate/overlap
in the source data, not a bug in this script. Only 17EBE1 (the larger of the two) is kept.

Ceiling height: this file DOES use a per-room "h=2.70" style note (found elsewhere in the
building, e.g. next to a kitchen -- confirmed genuine by nearby fixture context), but only
in scattered spots, not systematically. A naive `h=` search also false-positives on area
labels like "E&H= 10,15mt²" (an "Estar & Hall" combined-area label, not a height) since the
substring "h=" appears right after the "&". HEIGHT_PATTERN below requires the character
before "h"/"H" not be a letter or "&" to reject that case, then falls back to the assumed
default -- same rule as before, just checking a real source first.
"""
import os
import re
import sys
import json
import math
import ezdxf
from ezdxf import recover
import shapely.geometry

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from dwg_parser import _dist, _nearest_edge, _midpoint, _polygon_area  # noqa: E402

_ASSETS = os.path.join(os.path.dirname(__file__), "..", "..", "Assets")
DWG_PATH = os.path.join(_ASSETS, "PLANOS CAD LA MERIDIANA.dwg")
DXF_PATH = os.path.join(_ASSETS, "converted", "la_meridiana_clean.dxf")


def ensure_dxf(dxf_path: str = DXF_PATH, dwg_path: str = DWG_PATH) -> str:
    """
    Return a loadable DXF, rebuilding it from the DWG if it is missing.

    The DXF used to be a build artefact with no recorded provenance -- produced by a manual
    command that was never committed, so nothing could regenerate it from the source DWG.
    That made the pipeline's first stage unreproducible. dwg_convert now performs that step,
    and verifies the output opens rather than trusting the converter's exit code.
    """
    if os.path.exists(dxf_path):
        return dxf_path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from dwg_convert import convert

    report = convert(dwg_path, dxf_path)
    if not report["ok"]:
        raise RuntimeError(
            f"No available converter produced a loadable DXF from {dwg_path}.\n"
            + "\n".join(f"  {a['backend']}: {a['note']}" for a in report["attempts"]))
    print(f"Rebuilt DXF from DWG via {report['detail']}")
    return dxf_path
ASSUMED_CEILING_HEIGHT_M = 2.6
HEIGHT_NEAR_TOLERANCE_M = 2.0
HEIGHT_PATTERN = re.compile(r'(?<![A-Za-z&])h\s*=\s*(\d+[.,]\d+)', re.IGNORECASE)

ROOM_DEFS = [
    ("15BE1F", "Bedroom / Living (labels D4 + GG)"),
    ("15BE24", "Hall (label HL)"),
    ("15BE26", "Bathroom (label A1)"),
    ("17EBE1", "Laundry (label G)"),
]

DOOR_HANDLES = ["15BDE8", "1F4BA4", "2420FE", "292732", "2BFFB1"]


def load_room_polygons(msp):
    polygons = {}
    for e in msp.query('LWPOLYLINE[layer=="0-AREAS"]'):
        if e.dxf.handle in {h for h, _ in ROOM_DEFS}:
            pts = [(p[0], p[1]) for p in e.get_points()]
            polygons[e.dxf.handle] = pts
    return polygons


def door_local_width(doc, insert_entity):
    """Door width = local block's own X-extent, not the world-space (rotated) bbox --
    keeps width meaningful regardless of the block's rotation (0/90/270 seen here)."""
    block = doc.blocks.get(insert_entity.dxf.name)
    xs = []
    for be in block:
        if be.dxftype() == "LINE":
            xs += [be.dxf.start.x, be.dxf.end.x]
        elif be.dxftype() == "LWPOLYLINE":
            xs += [p[0] for p in be.get_points()]
        elif be.dxftype() == "ARC":
            xs.append(be.dxf.center.x)
    return (max(xs) - min(xs)) if xs else 0.8  # 0.8m fallback if the block is unreadable


def assign_door_to_room(door_point, rooms_polygons):
    """Finds which room polygon this door insert point sits closest to, and where on
    that room's boundary -- same nearest-edge logic as the synthetic parser."""
    best = None
    for handle, polygon in rooms_polygons.items():
        edge_idx = _nearest_edge(polygon, door_point)
        a, b = polygon[edge_idx], polygon[(edge_idx + 1) % len(polygon)]
        ax, ay = a; bx, by = b; px, py = door_point
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        t = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
        proj = (ax + t * dx, ay + t * dy)
        dist = _dist(door_point, proj)
        if best is None or dist < best[0]:
            best = (dist, handle, edge_idx)
    return best[1], best[2]


def find_height_labels(msp):
    """Real per-room 'h=2.70' style notes, wherever they exist in the file (not just near
    our cluster -- callers do their own proximity check against the returned positions)."""
    labels = []
    for e in list(msp.query('TEXT[layer=="A-ANOT-TEXTO"]')) + list(msp.query('MTEXT[layer=="A-ANOT-TEXTO"]')):
        text = e.dxf.text if e.dxftype() == "TEXT" else e.plain_text()
        m = HEIGHT_PATTERN.search(text)
        if m:
            labels.append((float(m.group(1).replace(",", ".")), (e.dxf.insert.x, e.dxf.insert.y)))
    return labels


def room_ceiling_height(polygon, height_labels):
    """Prefers a real height label sitting inside (or very close to) this room's polygon
    over the assumed default -- checked per room, since a real value found for one room
    says nothing about its neighbors."""
    poly = shapely.geometry.Polygon(polygon)
    best = None
    for height, pos in height_labels:
        point = shapely.geometry.Point(pos)
        dist = 0.0 if poly.contains(point) else poly.exterior.distance(point)
        if dist <= HEIGHT_NEAR_TOLERANCE_M and (best is None or dist < best[0]):
            best = (dist, height)
    if best is not None:
        return best[1], "extracted"
    return ASSUMED_CEILING_HEIGHT_M, "assumed-default"


def extract(dxf_path: str = None):
    doc, auditor = recover.readfile(dxf_path or ensure_dxf())
    msp = doc.modelspace()

    raw_polygons = load_room_polygons(msp)
    height_labels = find_height_labels(msp)
    rooms = {}
    for h, name in ROOM_DEFS:
        ceiling_height_m, confidence = room_ceiling_height(raw_polygons[h], height_labels)
        rooms[h] = {"room_name": name, "polygon": raw_polygons[h],
                    "area_m2": round(_polygon_area(raw_polygons[h]), 1),
                    "ceiling_height_m": ceiling_height_m,
                    "ceiling_height_confidence": confidence,
                    "doors": [], "windows": [],
                    "extraction_confidence": "approximate"}

    for e in msp.query('INSERT[layer=="A-PUERTAS"]'):
        if e.dxf.handle not in DOOR_HANDLES:
            continue
        point = (e.dxf.insert.x, e.dxf.insert.y)
        width = door_local_width(doc, e)
        polygons_by_handle = {h: r["polygon"] for h, r in rooms.items()}
        room_handle, edge_idx = assign_door_to_room(point, polygons_by_handle)
        polygon = rooms[room_handle]["polygon"]
        a, b = polygon[edge_idx], polygon[(edge_idx + 1) % len(polygon)]
        wall_len = _dist(a, b)
        offset = _dist(a, point)
        offset = max(0.0, min(wall_len - width, offset - width / 2))
        rooms[room_handle]["doors"].append({
            "wall_edge_index": edge_idx,
            "offset_along_wall_m": round(offset, 3),
            "wall_length_m": round(wall_len, 3),
            "width_m": round(width, 3),
            "sill_m": 0.0,
            "head_m": 2.10,
            "height_confidence": "assumed-default",
        })

    return list(rooms.values())


if __name__ == "__main__":
    rooms = extract()
    print(f"Extracted {len(rooms)} rooms from a real DWG-derived floor plan.")
    print(json.dumps(rooms, indent=2))
    out_path = os.path.join(os.path.dirname(__file__), "la_meridiana_unit.json")
    with open(out_path, "w") as f:
        json.dump(rooms, f, indent=2)
    print(f"\nWrote {out_path}")
