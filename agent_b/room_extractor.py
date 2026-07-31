"""
Convention-driven room extraction: rooms, doors and windows from any drawing whose layers
can be identified, rather than from one office's layer names and one file's entity handles.

Replaces two hardcodings that made the previous path unrepeatable:

  LAYER NAMES. dwg_parser.py matched A-ROOM / A-DOOR / A-GLAZ literally. The real client file
  uses 0-AREAS / A-PUERTAS / A-VIDRIO, so nothing matched. Layers now come from
  layer_conventions.detect().

  WHICH POLYLINES ARE ROOMS. The working extractor named four DXF entity handles --
  "15BE1F", "15BE24" -- to pick this apartment's rooms out of 227 area polygons for the whole
  building. Handles are internal ids; re-saving the file can change them. Selection is now
  spatial (a region, an area range), which is a property of the drawing rather than of one
  copy of it.

OPENINGS ARE USUALLY BLOCKS, NOT LINES. This is why the previous parser found no windows at
all in a drawing containing 46 glazing entities: it queried only LINE, and the file holds 23
INSERTs, 20 LWPOLYLINEs and 3 LINEs on that layer. Doors are worse -- 227 INSERTs against 72
LINEs. Openings here are measured from their bounding box, whatever entity type carries them,
so a door drawn as a block, a polyline or a plain line all extract the same way.
"""
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_DOOR_SILL_M = 0.0
DEFAULT_DOOR_HEAD_M = 2.10
DEFAULT_WINDOW_SILL_M = 0.90
DEFAULT_WINDOW_HEAD_M = 2.10

# How close an opening's centre must be to a room's boundary to belong to that room. Walls
# are 0.10-0.25 m thick here and openings are drawn on either face, so this must exceed a
# wall thickness without being so loose that an opening claims the room across the corridor.
OPENING_TO_WALL_TOLERANCE_M = 0.45
# Openings narrower than this are hardware (handles, hinges) rather than an aperture.
MIN_OPENING_WIDTH_M = 0.30
MAX_OPENING_WIDTH_M = 6.0


def _entity_bbox(entity, doc) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    2D bounding box of any entity, including blocks.

    ezdxf.bbox virtualises INSERTs -- resolving the block definition and applying scale and
    rotation -- which is the only reason block-based doors can be measured at all. It fails
    on some entities (empty blocks, unsupported types); those are skipped rather than
    guessed at.
    """
    from ezdxf import bbox

    try:
        extents = bbox.extents([entity], cache=None)
    except Exception:                                # noqa: BLE001 - unsupported entity
        return None
    if not extents.has_data:
        return None
    return (np.array([extents.extmin.x, extents.extmin.y]),
            np.array([extents.extmax.x, extents.extmax.y]))


def _closed_polylines(msp, layer: str) -> List[Tuple[str, np.ndarray]]:
    out = []
    for e in msp.query(f'LWPOLYLINE[layer=="{layer}"]'):
        if not getattr(e, "closed", False):
            continue
        pts = np.array([(p[0], p[1]) for p in e.get_points()], dtype=float)
        if len(pts) >= 3:
            out.append((e.dxf.handle, pts))
    return out


def _polygon_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _nearest_edge(polygon: np.ndarray, point: np.ndarray) -> Tuple[int, float, float]:
    """(edge index, distance to it, distance along it) for the closest boundary edge."""
    best = (0, math.inf, 0.0)
    n = len(polygon)
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            continue
        t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
        distance = float(np.linalg.norm(point - (a + t * ab)))
        if distance < best[1]:
            best = (i, distance, t * math.sqrt(denom))
    return best


def _opening_from_bbox(polygon: np.ndarray, lo: np.ndarray, hi: np.ndarray
                       ) -> Optional[Dict[str, Any]]:
    """
    Turn a bounding box into an opening on the nearest wall.

    Width is the box's extent measured ALONG the wall, not the box's diagonal or its larger
    side: a door block's bounding box includes its swing arc, which is roughly as deep as the
    door is wide. Taking the larger side would report a 0.8 m door as 1.1 m wide. Projecting
    onto the wall direction discards the swing.
    """
    centre = (lo + hi) / 2.0
    edge_index, distance, _ = _nearest_edge(polygon, centre)
    if distance > OPENING_TO_WALL_TOLERANCE_M:
        return None

    n = len(polygon)
    a, b = polygon[edge_index], polygon[(edge_index + 1) % n]
    edge_length = float(np.linalg.norm(b - a))
    if edge_length <= 1e-9:
        return None
    direction = (b - a) / edge_length

    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]], [hi[0], hi[1]], [lo[0], hi[1]]])
    projections = [float(np.dot(c - a, direction)) for c in corners]
    start, end = min(projections), max(projections)
    width = end - start
    if not (MIN_OPENING_WIDTH_M <= width <= MAX_OPENING_WIDTH_M):
        return None

    start = max(0.0, min(edge_length - width, start))
    return {
        "wall_edge_index": edge_index,
        "offset_along_wall_m": round(start, 3),
        "wall_length_m": round(edge_length, 3),
        "width_m": round(width, 3),
        "wall_distance_m": round(distance, 3),
    }


def _openings_on_layer(msp, doc, polygon: np.ndarray, layer: Optional[str],
                       sill: float, head: float) -> List[Dict[str, Any]]:
    if not layer:
        return []
    found: List[Dict[str, Any]] = []
    for entity in msp.query(f'*[layer=="{layer}"]'):
        box = _entity_bbox(entity, doc)
        if box is None:
            continue
        opening = _opening_from_bbox(polygon, *box)
        if opening is None:
            continue
        opening.update(sill_m=sill, head_m=head, height_confidence="assumed-default",
                       source_type=entity.dxftype())
        found.append(opening)

    # One physical door is often drawn as several entities on the same layer -- a leaf block,
    # a swing arc, a threshold line. Without this, a single door becomes three openings and
    # the blockout cuts three overlapping holes.
    found.sort(key=lambda o: (o["wall_edge_index"], o["offset_along_wall_m"]))
    merged: List[Dict[str, Any]] = []
    for opening in found:
        previous = merged[-1] if merged else None
        if (previous and previous["wall_edge_index"] == opening["wall_edge_index"]
                and opening["offset_along_wall_m"] <= previous["offset_along_wall_m"]
                + previous["width_m"] - 0.05):
            far = max(previous["offset_along_wall_m"] + previous["width_m"],
                      opening["offset_along_wall_m"] + opening["width_m"])
            previous["width_m"] = round(far - previous["offset_along_wall_m"], 3)
        else:
            merged.append(opening)
    return merged


def _label_for(msp, layer: Optional[str], polygon: np.ndarray) -> Optional[str]:
    """Nearest text inside the room outline, if there is a label layer at all."""
    if not layer:
        return None
    from matplotlib.path import Path            # already a dependency, and fast for this

    path = Path(polygon)
    best, best_distance = None, math.inf
    centre = polygon.mean(axis=0)
    for kind in ("TEXT", "MTEXT"):
        for e in msp.query(f'{kind}[layer=="{layer}"]'):
            try:
                position = np.array([e.dxf.insert.x, e.dxf.insert.y], dtype=float)
            except AttributeError:
                continue
            if not path.contains_point(position):
                continue
            content = e.plain_text() if kind == "MTEXT" else e.dxf.text
            content = (content or "").strip()
            if not content:
                continue
            distance = float(np.linalg.norm(position - centre))
            if distance < best_distance:
                best, best_distance = content, distance
    return best


def extract_rooms(msp, doc, detection: Dict[str, Any],
                  region: Sequence[float] = None,
                  min_area_m2: float = 2.0,
                  max_area_m2: float = 400.0) -> List[Dict[str, Any]]:
    """
    Extract rooms using the detected layers.

    `region` is (xmin, ymin, xmax, ymax); rooms whose centre falls outside it are skipped.
    That is how one apartment is selected out of a whole-building drawing -- by where it is,
    not by which entity handles it happens to have.

    Returns [] when no room layer was identified. That is a true statement about the drawing
    and is preferable to inferring rooms from wall geometry, which this does not attempt.
    """
    layers = detection["layers"]
    if not layers.get("room_boundary"):
        return []

    rooms: List[Dict[str, Any]] = []
    for handle, polygon in _closed_polylines(msp, layers["room_boundary"]):
        area = _polygon_area(polygon)
        if not (min_area_m2 <= area <= max_area_m2):
            continue
        centre = polygon.mean(axis=0)
        if region is not None:
            xmin, ymin, xmax, ymax = region
            if not (xmin <= centre[0] <= xmax and ymin <= centre[1] <= ymax):
                continue

        rooms.append({
            "room_name": _label_for(msp, layers.get("room_label"), polygon) or f"Room {handle}",
            "source_handle": handle,
            "polygon": [[round(float(x), 3), round(float(y), 3)] for x, y in polygon],
            "area_m2": round(area, 2),
            "doors": _openings_on_layer(msp, doc, polygon, layers.get("door"),
                                        DEFAULT_DOOR_SILL_M, DEFAULT_DOOR_HEAD_M),
            "windows": _openings_on_layer(msp, doc, polygon, layers.get("window"),
                                          DEFAULT_WINDOW_SILL_M, DEFAULT_WINDOW_HEAD_M),
            "ceiling_height_m": None,
            "height_confidence": "needs-review",
        })
    return rooms
