import re
import math
import ezdxf
from typing import List, Dict, Any, Optional, Tuple

# Layer / annotation convention (see agent_b/fixtures/generate_sample_dxf.py for a worked example):
#   A-ROOM        closed LWPOLYLINE -> room footprint boundary, world XY, meters
#   A-ROOM-ID     TEXT inside the room polygon -> room name
#   A-ROOM-DATA   TEXT inside the room polygon -> "CH=<ceiling_height_m>"
#   A-DOOR        LINE on a room-boundary edge -> door opening extents
#   A-DOOR-DATA   TEXT near the door midpoint -> "HEAD=<m>" (sill assumed 0.0 = floor)
#   A-GLAZ        LINE on a room-boundary edge -> window opening extents
#   A-GLAZ-DATA   TEXT near the window midpoint -> "SILL=<m>;HEAD=<m>"
#
# This is a real, but narrow, convention: it assumes rooms are drawn as their own boundary
# polyline (common on AIA-style layer standards) rather than derived from a wall-centerline
# network, and that vertical dimensions are annotated as text rather than carried as block
# attributes. Real client files that don't follow it will extract nothing for a room, which
# is intentional -- extract_rooms() never guesses geometry it wasn't given.

DEFAULT_DOOR_HEAD_M = 2.10
DEFAULT_WINDOW_SILL_M = 0.90
DEFAULT_WINDOW_HEAD_M = 2.10
NEAR_TOLERANCE_M = 1.0


def _point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
    """Standard ray-casting point-in-polygon test. No shapely dependency needed for this."""
    x, y = point
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside


def _polygon_area(polygon: List[Tuple[float, float]]) -> float:
    area = 0.0
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _midpoint(a: Tuple[float, float], b: Tuple[float, float]) -> Tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _nearest_edge(polygon: List[Tuple[float, float]], point: Tuple[float, float]) -> int:
    """Returns the index of the polygon edge (i -> i+1) closest to `point`."""
    best_idx, best_dist = 0, float("inf")
    n = len(polygon)
    for i in range(n):
        a, b = polygon[i], polygon[(i + 1) % n]
        # Distance from point to segment a-b
        ax, ay = a
        bx, by = b
        px, py = point
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        t = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / length_sq))
        proj = (ax + t * dx, ay + t * dy)
        d = _dist(point, proj)
        if d < best_dist:
            best_dist, best_idx = d, i
    return best_idx


class DWGParser:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.doc = None
        self.msp = None

    def load(self) -> bool:
        """Loads the DWG/DXF file."""
        try:
            self.doc = ezdxf.readfile(self.file_path)
            self.msp = self.doc.modelspace()
            print(f"Successfully loaded {self.file_path}")
            return True
        except IOError:
            print(f"Not a DXF file or a generic I/O error.")
            return False
        except ezdxf.DXFStructureError:
            print(f"Invalid or corrupted DXF file.")
            return False

    def _texts_on_layer(self, layer: str) -> List[Dict[str, Any]]:
        out = []
        for e in self.msp.query(f'TEXT[layer=="{layer}"]'):
            out.append({"content": e.dxf.text, "pos": (e.dxf.insert.x, e.dxf.insert.y)})
        for e in self.msp.query(f'MTEXT[layer=="{layer}"]'):
            out.append({"content": e.plain_text(), "pos": (e.dxf.insert.x, e.dxf.insert.y)})
        return out

    def _room_polygons(self) -> List[List[Tuple[float, float]]]:
        polygons = []
        for e in self.msp.query('LWPOLYLINE[layer=="A-ROOM"]'):
            if not e.closed:
                print(f"Skipping A-ROOM polyline (handle {e.dxf.handle}): not closed.")
                continue
            pts = [(p[0], p[1]) for p in e.get_points()]
            polygons.append(pts)
        return polygons

    def _openings_on_layer(self, room_polygon: List[Tuple[float, float]], layer: str,
                            data_layer: str, default_sill: float, default_head: float,
                            has_sill: bool) -> List[Dict[str, Any]]:
        openings = []
        data_texts = self._texts_on_layer(data_layer)

        for e in self.msp.query(f'LINE[layer=="{layer}"]'):
            a = (e.dxf.start.x, e.dxf.start.y)
            b = (e.dxf.end.x, e.dxf.end.y)
            mid = _midpoint(a, b)

            edge_idx = _nearest_edge(room_polygon, mid)
            wall_a, wall_b = room_polygon[edge_idx], room_polygon[(edge_idx + 1) % len(room_polygon)]
            wall_len = _dist(wall_a, wall_b)
            offset_along_wall = _dist(wall_a, a) if _dist(wall_a, a) < _dist(wall_a, b) else _dist(wall_a, b)
            width_m = _dist(a, b)

            sill_m, head_m, confidence = default_sill, default_head, "assumed-default"
            nearest_text = None
            nearest_text_dist = float("inf")
            for t in data_texts:
                d = _dist(t["pos"], mid)
                if d <= NEAR_TOLERANCE_M and d < nearest_text_dist:
                    nearest_text, nearest_text_dist = t, d

            if nearest_text:
                content = nearest_text["content"]
                head_match = re.search(r"HEAD=([\d.]+)", content)
                sill_match = re.search(r"SILL=([\d.]+)", content)
                if head_match:
                    head_m = float(head_match.group(1))
                    confidence = "extracted"
                if has_sill and sill_match:
                    sill_m = float(sill_match.group(1))

            openings.append({
                "wall_edge_index": edge_idx,
                "offset_along_wall_m": round(offset_along_wall, 3),
                "wall_length_m": round(wall_len, 3),
                "width_m": round(width_m, 3),
                "sill_m": sill_m,
                "head_m": head_m,
                "height_confidence": confidence,
            })
        return openings

    def extract_rooms(self) -> List[Dict[str, Any]]:
        """
        Extracts room geometry from the DWG using the layer convention documented at the
        top of this file. Returns [] (not a guess) for anything the convention doesn't cover.
        """
        if not self.msp:
            print("Modelspace not loaded. Call load() first.")
            return []

        rooms = []
        room_id_texts = self._texts_on_layer("A-ROOM-ID")
        room_data_texts = self._texts_on_layer("A-ROOM-DATA")

        for polygon in self._room_polygons():
            room_name = "Unnamed Room"
            for t in room_id_texts:
                if _point_in_polygon(t["pos"], polygon):
                    room_name = t["content"]
                    break

            ceiling_height_m: Optional[float] = None
            ceiling_confidence = "missing"
            for t in room_data_texts:
                if _point_in_polygon(t["pos"], polygon):
                    match = re.search(r"CH=([\d.]+)", t["content"])
                    if match:
                        ceiling_height_m = float(match.group(1))
                        ceiling_confidence = "extracted"
                    break

            doors = self._openings_on_layer(polygon, "A-DOOR", "A-DOOR-DATA",
                                             default_sill=0.0, default_head=DEFAULT_DOOR_HEAD_M,
                                             has_sill=False)
            windows = self._openings_on_layer(polygon, "A-GLAZ", "A-GLAZ-DATA",
                                               default_sill=DEFAULT_WINDOW_SILL_M,
                                               default_head=DEFAULT_WINDOW_HEAD_M, has_sill=True)

            # extraction_confidence reflects whether anything downstream would be *assumed*
            # rather than read from the file. Ceiling height is the big one -- see the
            # architectural-visualization spec: a wrong vertical dimension is the single most
            # visible error in a render, so rooms with a missing ceiling height must not
            # silently fall through to a hardcoded default.
            if ceiling_height_m is None:
                extraction_confidence = "needs-review"
            elif any(o["height_confidence"] == "assumed-default" for o in doors + windows):
                extraction_confidence = "approximate"
            else:
                extraction_confidence = "auto"

            rooms.append({
                "room_name": room_name,
                "polygon": [list(p) for p in polygon],
                "area_m2": round(_polygon_area(polygon), 2),
                "ceiling_height_m": ceiling_height_m,
                "ceiling_height_confidence": ceiling_confidence,
                "doors": doors,
                "windows": windows,
                "extraction_confidence": extraction_confidence,
            })

        return rooms


if __name__ == "__main__":
    import os
    import json
    sample_path = os.path.join(os.path.dirname(__file__), "fixtures", "sample_room.dxf")
    parser = DWGParser(sample_path)
    if parser.load():
        rooms = parser.extract_rooms()
        print(f"Found {len(rooms)} rooms.")
        print(json.dumps(rooms, indent=2))
