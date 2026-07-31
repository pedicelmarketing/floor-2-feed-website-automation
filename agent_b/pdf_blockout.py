"""
Build a 3D blockout directly from a PDF floor plan's wall outlines.

This models WALLS, where the DWG path models ROOMS. The difference is not cosmetic. Extruding
a room outline produces a void with walls implied around it, so a doorway has to be cut into
room A and then mirrored onto room B's facing wall, and the two have to line up. That step is
agent_b/door_pairing.py, it broke three separate times, and on the one apartment it was tuned
for it still only achieves 0.30 m of overlap on an 0.80 m door.

Extruding the wall outlines removes the problem rather than fixing it. A doorway is a gap the
architect already drew, present once, in the only wall there is. Nothing to mirror, nothing to
align, no facing-wall search. The blockout also gains real wall thickness and exterior walls,
neither of which a room-void model has at all.

CEILING HEIGHT IS NOT IN THE FILE. Eleven pages of this drawing contain no "h=" note of any
kind, unlike the DWG which carries them sporadically. Height is therefore a stated assumption,
flagged as such, and never presented as extracted. It is the one dimension a plan view cannot
give you.

Everything here works in metres, converted from PDF points using the scale that
wall_regions/measure_pdf_accuracy verified against 278 doors of known width.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ASSUMED_CEILING_HEIGHT_M = 2.60
# Walls drawn as open polylines rather than closed outlines get this much thickness, which is
# a typical partition. Only used for linework that has no outline of its own to extrude.
OPEN_WALL_THICKNESS_M = 0.10
# Discard scraps: hatch fragments and drafting marks that are not walls.
MIN_WALL_AREA_M2 = 0.01

WALL_LAYER = "A-MUROS"
GLAZING_LAYER = "A-VIDRIO"
# How far the thin glazing linework is widened so the fill meets solid wall on both faces.
GLAZING_REACH_M = 0.14
# Standard domestic window band. The PDF states neither, like every other vertical dimension.
WINDOW_SILL_M = 0.90
WINDOW_HEAD_M = 2.10
# Deliberately NOT included. "Proyectado" is projected geometry -- what is above, such as a
# beam or the wall over an opening. Extruding it from floor level would brick up the doorways
# this whole approach exists to keep open.
EXCLUDED_LAYERS = ("A-MUROS-PROYECTADO",)


def wall_polygons(page: Dict[str, Any], mm_per_pt: float,
                  region_m: Optional[Sequence[float]] = None,
                  wall_layer: str = WALL_LAYER) -> Tuple[List[Any], Dict[str, int]]:
    """
    Wall footprints as shapely polygons in metres.

    `region_m` is (xmin, ymin, xmax, ymax) in metres and selects one apartment out of a floor
    that holds several -- the spatial equivalent of picking a unit, rather than naming entity
    ids as the DWG path originally did.
    """
    from shapely.geometry import LineString, Polygon, box

    scale = mm_per_pt / 1000.0
    clip = box(*region_m) if region_m is not None else None
    polygons, stats = [], {"closed": 0, "open": 0, "dropped": 0}

    for polyline in page["layers"].get(wall_layer, []):
        points = np.asarray(polyline, dtype=float) * scale
        if len(points) < 2:
            stats["dropped"] += 1
            continue

        if len(points) > 3 and np.allclose(points[0], points[-1]):
            shape = Polygon(points)
            if not shape.is_valid:
                shape = shape.buffer(0)          # self-intersecting outline; repair
            stats["closed"] += 1
        else:
            shape = LineString(points).buffer(OPEN_WALL_THICKNESS_M / 2.0, cap_style=2)
            stats["open"] += 1

        if shape.is_empty or shape.area < MIN_WALL_AREA_M2:
            stats["dropped"] += 1
            continue
        if clip is not None:
            shape = shape.intersection(clip)
            if shape.is_empty or shape.area < MIN_WALL_AREA_M2:
                continue
        for part in getattr(shape, "geoms", [shape]):
            if part.geom_type == "Polygon" and part.area >= MIN_WALL_AREA_M2:
                polygons.append(part)

    return polygons, stats


def window_polygons(page: Dict[str, Any], mm_per_pt: float,
                    region_m: Optional[Sequence[float]] = None,
                    glazing_layer: str = GLAZING_LAYER,
                    reach_m: float = GLAZING_REACH_M) -> List[Any]:
    """
    Window footprints, widened enough to bridge the wall they sit in.

    The glazing is drawn as thin lines occupying a GAP the architect left in A-MUROS -- 0%
    of it overlaps the wall solids. So the wall is already open full height at every window,
    and the opening is made into a window by filling BELOW the sill and ABOVE the head, not by
    cutting a hole. These footprints are the shapes used for that fill, buffered across a
    wall's thickness so they meet solid wall on both sides.
    """
    from shapely.geometry import LineString, Polygon, box
    from shapely.ops import unary_union

    scale = mm_per_pt / 1000.0
    clip = box(*region_m) if region_m is not None else None
    shapes = []
    for polyline in page["layers"].get(glazing_layer, []):
        points = np.asarray(polyline, dtype=float) * scale
        if len(points) < 2:
            continue
        if len(points) > 3 and np.allclose(points[0], points[-1]):
            shape = Polygon(points).buffer(0).buffer(reach_m)
        else:
            shape = LineString(points).buffer(reach_m, cap_style=2)
        if clip is not None:
            shape = shape.intersection(clip)
        if not shape.is_empty and shape.area > MIN_WALL_AREA_M2:
            shapes.append(shape)
    return [g for g in getattr(unary_union(shapes), "geoms", [unary_union(shapes)])
            if not g.is_empty] if shapes else []


def build_mesh(wall_polys: List[Any], ceiling_height_m: float = ASSUMED_CEILING_HEIGHT_M,
               footprint: Optional[Sequence[float]] = None,
               add_floor: bool = True, add_ceiling: bool = True,
               window_polys: Optional[List[Any]] = None,
               sill_m: float = WINDOW_SILL_M, head_m: float = WINDOW_HEAD_M):
    """
    Extrude wall footprints into solids, with a floor and ceiling slab across the footprint.

    Floor and ceiling are not decoration: without them a camera looking down or up sees
    nothing at all, and those rays come back as void rather than as a surface at a known
    distance. The depth map would then be missing exactly the large flat regions that tell a
    video model where the ground is.

    With `window_polys`, walls are built as three horizontal bands instead of one extrusion:
    solid up to the sill, the wall alone between sill and head so the glazing gap stays open,
    and solid again above the head. Doing it by band avoids needing a mesh boolean engine --
    the shapes are combined in 2D, where the operation is exact and cheap, and only then
    extruded.
    """
    import trimesh
    from shapely.geometry import box
    from shapely.ops import unary_union

    meshes = []

    def extrude(polygon, height, z):
        if height <= 1e-6:
            return
        try:
            solid = trimesh.creation.extrude_polygon(polygon, height=height)
        except Exception:                        # noqa: BLE001 - untriangulable scrap
            return
        solid.apply_translation([0, 0, z])
        meshes.append(solid)

    if window_polys:
        walls = unary_union(wall_polys)
        filled = unary_union([walls, unary_union(window_polys)])
        bands = [(filled, 0.0, sill_m),                     # under the window
                 (walls, sill_m, head_m),                   # the opening itself
                 (filled, head_m, ceiling_height_m)]        # lintel above
        for shape, z0, z1 in bands:
            for part in getattr(shape, "geoms", [shape]):
                if part.geom_type == "Polygon" and part.area >= MIN_WALL_AREA_M2:
                    extrude(part, z1 - z0, z0)
    else:
        for polygon in wall_polys:
            extrude(polygon, ceiling_height_m, 0.0)

    if footprint is not None and (add_floor or add_ceiling):
        slab = box(*footprint)
        if add_floor:
            floor = trimesh.creation.extrude_polygon(slab, height=0.02)
            floor.apply_translation([0, 0, -0.02])
            meshes.append(floor)
        if add_ceiling:
            ceiling = trimesh.creation.extrude_polygon(slab, height=0.02)
            ceiling.apply_translation([0, 0, ceiling_height_m])
            meshes.append(ceiling)

    if not meshes:
        return None
    return trimesh.util.concatenate(meshes)


def blockout_from_page(page: Dict[str, Any], mm_per_pt: float,
                       region_m: Optional[Sequence[float]] = None,
                       ceiling_height_m: float = ASSUMED_CEILING_HEIGHT_M) -> Dict[str, Any]:
    """Walls to mesh in one call, reporting what went in and what was assumed."""
    polygons, stats = wall_polygons(page, mm_per_pt, region_m)
    windows = window_polygons(page, mm_per_pt, region_m)
    if not polygons:
        return {"mesh": None, "reason": "no wall geometry in this region", "stats": stats}

    bounds = region_m
    if bounds is None:
        xs = [p for poly in polygons for p in poly.bounds[0::2]]
        ys = [p for poly in polygons for p in poly.bounds[1::2]]
        bounds = (min(xs), min(ys), max(xs), max(ys))

    mesh = build_mesh(polygons, ceiling_height_m, footprint=bounds, window_polys=windows)
    return {
        "mesh": mesh,
        "wall_polygons": len(polygons),
        "window_polygons": len(windows),
        "window_sill_m": WINDOW_SILL_M,
        "window_head_m": WINDOW_HEAD_M,
        "stats": stats,
        "footprint_m": bounds,
        "ceiling_height_m": ceiling_height_m,
        # The PDF states no height anywhere, so this must travel with the result rather than
        # being silently baked into the geometry.
        "height_confidence": "assumed-default",
    }
