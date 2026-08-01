"""
Turn the furniture the architect drew into volumes the 3D world can contain.

The blockout was built from walls, doors and glazing -- 55 polylines in the test apartment --
and threw away 823 others. `A-MOB` alone holds 731 lines of furniture inside that one flat,
plus 52 of stairs and 38 of bathroom fittings. The rooms handed to the video models were
therefore empty shells, and every model filled them differently. That is not a missing ground
truth so much as a discarded one: the drawing says where the bed is, and we deleted it.

THE LINEWORK IS SYMBOLS, NOT SHAPES. Only 18 of those 731 lines close into a polygon. The rest
are the marks that make a symbol legible on paper -- cushion seams, drawer fronts, the arc of a
chair back -- and none of them individually is an object. So nothing here extrudes a polyline.
Instead the lines are clustered by proximity into objects, and each object's footprint is
extruded to a height chosen from its size.

WHAT THIS IS FOR, which sets how accurate it needs to be. The goal is not to model the sofa.
It is to tell the generator *there is an object about this big, here, and you cannot see
through it*. A box in the right place with the right footprint does that. A beautifully
modelled sofa 30 cm to the left does not.

HEIGHTS ARE ASSUMED, like every other vertical dimension in this pipeline -- the drawing is
flat, all of its geometry sits at elevation zero, and the whole building carries two ceiling
notes. They are chosen by footprint and layer, and travel with a confidence flag so nothing
downstream mistakes them for measurements.
"""
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FURNITURE_LAYERS = ("A-MOB", "A-ARMARIOS", "A-EQUIPAMIENTO")
SANITARY_LAYERS = ("A-SANITARIOS", "SANITARIOS")
FIXED_LAYERS = ("A-ESCALERAS",)

# Lines closer than this belong to the same object. Wide enough to gather a sofa's seams into
# one cluster, narrow enough not to weld a bed to the wardrobe beside it. Bedside tables sit
# ~0.1 m from a bed in plan, so this is the value that decides whether they merge.
CLUSTER_GAP_M = 0.06
# Below this a cluster is hatching, a dimension tick or a stray annotation, not furniture.
MIN_OBJECT_AREA_M2 = 0.06
# Above this the clustering has welded a room's worth of symbols together and the result would
# be one box filling the room -- worse than no furniture at all.
#
# Two shapes are allowed, because furniture comes in two shapes and a single area cap cannot
# tell either of them from a merge failure.
#
#   COMPACT   up to 6 m2 whatever its proportions -- a bed, a sofa, a table.
#   RUN       up to 16 m2 but no more than 1.0 m deep -- fitted units, worktops, shelving.
#
# The history is worth keeping because the first attempt was wrong. The cap was 8.0 and was
# discarding a 9.69 m2 cluster in the living room; inspecting only its bounding box (2.08 x
# 5.12 m) it looked like a legitimate wall-length run, so the cap was raised to 16 with a 2.6 m
# depth guard. It is not a run. Testing containment afterwards showed that object swallows the
# living room's centre point, which is how the route planner discovered it -- every route in the
# flat went from clear to impossible. A 2 m deep, 5 m long solid is a room, not a sideboard.
#
# So depth is the real discriminator, and 2.6 m was far too generous: a sofa is about 0.9 m
# deep, a wardrobe 0.6 m. Anything both large AND deep is a merge, whatever its area.
MAX_COMPACT_AREA_M2 = 6.0
MAX_RUN_AREA_M2 = 16.0
MAX_RUN_DEPTH_M = 1.0
MAX_OBJECT_AREA_M2 = MAX_RUN_AREA_M2      # kept for callers that read the old name

# Heights in metres, by what the object is and how big its footprint is. Deliberately coarse:
# there is no symbol recognition here, so these are the heights that make an object read as an
# obstacle of the right kind rather than an attempt at the real dimension.
HEIGHTS = {
    "sanitary": 0.80,          # WC, basin, bath rim
    "fixed": 1.00,             # stairs and built-ins, treated as solid
    "tall": 2.10,              # narrow and deep footprint -> wardrobe, tall unit
    "large": 0.75,             # sofa, bed, table
    "small": 0.50,             # chair, side table, appliance
}


def _classify(footprint, layer: str) -> str:
    """Pick a height class from the layer and the footprint's proportions."""
    if layer in SANITARY_LAYERS:
        return "sanitary"
    if layer in FIXED_LAYERS:
        return "fixed"
    bounds = footprint.bounds
    width = bounds[2] - bounds[0]
    depth = bounds[3] - bounds[1]
    short, long_ = min(width, depth), max(width, depth)
    # A wardrobe is long and shallow. The threshold is generous because a shallow footprint on
    # a furniture layer is far more often a tall unit than anything else.
    if short <= 0.75 and long_ >= 1.2:
        return "tall"
    return "large" if footprint.area >= 0.9 else "small"


def extract_objects(page: Dict[str, Any], mm_per_pt: float,
                    region_m: Optional[Sequence[float]] = None,
                    layers: Optional[Sequence[str]] = None,
                    cluster_gap_m: float = CLUSTER_GAP_M) -> List[Dict[str, Any]]:
    """
    Cluster symbol linework into objects with a footprint, a height and a confidence.

    Clustering is done per layer rather than across all of them at once, so a chair drawn
    beside a WC does not merge into a single blob. The layer is also what decides the height
    class, and merging would lose it.
    """
    from shapely.geometry import LineString, Polygon, box
    from shapely.ops import unary_union

    scale = mm_per_pt / 1000.0
    clip = box(*region_m) if region_m is not None else None
    wanted = layers or (FURNITURE_LAYERS + SANITARY_LAYERS + FIXED_LAYERS)

    objects: List[Dict[str, Any]] = []
    for layer in wanted:
        pieces = []
        for polyline in page["layers"].get(layer, []):
            points = np.asarray(polyline, dtype=float) * scale
            if len(points) < 2:
                continue
            if clip is not None:
                centre = points.mean(axis=0)
                if not (region_m[0] <= centre[0] <= region_m[2]
                        and region_m[1] <= centre[1] <= region_m[3]):
                    continue
            try:
                pieces.append(LineString(points).buffer(cluster_gap_m / 2.0, cap_style=2))
            except Exception:                    # noqa: BLE001 - degenerate symbol line
                continue
        if not pieces:
            continue

        merged = unary_union(pieces)
        for cluster in getattr(merged, "geoms", [merged]):
            for part in getattr(cluster, "geoms", [cluster]):
                if part.geom_type != "Polygon":
                    continue
                # Convex hull of the cluster, and deliberately NOT shrunk back by the buffer
                # first. Shrinking looks like the tidy thing to do -- undo the halo added to
                # make the lines touch -- but a cluster of thin symbol lines is itself thin,
                # and eroding it by the same amount annihilates it. Measured: with the shrink
                # this flat yielded 0.78 m2 of furniture across 7 slivers; without it, 7.26 m2
                # across the same 7 objects. The 3 cm halo left behind is far smaller than the
                # error that removing it introduced.
                #
                # Convex hull because furniture symbols are hollow outlines: the space inside a
                # sofa symbol is sofa, not a hole in one.
                solid = Polygon(part.exterior).convex_hull
                if solid.area < MIN_OBJECT_AREA_M2:
                    continue
                minx, miny, maxx, maxy = solid.bounds
                depth = min(maxx - minx, maxy - miny)
                compact = solid.area <= MAX_COMPACT_AREA_M2
                run = solid.area <= MAX_RUN_AREA_M2 and depth <= MAX_RUN_DEPTH_M
                if not (compact or run):
                    continue
                kind = _classify(solid, layer)
                objects.append({
                    "layer": layer,
                    "kind": kind,
                    "footprint": solid,
                    "area_m2": round(float(solid.area), 3),
                    "height_m": HEIGHTS[kind],
                    "height_confidence": "assumed-by-class",
                })
    return objects


def build_volumes(objects: List[Dict[str, Any]]):
    """Extrude each object's footprint to its class height. Returns a list of meshes."""
    import trimesh

    meshes = []
    for obj in objects:
        try:
            meshes.append(trimesh.creation.extrude_polygon(obj["footprint"], height=obj["height_m"]))
        except Exception:                        # noqa: BLE001 - untriangulable footprint
            continue
    return meshes


def summarise(objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Counts and areas, for the sanity check that matters.

    Furniture covering most of a room's floor means the clustering welded everything together;
    a handful of tiny objects means it shattered the symbols. Both fail silently in the mesh
    and are obvious here.
    """
    from collections import Counter

    by_kind = Counter(o["kind"] for o in objects)
    by_layer = Counter(o["layer"] for o in objects)
    areas = [o["area_m2"] for o in objects]
    return {
        "objects": len(objects),
        "total_footprint_m2": round(float(sum(areas)), 2),
        "largest_m2": round(float(max(areas)), 2) if areas else 0.0,
        "smallest_m2": round(float(min(areas)), 2) if areas else 0.0,
        "by_kind": dict(by_kind),
        "by_layer": dict(by_layer),
    }
