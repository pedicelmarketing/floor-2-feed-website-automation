"""
Derive room regions and a real-world scale from a PDF floor plan's wall linework.

The PDF has no room-outline layer -- 0-AREAS is non-printing and absent -- so rooms have to
come from the walls. Walls, doors and glazing are buffered into solid barriers, and the free
space they leave is split into regions. Each region that contains exactly one area label is a
room whose printed area can be checked against its measured one.

WHAT WORKS AND WHAT DOES NOT, measured on the real client PDF (11 pages, 212 labelled rooms):

  SCALE: solved. Fitting metres-per-point independently on each page gives 35.0-37.0 mm per
  point on 10 of 11 pages -- a ±3% spread across pages that never see each other's data. That
  is 1:100, and it contradicts the "AR-50" title-block hint of 1:50, which is why that hint is
  used only as a sanity check and never as the answer.

  ROOM SEPARATION: partly solved, and the two halves must not be blended.

  Enclosure alone isolates 110 of 212 rooms (52%). Those are good: median area error 5.9%
  against the door-verified scale, 69% of them inside 15%. Buffering the barriers more thickly
  does not raise that count -- 0.8, 1.5 and 2.5 pt merge identically -- so the openings are
  real gaps in the linework, not walls drawn too thin to close.

  Subdividing the merged regions by flooding outward from their labels lifts coverage to 183
  of 212 (86%). But the 73 rooms it adds are only good enough to say a room EXISTS and roughly
  where: median absolute area error 99.9%, and just 14% inside 15%. They are tagged
  confidence="low" and excluded from scale fitting. The cause is unlabelled circulation space
  inside the shared region -- it carries no label of its own, so it gets divided among the
  rooms that do have one.

WALL POSITION IS NOT AFFECTED BY ANY OF THIS. Measured directly against 278 doors of known
standard width, extracted geometry is accurate to 1-4 cm (see measure_pdf_accuracy.py). Room
AREA error and wall POSITION error are different quantities; only the former depends on
getting regions right.

So this module reports the two populations separately and does not pretend. Any caller
drawing conclusions from areas must read `confidence`.
"""
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Layers buffered into barriers. Walls alone leave doorways open, which merges every room on a
# floor into one blob; adding door and glazing geometry plugs most openings.
BARRIER_LAYERS = ["A-MUROS", "A-MUROS-PROYECTADO", "A-PUERTAS", "A-HTCH", "A-VIDRIO"]
# Half the drawn line weight, in points. Measured to make no difference between 0.8 and 2.5,
# so it is not a tuning knob -- see the module docstring.
BARRIER_BUFFER_PT = 0.8
MIN_REGION_PT2 = 500.0

# "A1= 5,9 mt²" / "HL= 7,80mt²" / "S/C= 35 mt²" -- comma decimals, optional space, m2 or mt2.
AREA_LABEL = re.compile(r'^(.{0,8}?)=\s*([\d]+[,\.]?[\d]*)\s*m')

# A page whose independently fitted scale disagrees with the document consensus by more than
# this has failed, not discovered a differently-scaled page. Page 11 of the client file fits
# 7.9 mm/pt against a consensus of ~36 and produces 95% area errors.
SCALE_OUTLIER_RATIO = 1.5


def area_labels(page: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Room labels that state their own area, with positions."""
    out = []
    for text in page["texts"]:
        match = AREA_LABEL.search(text["content"])
        if match:
            out.append({
                "name": match.group(1).strip(),
                "printed_area_m2": float(match.group(2).replace(",", ".")),
                "pos": text["pos"],
            })
    return out


def regions_for_page(page: Dict[str, Any],
                     barrier_layers: List[str] = None,
                     buffer_pt: float = BARRIER_BUFFER_PT,
                     min_region_pt2: float = MIN_REGION_PT2) -> Dict[str, Any]:
    """
    Split the plan into free-space regions and attach each area label to the one containing it.

    Returns the cleanly separated rooms AND the merged ones. Reporting only the clean rooms
    would make the method look far better than it is: on this file that is half the rooms.
    """
    from shapely.geometry import LineString, Point
    from shapely.ops import unary_union

    solids = []
    for layer in (barrier_layers or BARRIER_LAYERS):
        for polyline in page["layers"].get(layer, []):
            if len(polyline) >= 2:
                try:
                    solids.append(LineString(polyline).buffer(buffer_pt))
                except Exception:              # noqa: BLE001 - degenerate path, skip
                    continue
    if not solids:
        return {"regions": [], "clean": [], "merged": [], "labels": area_labels(page),
                "reason": "no barrier geometry on this page"}

    barriers = unary_union(solids)
    hull = barriers.convex_hull
    free = hull.difference(barriers)
    candidates = [g for g in getattr(free, "geoms", [free]) if g.area > min_region_pt2]

    # Drop whatever touches the hull edge. That piece is the OUTSIDE -- the site around the
    # block, the gaps between wings, the page margin -- and because it wraps the building it
    # contains a great many labels. Keeping it is harmless while only single-label regions are
    # used, which is why it went unnoticed, but it is catastrophic once merged regions get
    # subdivided: the exterior is then split among its labels and every room inherits a slice
    # of the site. Measured, that took the fitted scale to 8-30 mm/pt against a true 36.
    rim = hull.exterior.buffer(buffer_pt * 2.0)
    regions = [g for g in candidates if not g.intersects(rim)]
    if not regions:                              # every region touched the rim; keep them all
        regions = candidates                     # rather than silently reporting an empty plan

    labels = area_labels(page)
    occupants: Dict[int, List[Dict[str, Any]]] = {}
    for index, region in enumerate(regions):
        for label in labels:
            if region.contains(Point(label["pos"])):
                occupants.setdefault(index, []).append(label)

    clean, merged = [], []
    for index, found in occupants.items():
        if len(found) == 1:
            clean.append({**found[0], "region_area_pt2": float(regions[index].area)})
        else:
            merged.append({"region_area_pt2": float(regions[index].area),
                           "labels": [f["name"] for f in found]})

    return {"regions": regions, "clean": clean, "merged": merged, "labels": labels,
            "unmatched": len(labels) - sum(len(v) for v in occupants.values())}


# Grid resolution for the watershed, in pixels per PDF point. At 36.1 mm per point, 2 px/pt
# is about 1.8 cm per pixel -- finer than the 1-4 cm the geometry itself is accurate to, so
# the raster is not the limiting factor.
PIXELS_PER_PT = 2.0


def rooms_by_watershed(page: Dict[str, Any],
                       barrier_layers: List[str] = None,
                       buffer_pt: float = BARRIER_BUFFER_PT,
                       pixels_per_pt: float = PIXELS_PER_PT,
                       min_region_pt2: float = MIN_REGION_PT2) -> Dict[str, Any]:
    """
    Assign every point of free space to the nearest room label, travelling through free space.

    This exists because splitting rooms by enclosure does not work on this drawing. Walls,
    doors and glazing buffered into barriers still leave 48% of rooms joined to a neighbour,
    and thicker buffers do not help -- 0.8, 1.5 and 2.5 pt all merge identically -- so the
    openings are real gaps, not thin walls. Waiting for the linework to close is waiting for
    something that is not going to happen.

    Growing outward from the labels instead sidesteps closure completely. Two rooms joined
    through a doorway are separated where their two claims meet, and because the passage is
    the narrowest part of the join, that boundary lands in the doorway -- which is exactly
    where a room ends. Nothing has to be closed for it to work.

    The label is doing real work here: it asserts "there is one room at this point", which is
    the fact enclosure was being used to infer, and it is stated on the drawing rather than
    reconstructed from it.
    """
    from scipy import ndimage
    from skimage.segmentation import watershed
    from PIL import Image, ImageDraw

    labels = area_labels(page)
    if not labels:
        return {"rooms": [], "labels": [], "reason": "no area labels on this page"}

    polylines = []
    for layer in (barrier_layers or BARRIER_LAYERS):
        for polyline in page["layers"].get(layer, []):
            if len(polyline) >= 2:
                polylines.append(np.asarray(polyline, dtype=float))
    if not polylines:
        return {"rooms": [], "labels": labels, "reason": "no barrier geometry"}

    stacked = np.vstack(polylines)
    minx, miny = stacked.min(axis=0)
    maxx, maxy = stacked.max(axis=0)
    width = int((maxx - minx) * pixels_per_pt) + 2
    height = int((maxy - miny) * pixels_per_pt) + 2

    def to_px(x, y):
        return (int((x - minx) * pixels_per_pt), int((y - miny) * pixels_per_pt))

    # Draw the barrier LINES with thickness. Do not buffer into polygons and fill them: the
    # union of buffered walls is one connected shape whose outer ring encloses the whole
    # floor, so filling that ring paints every room solid. Measured -- it dropped recovery
    # from 111 rooms to 14 while looking like a working segmentation.
    canvas = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    pen = max(1, int(round(2 * buffer_pt * pixels_per_pt)))
    for polyline in polylines:
        points = [to_px(x, y) for x, y in polyline]
        if len(points) >= 2:
            draw.line(points, fill=1, width=pen, joint="curve")
    barrier_px = np.array(canvas, dtype=bool)
    free = ~barrier_px

    markers = np.zeros((height, width), dtype=np.int32)
    seeded = []
    for i, label in enumerate(labels, start=1):
        cx, cy = to_px(*label["pos"])
        if not (0 <= cx < width and 0 <= cy < height) or not free[cy, cx]:
            continue                            # label sits on a wall or off-plan; skip it
        markers[cy, cx] = i
        seeded.append((i, label))
    if not seeded:
        return {"rooms": [], "labels": labels, "reason": "no label landed in free space"}

    # Seed the page border as "outside". Without somewhere to put it, every room also claims
    # the space beyond the building -- margins, the gap between blocks, the site around it --
    # because that space is free and has to go to the nearest label. Measured: rooms came out
    # roughly twenty times too large and the fitted scale collapsed from 36 to 6 mm/pt, while
    # room RECOVERY looked like it had improved. Reaching the border from inside a room means
    # crossing an exterior wall, so this costs a real room nothing.
    outside = len(labels) + 1
    markers[0, :] = outside
    markers[-1, :] = outside
    markers[:, 0] = outside
    markers[:, -1] = outside

    # Flooding downhill from each seed on the negated distance-to-wall means the boundary
    # between two rooms settles at the narrowest point of the passage joining them.
    elevation = -ndimage.distance_transform_edt(free)
    filled = watershed(elevation, markers, mask=free)

    px_area_pt2 = 1.0 / (pixels_per_pt ** 2)
    rooms = []
    for index, label in seeded:
        pixels = int((filled == index).sum())
        area_pt2 = pixels * px_area_pt2
        if area_pt2 < min_region_pt2:
            continue
        rooms.append({**label, "region_area_pt2": area_pt2})

    return {"rooms": rooms, "labels": labels,
            "seeded": len(seeded), "unseeded": len(labels) - len(seeded)}


def split_merged_region(region, labels_inside: List[Dict[str, Any]],
                        pixels_per_pt: float = PIXELS_PER_PT) -> List[Dict[str, Any]]:
    """
    Divide one enclosed region that contains several labels into a room per label.

    Restricted to the region on purpose. Flooding the whole page from the labels instead was
    tried and over-claims badly: a floor plan contains corridors, stairs and lift shafts that
    carry no area label, and any space without a label of its own is absorbed by the nearest
    room that has one. Measured across 11 pages, that inflated rooms enough to drag the fitted
    scale from 36 mm/pt down to 21-33 and left a 45% median area error, even after the page
    border was seeded as "outside".

    Confining the flood to an already-enclosed region removes the problem by construction:
    unlabelled corridors are their own regions, so they are never in the pool being divided.
    """
    from scipy import ndimage
    from skimage.segmentation import watershed
    from PIL import Image, ImageDraw

    minx, miny, maxx, maxy = region.bounds
    width = int((maxx - minx) * pixels_per_pt) + 2
    height = int((maxy - miny) * pixels_per_pt) + 2
    if width < 3 or height < 3:
        return []

    def to_px(x, y):
        return (int((x - minx) * pixels_per_pt), int((y - miny) * pixels_per_pt))

    canvas = Image.new("1", (width, height), 0)
    draw = ImageDraw.Draw(canvas)
    ring = [to_px(x, y) for x, y in region.exterior.coords]
    if len(ring) >= 3:
        draw.polygon(ring, fill=1)
    for interior in region.interiors:            # holes are not part of the room
        hole = [to_px(x, y) for x, y in interior.coords]
        if len(hole) >= 3:
            draw.polygon(hole, fill=0)
    inside = np.array(canvas, dtype=bool)

    markers = np.zeros((height, width), dtype=np.int32)
    seeded = []
    for i, label in enumerate(labels_inside, start=1):
        cx, cy = to_px(*label["pos"])
        if 0 <= cx < width and 0 <= cy < height and inside[cy, cx]:
            markers[cy, cx] = i
            seeded.append((i, label))
    if len(seeded) < 2:
        return []

    elevation = -ndimage.distance_transform_edt(inside)
    filled = watershed(elevation, markers, mask=inside)

    px_area_pt2 = 1.0 / (pixels_per_pt ** 2)
    # Marked low confidence, and measured to deserve it. Against the door-verified scale these
    # rooms land at a median absolute area error of 99.9%, with only 14% inside 15% -- versus
    # 5.9% and 69% for rooms enclosure isolated on its own. Splitting recovers a room's
    # EXISTENCE and roughly where it is; it does not yet recover its extent. The remaining
    # error is unlabelled space inside the shared region -- circulation between the rooms --
    # which has no label to claim it and so is divided among the rooms that do.
    return [{**label, "region_area_pt2": float((filled == index).sum()) * px_area_pt2,
             "split": True, "confidence": "low"}
            for index, label in seeded]


def rooms_for_page(page: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """
    Every labelled room on the page, using enclosure first and splitting only what it merges.

    Enclosure already isolates about half the rooms correctly; those are kept untouched. The
    rest arrive merged into shared regions, and each of those is divided among its own labels.
    Regions holding no label -- corridors, stairs, shafts -- are deliberately left out rather
    than distributed, which is what keeps rooms from inflating.
    """
    enclosed = regions_for_page(page, **kwargs)
    if not enclosed.get("regions"):
        return {"rooms": [], "labels": enclosed.get("labels", []),
                "reason": enclosed.get("reason", "no regions")}

    from shapely.geometry import Point

    rooms = [{**r, "confidence": "high"} for r in enclosed["clean"]]
    split_count = 0
    for region in enclosed["regions"]:
        inside = [l for l in enclosed["labels"] if region.contains(Point(l["pos"]))]
        if len(inside) < 2:
            continue
        produced = split_merged_region(region, inside)
        rooms.extend(produced)
        split_count += len(produced)

    return {"rooms": rooms, "labels": enclosed["labels"],
            "high_confidence": [r for r in rooms if r.get("confidence") == "high"],
            "low_confidence": [r for r in rooms if r.get("confidence") == "low"],
            "from_enclosure": len(enclosed["clean"]), "from_split": split_count,
            "unrecovered": len(enclosed["labels"]) - len(rooms)}


def fit_scale(clean_rooms: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
    """
    Least-squares metres² per point², through the origin, from rooms with a printed area.

    Overdetermined by roughly ten rooms per page, so the residual spread is a usable
    confidence signal rather than just a fit statistic.
    """
    # Low-confidence rooms are excluded: they are wrong by a median of 100% and would drag the
    # fit rather than being outvoted by it.
    clean_rooms = [r for r in clean_rooms if r.get("confidence", "high") == "high"]
    if len(clean_rooms) < 3:
        return None
    x = np.array([r["region_area_pt2"] for r in clean_rooms], dtype=float)
    y = np.array([r["printed_area_m2"] for r in clean_rooms], dtype=float)
    if not (x > 0).all():
        return None
    m2_per_pt2 = float((x * y).sum() / (x * x).sum())
    mm_per_pt = float(np.sqrt(m2_per_pt2) * 1000.0)
    residuals = 100.0 * (x * m2_per_pt2 - y) / y
    return {
        "m2_per_pt2": m2_per_pt2,
        "mm_per_pt": mm_per_pt,
        # 1 pt is 0.3528 mm of paper, so this is the drawing's stated scale denominator.
        "drawing_scale": mm_per_pt / 0.3528,
        "median_abs_residual_pct": float(np.median(np.abs(residuals))),
        "worst_abs_residual_pct": float(np.abs(residuals).max()),
        "rooms_used": len(clean_rooms),
    }


def consensus_scale(per_page: List[Optional[Dict[str, float]]]) -> Dict[str, Any]:
    """
    Document-wide scale, with pages whose own fit disagrees wildly marked as failures.

    A page is not permitted to define its own scale when it disagrees with every other page by
    more than SCALE_OUTLIER_RATIO -- that is a broken fit, and averaging it in would corrupt
    the consensus rather than being outvoted by it.
    """
    values = [p["mm_per_pt"] for p in per_page if p]
    if not values:
        return {"mm_per_pt": None, "outliers": [], "reason": "no page produced a fit"}
    median = float(np.median(values))
    outliers = [i for i, p in enumerate(per_page)
                if p and (p["mm_per_pt"] > median * SCALE_OUTLIER_RATIO
                          or p["mm_per_pt"] < median / SCALE_OUTLIER_RATIO)]
    kept = [p["mm_per_pt"] for i, p in enumerate(per_page) if p and i not in outliers]
    return {
        "mm_per_pt": float(np.median(kept)) if kept else median,
        "spread_pct": float(100.0 * (max(kept) - min(kept)) / np.median(kept)) if kept else None,
        "pages_fitted": len(kept),
        "outlier_pages": outliers,
    }
